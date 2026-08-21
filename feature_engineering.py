import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from consts import HOUR_COL, LABEL_COL


def build_preprocessing_pipeline(
    cat_cols: list[str], num_cols: list[str], max_categories: int = 50
) -> ColumnTransformer:
    """Capped one-hot + standard-scaling preprocessor for the `freq_agg` feature set.

    max_categories caps cardinality on the anonymized high-cardinality context
    columns (C14-C21 etc.) so one-hot encoding stays a manageable size. Dense
    output (sparse_output=False) because HistGradientBoostingClassifier in the
    installed scikit-learn version requires dense X.

    Args:
        cat_cols: List of categorical column names to one-hot encode; each is
            a 1D column of shape `(n,)` in the input DataFrame.
        num_cols: List of numeric column names to standardize (zero mean,
            unit variance, fit on train only).
        max_categories: Cap (scalar int) on distinct categories kept per
            column in `cat_cols` — extras collapse into an "infrequent"
            bucket.

    Returns:
        An unfitted `ColumnTransformer`; once fit+transformed on a DataFrame
        of shape `(n_rows, len(cat_cols) + len(num_cols))`, produces a dense
        `np.ndarray` of shape `(n_rows, d)` where `d = sum(min(cardinality_i,
        max_categories) for i in cat_cols) + len(num_cols)`.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", max_categories=max_categories, sparse_output=False),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ]
    )


def build_ohe_only_pipeline(cat_cols: list[str]) -> ColumnTransformer:
    """Cat-only, UNCAPPED one-hot encoding for the `baseline_ohe` feature set —
    a naive baseline that one-hots every raw categorical column including
    high-cardinality device_id/device_ip/site_id/app_id, with no
    max_categories cap and no numeric branch. sparse_output=True is required
    here — dense would be infeasible at this cardinality; LogisticRegression
    (unlike HistGradientBoostingClassifier) accepts sparse input natively.
    handle_unknown="ignore" is expected to produce many all-zero blocks in val
    for the highest-cardinality columns under a time-based split — a known
    limitation of this baseline, not a leakage issue.

    Args:
        cat_cols: List of categorical column names to one-hot encode,
            uncapped — every distinct training-set value gets its own
            column; each is a 1D column of shape `(n,)` in the input
            DataFrame.

    Returns:
        An unfitted `ColumnTransformer`; once fit+transformed on a DataFrame
        of shape `(n_rows, len(cat_cols))`, produces a sparse
        `scipy.sparse.csr_matrix` of shape `(n_rows, d)` where
        `d = sum(cardinality_i for i in cat_cols)` (cardinalities as seen in
        the fit data).
    """
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)],
        sparse_threshold=1.0,
    )


def to_lgbm_categoricals(
    train_df: pd.DataFrame, val_df: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast `cat_cols` to pandas 'category' dtype for lightgbm's native
    categorical handling, with each column's categories fixed from
    `train_df` only (per CLAUDE.md's no-leakage rule) — values in `val_df`
    that never appeared in train become NaN, which lightgbm treats as
    missing rather than raising.

    Args:
        train_df: Training DataFrame, shape `(n_train, n_cols)` — the
            category vocabulary for each entry of `cat_cols` is derived from
            this DataFrame only.
        val_df: Validation DataFrame, shape `(n_val, n_cols)`, same schema
            as `train_df`.
        cat_cols: List of column names to cast to pandas 'category' dtype.

    Returns:
        `(train_df, val_df)` — same shapes as the inputs, with each column
        in `cat_cols` now 'category'-dtyped (fixed vocabulary from
        `train_df`; values in `val_df` absent from that vocabulary become
        `NaN`).
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    for col in cat_cols:
        categories = train_df[col].astype("category").cat.categories
        train_df[col] = pd.Categorical(train_df[col], categories=categories)
        val_df[col] = pd.Categorical(val_df[col], categories=categories)
    return train_df, val_df


class GBDTLeafEncoder(BaseEstimator, TransformerMixin):
    """GBDT + LR leaf-embedding transformer (the Facebook 2014 ads-CTR
    technique): fits a lightgbm GBDT on X/y, then re-encodes every row as the
    one-hot concatenation of which leaf it landed in, per tree — the GBDT
    does its own feature engineering, and a downstream linear model (e.g.
    LogisticRegression) learns weights over the induced leaf features rather
    than the raw columns. `gbdt_params` are passed straight to
    `lightgbm.LGBMClassifier` (e.g. n_estimators, num_leaves).

    Only `fit(X_train, y_train)` sees labels/leaf assignments derived from
    training data — `transform` on val/test reuses the already-fit GBDT and
    OneHotEncoder, so this satisfies CLAUDE.md's no-leakage rule the same way
    the rest of this module's fit-on-train-only functions do.
    """

    def __init__(self, gbdt_params: dict | None = None):
        """
        Args:
            gbdt_params: dict of keyword arguments forwarded to
                `lightgbm.LGBMClassifier`'s constructor (e.g.
                `{"n_estimators": 100, "num_leaves": 31, ...}`), or `None`
                for lightgbm's own defaults.
        """
        self.gbdt_params = gbdt_params

    def fit(self, X, y):
        """Fit the internal GBDT on `(X, y)`, then fit a leaf-index one-hot encoder.

        Args:
            X: Training feature matrix for the internal GBDT, shape
                `(n, d_in)` — any format `lightgbm.LGBMClassifier.fit`
                accepts (dense array, sparse matrix, or a DataFrame with
                'category'-dtype columns).
            y: Training labels, shape `(n,)`, binary 0/1.

        Returns:
            self, fitted (sets `self.gbdt_` and `self.ohe_`).
        """
        import lightgbm as lgb

        self.gbdt_ = lgb.LGBMClassifier(**(self.gbdt_params or {}))
        self.gbdt_.fit(X, y)
        leaves = self.gbdt_.predict(X, pred_leaf=True)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.ohe_.fit(leaves)
        return self

    def transform(self, X):
        """Re-encode `X` as the one-hot concatenation of which leaf each row lands in, per tree.

        Args:
            X: Feature matrix in the same format/columns as `fit`'s `X`,
                shape `(n, d_in)`.

        Returns:
            Sparse `csr_matrix` of shape `(n, d_leaf)`, where `d_leaf` is the
            total number of distinct (tree, leaf_index) pairs seen at fit
            time (at most `gbdt_.n_estimators_ * num_leaves`, one active
            column per row per tree).
        """
        leaves = self.gbdt_.predict(X, pred_leaf=True)
        return self.ohe_.transform(leaves)


def build_gbdt_leaves_ohe_pipeline(cat_cols: list[str], gbdt_params: dict | None = None) -> Pipeline:
    """`gbdt_leaves_ohe` feature set: like `GBDTLeafEncoder`, but the internal
    GBDT trains on the same uncapped one-hot vectors as `baseline_ohe`
    (via `build_ohe_only_pipeline`) instead of raw categorical columns with
    lightgbm's native categorical handling. This sidesteps lightgbm's
    max_bin cap on high-cardinality categoricals (device_id/device_ip etc.
    silently lose resolution under native handling — see gbdt_leaves'
    training warning) at the cost of a much wider, sparser GBDT input.

    Args:
        cat_cols: List of categorical column names one-hot encoded before
            being fed to the internal GBDT.
        gbdt_params: dict forwarded to `GBDTLeafEncoder`'s internal
            `LGBMClassifier`.

    Returns:
        An unfitted `Pipeline`; once fit+transformed on a DataFrame of shape
        `(n_rows, len(cat_cols))`, produces a sparse `csr_matrix` of shape
        `(n_rows, d_leaf)` (see `GBDTLeafEncoder.transform`).
    """
    return Pipeline(
        [
            ("ohe", build_ohe_only_pipeline(cat_cols)),
            ("gbdt_leaves", GBDTLeafEncoder(gbdt_params=gbdt_params)),
        ]
    )


def build_gbdt_leaves_concat_pipeline(cat_cols: list[str], gbdt_params: dict | None = None) -> FeatureUnion:
    """`gbdt_leaves_concat` feature set: concatenates the raw uncapped
    one-hot vectors (same as `baseline_ohe`) with the GBDT-leaf one-hot
    features from `build_gbdt_leaves_ohe_pipeline`, so the downstream linear
    model sees both the original linear signal AND the GBDT's induced
    non-linear interaction/binning signal — matching the Facebook GBDT+LR
    paper's actual design (leaves augment, not replace, the base features),
    unlike the leaf-only `gbdt_leaves_ohe` variant. Note: the "raw_ohe" branch
    and the OHE step inside "gbdt_leaves" each fit their own OneHotEncoder on
    the same cat_cols independently (FeatureUnion doesn't share fitted state
    across branches) — a small duplicated cost, cheap relative to the GBDT/LR
    fits themselves.

    Args:
        cat_cols: List of categorical column names, used both for the raw
            one-hot branch and (independently, via its own OneHotEncoder)
            the internal GBDT's one-hot input.
        gbdt_params: dict forwarded to the internal `GBDTLeafEncoder`.

    Returns:
        An unfitted `FeatureUnion`; once fit+transformed on a DataFrame of
        shape `(n_rows, len(cat_cols))`, produces a sparse `csr_matrix` of
        shape `(n_rows, d_ohe + d_leaf)` — horizontal concatenation of the
        raw one-hot block (`d_ohe` columns, see `build_ohe_only_pipeline`)
        and the GBDT-leaf block (`d_leaf` columns, see
        `GBDTLeafEncoder.transform`).
    """
    return FeatureUnion(
        [
            ("raw_ohe", build_ohe_only_pipeline(cat_cols)),
            ("gbdt_leaves", build_gbdt_leaves_ohe_pipeline(cat_cols, gbdt_params=gbdt_params)),
        ]
    )


def build_freq_agg_leaves_concat_pipeline(
    freq_cat_cols: list[str],
    freq_num_cols: list[str],
    gbdt_cat_cols: list[str],
    gbdt_params: dict | None = None,
) -> FeatureUnion:
    """`freq_agg_leaves_concat` feature set: concatenates `freq_agg`'s
    engineered features (capped context one-hot + smoothed target-encoded
    user/ad aggregates + hour) with GBDT-leaf one-hot features from a GBDT
    trained on raw uncapped one-hot vectors (`gbdt_cat_cols`, same as
    `gbdt_leaves_ohe`) — deliberately NOT trained on freq_agg's own
    `_ctr`/`_count` columns.

    Why not train the internal GBDT on freq_agg's own features: `_ctr` is
    computed via **leave-one-out** for train_df (each row's own label is
    subtracted before computing its smoothed CTR — see
    `add_freq_agg_features`), which introduces a small systematic train-only
    artifact: a clicked row's own `_ctr` value is nudged slightly *lower*
    than an otherwise-identical unclicked row's, purely because its own click
    was excluded from the numerator. A linear model barely notices this
    (swamped by the real cross-category signal); a second GBDT can and
    empirically does exploit it — confirmed directly: a GBDT trained on
    freq_agg's dense matrix scored AUC ~0.48 (worse than random) on held-out
    val, because the leaf splits it learned encode a train-only LOO artifact
    that doesn't exist in val, so they transfer as noise. Training the leaf
    GBDT on plain one-hot instead avoids this entirely, since raw category
    indicators carry no target-derived/self-referential information — this
    is exactly the same GBDT `gbdt_leaves_ohe` already trains and validates.

    Args:
        freq_cat_cols: Capped categorical column names for the freq_agg
            branch (`build_preprocessing_pipeline`'s `cat_cols`).
        freq_num_cols: Numeric column names (the freq_agg `_ctr`/`_count`
            plus `hour_of_day`/`day_of_week` columns) for the freq_agg
            branch.
        gbdt_cat_cols: Raw, uncapped categorical column names fed to the
            internal leaf-inducing GBDT — deliberately NOT freq_agg's own
            `_ctr`/`_count` columns (see rationale above).
        gbdt_params: dict forwarded to the internal `GBDTLeafEncoder`.

    Returns:
        An unfitted `FeatureUnion`; once fit+transformed, produces a matrix
        of shape `(n_rows, d_freq + d_leaf)` — horizontal concatenation of
        freq_agg's dense block (`d_freq` columns, see
        `build_preprocessing_pipeline`) and the GBDT-leaf block (`d_leaf`
        columns, see `GBDTLeafEncoder.transform`).
    """
    return FeatureUnion(
        [
            ("raw_freq_agg", build_preprocessing_pipeline(freq_cat_cols, freq_num_cols)),
            ("gbdt_leaves", build_gbdt_leaves_ohe_pipeline(gbdt_cat_cols, gbdt_params=gbdt_params)),
        ]
    )


DEFAULT_FREQ_SHRINK_M = 20.0
DEFAULT_HIERARCHY_M = 20.0


def _pseudo_count_weight(count: np.ndarray, m: float) -> np.ndarray:
    """Shared pseudo-count smoothing weight `m / (count + m)`, decaying from
    ~1 at `count=0` to ~0 as `count -> infinity` — the same underlying
    weighting form both `_fit_fm_sgd`'s Technique A (frequency-adaptive
    shrinkage) and Technique B (hierarchical back-off) use to scale their
    regularization strength by column rarity (previously duplicated
    inline in both places). Also the weight implicit in
    `apply_freq_agg_stats`' `(S + m*p̄)/(N+m)` smoothing formula — that
    function is left computing its combined numerator/denominator directly
    rather than calling this helper, since rewriting it via an explicit
    `rho` would introduce a `0/0` risk at `count=0` that its current form
    avoids by construction.

    Args:
        count: Per-column counts, shape `(n_active,)`.
        m: Scalar float pseudo-count smoothing constant, must be `> 0`.

    Returns:
        `np.ndarray` of shape `(n_active,)`, values in `(0, 1]`.
    """
    return m / (count + m)


def build_hierarchy_parent_map(
    train_df: pd.DataFrame,
    cat_cols: list[str],
    ohe: OneHotEncoder,
    hierarchy_pairs: list[tuple[str, str]],
) -> np.ndarray:
    """Build a `(d,)` int array mapping each one-hot column index to its
    "parent" column index for `_fit_fm_sgd`'s hierarchical back-off term, or
    -1 for columns with no parent. `cat_cols`/`ohe.categories_` must be the
    exact column list/order `ohe` was fit on (i.e. what was passed to
    `build_ohe_only_pipeline`), since column offsets are derived by walking
    them in lockstep.

    For each `(child_col, parent_col)` pair, each child *value*'s parent is
    its most frequent (mode) co-occurring parent value in `train_df` — not a
    defensive fallback for a hypothetically-dirty edge case, but a load-
    bearing choice: on the real Avazu sample, `site_domain -> site_category`
    and `app_domain -> app_category` are NOT clean 1:1 mappings (up to ~92%
    of rows touch an ambiguous domain), so `hierarchy_pairs` is expected to
    contain only the pairs verified clean enough to trust (see
    `consts.AD_HIERARCHY_PAIRS`), and the mode fallback handles the residual
    noise in those pairs gracefully. Pairs whose child or parent column isn't
    in `cat_cols` are skipped, so this degrades safely for a feature_set that
    doesn't include the full hierarchy.

    Only `train_df` is read — this is fit-on-train-only state, same pattern
    as `fit_freq_agg_stats`.

    Args:
        train_df: Training DataFrame, shape `(n_train, n_cols)` — must
            contain every column referenced in `hierarchy_pairs`.
        cat_cols: List of column names, in the exact order `ohe` was fit
            on; `len(cat_cols)` must equal `len(ohe.categories_)`.
        ohe: A fitted `sklearn.preprocessing.OneHotEncoder`. `ohe.categories_`
            is a list of 1D arrays, one per `cat_cols` entry, where
            `categories_[i]` has length equal to that column's cardinality.
        hierarchy_pairs: List of `(child_col, parent_col)` name tuples.

    Returns:
        `parent_idx`, a 1D `np.ndarray` of shape `(d,)` (int64) where
        `d = sum(len(c) for c in ohe.categories_)` is the total one-hot
        width — `parent_idx[i]` is the global one-hot column index of column
        `i`'s parent, or `-1` if column `i` has no parent (not part of any
        pair, or its mode-parent value couldn't be resolved).
    """
    # The offset walk below assumes cat_cols is in the exact order `ohe` was
    # fit on (per the docstring's ordering contract) — enforce it rather
    # than trust it silently, wherever sklearn gives us a way to check:
    # fitting a OneHotEncoder on a DataFrame records the input column order
    # as `feature_names_in_` (sklearn >=1.0). If that attribute is present
    # and disagrees with cat_cols, every offset (and therefore every parent
    # index) computed below would be silently wrong — fail loudly instead.
    feature_names_in = getattr(ohe, "feature_names_in_", None)
    if feature_names_in is not None and list(feature_names_in) != list(cat_cols):
        raise ValueError(
            f"cat_cols order {list(cat_cols)} does not match the order ohe was fit on "
            f"{list(feature_names_in)} — build_hierarchy_parent_map's offset walk requires "
            "these to match exactly."
        )

    offsets: dict[str, tuple[int, dict]] = {}
    offset = 0
    for col, categories in zip(cat_cols, ohe.categories_, strict=True):
        offsets[col] = (offset, {val: offset + i for i, val in enumerate(categories)})
        offset += len(categories)
    d = offset

    parent_idx = np.full(d, -1, dtype=np.int64)
    for child_col, parent_col in hierarchy_pairs:
        if child_col not in offsets or parent_col not in offsets:
            continue
        _child_offset, child_value_to_idx = offsets[child_col]
        _parent_offset, parent_value_to_idx = offsets[parent_col]

        def _mode_or_none(s: pd.Series):
            # value_counts() drops NaN by default, so a child value whose
            # every co-occurring parent value is NaN yields an empty Series
            # here — idxmax() would raise on that rather than the "no
            # reliable parent, leave as -1" behavior the rest of this
            # function already gives unseen/unmapped values, so guard it
            # explicitly instead of assuming every group has at least one
            # non-null parent value. Computed once (not twice) per group.
            counts = s.value_counts()
            return counts.idxmax() if not counts.empty else None

        mode_parent = train_df.groupby(child_col)[parent_col].agg(_mode_or_none)
        for child_val, parent_val in mode_parent.items():
            if parent_val is None or child_val not in child_value_to_idx or parent_val not in parent_value_to_idx:
                continue
            parent_idx[child_value_to_idx[child_val]] = parent_value_to_idx[parent_val]
    return parent_idx


def _fm_forward(X: sp.csr_matrix, w0: float, w: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The degree-2 FM forward pass — `z = w0 + Xw + interaction`, plus the
    per-row latent projection `s = Xv` (needed separately, alongside `z`,
    for `_fit_fm_sgd`'s gradient computation). Exploits `x_i^2 = x_i` for
    binary one-hot input to reduce the pairwise interaction term to two
    sparse matrix multiplies, without ever materializing the full pairwise
    interaction matrix. Shared by `_fit_fm_sgd`'s training loop and
    `FMClassifier.predict_proba` so the two can never drift out of sync —
    the same forward pass must produce the same score whether it's being
    evaluated during training or at inference time.

    Args:
        X: One-hot input matrix, shape `(n, d)` — sparse `csr_matrix`, must
            be binary 0/1 (the `x_i^2 = x_i` identity this function relies
            on requires it).
        w0: Scalar bias term (Python float).
        w: First-order weights, shape `(d,)`.
        v: Latent factor matrix, shape `(d, k)` where `k = n_factors` — one
            length-`k` embedding row per one-hot column.

    Returns:
        `(z, s)` — `z`: shape `(n,)`, the FM logit per row
        (`w0 + Xw + interaction`); `s`: shape `(n, k)`, each row's summed
        latent projection `Xv` (returned alongside `z` since the caller's
        gradient computation needs it too).
    """
    s = X @ v  # (n, k): sum of active categories' latent vectors
    sq_sum = X @ (v**2)  # (n, k): x_i^2 == x_i for binary one-hot input
    interaction = 0.5 * np.sum(s**2 - sq_sum, axis=1)  # (n,)
    z = w0 + X @ w + interaction
    return z, s


def _weighted_bce_loss(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray) -> float:
    """Weighted mean binary cross-entropy — shared by `_fit_fm_sgd`'s
    per-batch training loss and `_fm_eval_loss`'s held-out validation loss,
    so training and evaluation can never silently compute loss differently.

    Args:
        y: True binary labels, shape `(n,)`, values in `{0.0, 1.0}`.
        p: Predicted probabilities, shape `(n,)`, in `(0, 1)` (clipped
            internally to `[1e-12, 1 - 1e-12]` to avoid `log(0)`).
        sample_weight: Per-row weights, shape `(n,)`, nonnegative floats.

    Returns:
        Scalar float — the `sample_weight`-weighted mean binary
        cross-entropy.
    """
    p_clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.sum(sample_weight * (y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))) / sample_weight.sum()


def _fm_eval_loss(
    X: sp.csr_matrix, y: np.ndarray, w0: float, w: np.ndarray, v: np.ndarray, sample_weight: np.ndarray
) -> float:
    """Weighted mean binary cross-entropy of the current FM weights on
    `(X, y)` — a plain forward pass via `_fm_forward` (no gradient/active-
    columns restriction needed, this never updates any parameter), used by
    `_fit_fm_sgd` to score the held-out internal validation split each
    epoch.

    Args:
        X: One-hot input matrix, shape `(n, d)`.
        y: True binary labels, shape `(n,)`.
        w0: Scalar bias term.
        w: First-order weights, shape `(d,)`.
        v: Latent factor matrix, shape `(d, k)`.
        sample_weight: Per-row weights, shape `(n,)`.

    Returns:
        Scalar float loss (weighted mean binary cross-entropy).
    """
    z, _s = _fm_forward(X, w0, w, v)
    return _weighted_bce_loss(y, expit(z), sample_weight)


def internal_train_idx(n: int, val_frac: float) -> np.ndarray:
    """The row-index prefix `_fit_fm_sgd` treats as its internal training
    portion for a given `n`/`val_frac`, mirroring its own chronological
    split exactly (see `_fit_fm_sgd`'s docstring for why this must be a
    prefix, not a random subset) — exposed as a public helper so callers
    that need to fit something on exactly the same rows `_fit_fm_sgd` will
    train on (e.g. `trainer.py` building `hierarchy_parent_idx` before
    calling `.fit()`, so its per-child mode-parent vote is scored on the
    same rows `col_counts`' rarity weighting sees) can replicate the split
    without duplicating — and risking drifting from — this arithmetic.

    Args:
        n: Scalar int — total row count of the array `_fit_fm_sgd` will be
            called with.
        val_frac: Scalar float in `[0, 1]` — same meaning as `_fit_fm_sgd`'s
            `val_frac` parameter.

    Returns:
        1D `np.ndarray` of shape `(n - int(n * val_frac),)` (or `(n,)` if
        `val_frac <= 0`) — `np.arange(0, n - n_val)`.
    """
    n_val = int(n * val_frac) if val_frac > 0 else 0
    return np.arange(0, n - n_val)


def _fit_fm_sgd(
    X: sp.csr_matrix,
    y: np.ndarray,
    n_factors: int,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    l2_reg: float,
    early_stopping_patience: int,
    early_stopping_tol: float,
    random_state: int | None,
    sample_weight: np.ndarray | None = None,
    val_frac: float = 0.1,
    log_prefix: str = "FM",
    freq_shrink_alpha: float = 0.0,
    freq_shrink_m: float = DEFAULT_FREQ_SHRINK_M,
    hierarchy_parent_idx: np.ndarray | None = None,
    hierarchy_beta: float = 0.0,
    hierarchy_m: float = DEFAULT_HIERARCHY_M,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minibatch-SGD training loop shared by `FMEmbeddingEncoder` and
    `FMClassifier` — both fit the exact same degree-2 FM
    (`w0 + Xw + pairwise interaction`, via `_fm_forward`), they just differ
    in what they *do* with the result: `FMEmbeddingEncoder` keeps only `v`
    (as induced features for a downstream linear model), `FMClassifier`
    uses all three as its own prediction. Returns `(w0, w_, v_)` — the
    weights from the **best** (lowest held-out-loss) epoch seen during
    training, not necessarily the last one.

    `val_frac` carves an internal validation split off of whatever `(X, y)`
    this function is given, used ONLY to judge early stopping/which epoch's
    weights to keep — never trained on. This is an internal, model-selection
    split entirely local to this function; it has no relationship to the
    project's outer time-based train/val split (`data_loader.time_based_split`)
    beyond being a further subset of whatever fold was already labeled
    "train" when it reaches here, so it doesn't touch or interact with
    CLAUDE.md's no-leakage rule. It exists because training-loss-based
    early stopping (this function's original design) judges "best" using
    the same rows the model is fitting, which can keep "improving" well
    past the point where it's still helping *held-out* performance —
    confirmed directly: `FMClassifier`'s first version, selected by
    training loss, shipped a val logloss of ~0.72 on the full 2M-row
    sample, well above every other model in this project (~0.53-0.64).

    The split is **chronological, not random**: the last `val_frac` fraction
    of rows (by position in `X`) becomes the internal validation set, the
    rest is trained on — mirroring `data_loader.time_based_split` exactly,
    one level deeper. This relies on `X`'s row order still matching the
    original chronological order `time_based_split` produced (nothing
    between there and here — `add_hour_features`, `ColumnTransformer`/
    `OneHotEncoder`, etc. — reorders rows, only transforms columns). A
    *random* internal split would be the wrong comparison here: shuffling
    minibatch order within a fixed training set doesn't change what the
    model is being asked to generalize to, but shuffling which rows count
    as "held out" would — it would estimate "generalizes to a random row
    from the same time period as training" rather than "generalizes to the
    future," letting e.g. a `device_id`/`device_ip` value shared between
    the internal-train and internal-val split (because they're close in
    time) give an optimistic read on how well the model handles identities
    it will never see again in genuinely future data. Per-epoch minibatch
    order is still shuffled (via `rng.permutation`) *within* the
    chronological training portion — that's fine, since no row ever moves
    across the train/val time boundary. `val_frac=0` (or too small an `X`
    to carve any rows off, e.g. `int(n * val_frac) == 0`) disables the
    internal split and falls back to judging by training loss instead — the
    internal validation loss must have at least one row and a nonzero total
    sample weight to be usable at all.

    `sample_weight`, if given, is a per-training-row weight (e.g. from
    `sklearn.utils.class_weight.compute_sample_weight`) folded into every
    batch average/gradient (and the internal validation loss) in place of
    the plain row count. When `sample_weight=None`, it's treated as
    all-ones. A training batch whose sample weights happen to sum to
    exactly 0 (e.g. a custom `class_weight` dict assigning weight 0 to a
    class, and an unlucky batch drawn entirely from that class) is skipped
    outright rather than dividing by zero and poisoning the weights with
    NaN.

    Raises `FloatingPointError` if the monitored loss ever becomes
    non-finite (e.g. `learning_rate` too high for this data) — silently
    continuing would otherwise leave `best_w0`/`best_w`/`best_v` pinned at
    their untrained random-init values forever (`NaN < best_loss` is always
    `False`, so the "new best" snapshot condition below can never fire
    again), and the function would return an effectively-random,
    silently-broken model instead of failing loudly at the point of
    divergence.

    `freq_shrink_alpha`/`freq_shrink_m` (frequency-adaptive shrinkage) and
    `hierarchy_parent_idx`/`hierarchy_beta`/`hierarchy_m` (hierarchical
    back-off) are two independent, opt-in regularization extensions for
    addressing undertrained embeddings of rare one-hot columns — both
    default to disabled (`freq_shrink_alpha=0`, `hierarchy_parent_idx=None`
    or `hierarchy_beta=0`), reducing exactly to today's uniform-`l2_reg`
    update with no floating-point difference. See `build_hierarchy_parent_map`
    for how `hierarchy_parent_idx` (a `(d,)` array, -1 = no parent) is built.

    Args:
        X: One-hot input matrix, shape `(n, d)` — sparse `csr_matrix`,
            binary 0/1 (enforced by an internal check).
        y: Training labels, shape `(n,)`, values in `{0.0, 1.0}`.
        n_factors: Scalar int — the latent dimension `k` of `v` (`0`
            structurally disables the pairwise interaction term).
        n_epochs: Scalar int — upper bound on training epochs (early
            stopping usually halts sooner).
        batch_size: Scalar int — target number of rows per minibatch.
        learning_rate: Scalar float — SGD step size.
        l2_reg: Scalar float — base uniform L2 penalty applied to `w` and
            `v` every step.
        early_stopping_patience: Scalar int — epochs without improvement
            (beyond `early_stopping_tol`) tolerated before stopping.
        early_stopping_tol: Scalar float — minimum loss decrease counted as
            "improvement".
        random_state: Scalar int or `None` — seed for weight init and
            per-epoch minibatch shuffling.
        sample_weight: Per-row weights, shape `(n,)`, or `None` (treated as
            all-ones).
        val_frac: Scalar float in `[0, 1]` — fraction of `X`'s rows (the
            chronologically last slice, by position) carved off as the
            internal early-stopping validation split.
        log_prefix: str prepended to each printed per-epoch log line.
        freq_shrink_alpha: Scalar float — frequency-adaptive shrinkage
            strength (`0` disables Technique A).
        freq_shrink_m: Scalar float — pseudo-count smoothing constant for
            Technique A.
        hierarchy_parent_idx: `(d,)` int array (or `None`) mapping each
            column index to its parent column index, `-1` = no parent (see
            `build_hierarchy_parent_map`).
        hierarchy_beta: Scalar float — hierarchical back-off strength (`0`
            or `hierarchy_parent_idx=None` disables Technique B).
        hierarchy_m: Scalar float — pseudo-count smoothing constant for
            Technique B.

    Returns:
        `(best_w0, best_w, best_v)` — the weights from the best (lowest
        held-out-loss, or lowest train-loss if `val_frac` disables the
        internal split) epoch: `best_w0` a scalar float, `best_w` shape
        `(d,)`, `best_v` shape `(d, n_factors)`.
    """
    n, d = X.shape
    if X.data.size and not np.array_equal(np.unique(X.data), np.array([1.0])):
        raise ValueError(
            "_fit_fm_sgd requires binary (0/1) one-hot input — the x_i^2 = x_i identity "
            "the interaction term and its gradient rely on does not hold otherwise."
        )
    # Enabling a technique (alpha/beta > 0) without the state/smoothing it
    # needs must fail loudly rather than silently no-op or divide by zero —
    # mirrors the existing n_factors==0 check below, which raises for the
    # same reason instead of quietly training a plain FM.
    if freq_shrink_alpha > 0 and freq_shrink_m <= 0:
        raise ValueError(f"freq_shrink_m must be > 0 when freq_shrink_alpha > 0, got {freq_shrink_m}")
    if hierarchy_beta > 0 and hierarchy_parent_idx is None:
        raise ValueError("hierarchy_beta > 0 requires hierarchy_parent_idx to be given (not None)")
    if hierarchy_beta > 0 and hierarchy_m <= 0:
        raise ValueError(f"hierarchy_m must be > 0 when hierarchy_beta > 0, got {hierarchy_m}")
    if hierarchy_parent_idx is not None:
        if not np.issubdtype(hierarchy_parent_idx.dtype, np.integer):
            raise ValueError(f"hierarchy_parent_idx must have an integer dtype, got {hierarchy_parent_idx.dtype}")
        if hierarchy_parent_idx.shape != (d,):
            raise ValueError(f"hierarchy_parent_idx must have shape ({d},), got {hierarchy_parent_idx.shape}")
        valid = hierarchy_parent_idx == -1
        valid |= (hierarchy_parent_idx >= 0) & (hierarchy_parent_idx < d)
        if not valid.all():
            raise ValueError("hierarchy_parent_idx entries must be -1 or a valid column index in [0, d)")
        if hierarchy_beta > 0 and n_factors == 0:
            raise ValueError(
                "hierarchy_beta > 0 has no effect when n_factors=0 — the back-off term only "
                "modifies v (shape (d, 0) here), never w. Not caught earlier so this fails "
                "loudly here regardless of caller (trainer.py's CLI also rejects this combo)."
            )
    rng = np.random.default_rng(random_state)
    sample_weight = np.ones(n, dtype=np.float64) if sample_weight is None else sample_weight

    # Chronological split (last val_frac fraction of rows, by position) —
    # see the docstring above for why this must not be a random split.
    # internal_train_idx is exposed so callers that need to fit something on
    # exactly the same rows (e.g. trainer.py building hierarchy_parent_idx
    # before calling .fit()) can replicate this split without duplicating
    # the arithmetic.
    train_idx = internal_train_idx(n, val_frac)
    n_train = len(train_idx)
    n_val = n - n_train
    val_idx = np.arange(n_train, n)
    use_val = n_val > 0 and sample_weight[val_idx].sum() > 0
    if use_val:
        X_val, y_val, sw_val = X[val_idx], y[val_idx], sample_weight[val_idx]
    if n_train == 0:
        raise ValueError(f"val_frac={val_frac} leaves no rows to train on (n_train=0) — reduce val_frac.")

    # Per-column training-set activation counts, needed by both Technique A
    # (frequency-adaptive shrinkage) and Technique B (hierarchical back-off)
    # — computed once here, not per-batch. X is binary 0/1, so a column sum
    # over rows is exactly that column's training activation count. Only
    # computed from the train_idx prefix (never val_idx), matching the rest
    # of this function's train-only fit-state discipline. X[:n_train]
    # (a contiguous slice) rather than X[train_idx] (fancy indexing that
    # would force a full copy) — train_idx is always the prefix
    # np.arange(0, n_train) by construction, so the two are equivalent.
    need_col_counts = freq_shrink_alpha > 0 or (hierarchy_parent_idx is not None and hierarchy_beta > 0)
    col_counts = np.asarray(X[:n_train].sum(axis=0)).ravel() if need_col_counts else None

    w0 = 0.0
    w_ = np.zeros(d, dtype=np.float64)
    # Small random init (not zero) is required: with v=0, the latent
    # projection S=Xv is 0, which makes grad_v identically 0 too (see
    # below) — a genuine degenerate fixed point that zero-init can never
    # escape. Small random values break that symmetry.
    v_ = rng.normal(scale=0.01, size=(d, n_factors))
    best_w0, best_w, best_v = w0, w_.copy(), v_.copy()

    n_batches = max(1, math.ceil(n_train / batch_size))
    best_loss = np.inf
    epochs_without_improvement = 0
    for epoch in range(n_epochs):
        order = train_idx[rng.permutation(n_train)]
        epoch_losses = []
        for batch_idx in np.array_split(order, n_batches):
            Xb = X[batch_idx]
            yb = y[batch_idx]
            swb = sample_weight[batch_idx]
            denom = swb.sum()
            if denom == 0:
                continue

            # Restrict all dense (rows x n_factors) work to the columns
            # actually active in this batch, rather than the full d rows
            # (up to ~600K) — a typical row only activates ~len(cat_cols)
            # one-hot columns, so this keeps each step's cost proportional
            # to the batch's actual sparsity instead of d.
            active_cols = np.unique(Xb.indices)
            Xb_active = Xb[:, active_cols]
            v_active = v_[active_cols]
            w_active = w_[active_cols]

            z, s = _fm_forward(Xb_active, w0, w_active, v_active)
            p = expit(z)
            batch_loss = _weighted_bce_loss(yb, p, swb)
            epoch_losses.append((batch_loss, denom))

            err = swb * (p - yb)  # (nb,) == weighted dL/dz for mean binary cross-entropy
            g_active = Xb_active.T @ err  # (n_active,)
            # grad_v[j,f] = sum_i[ err_i * x_i,j * (s_i,f - v_j,f * x_i,j) ] / denom, vectorized
            # via the same x_i^2=x_i trick used inside _fm_forward.
            grad_v_active = (Xb_active.T @ (err[:, None] * s)) / denom - v_active * (g_active / denom)[:, None]

            # NOTE for anyone extending l2_eff_w/l2_eff_v/grad_hier_active
            # below: each is deliberately EITHER a bare Python float (the
            # disabled-by-default path, zero extra allocation) OR an
            # (n_active,)/(n_active, 1)/(n_active, k) ndarray (the enabled
            # path) — never assume one or the other. numpy broadcasting
            # makes both cases work in the final update line unmodified, but
            # any new code touching these three variables directly (e.g.
            # indexing into them) must handle both types.

            # Technique A: frequency-adaptive shrinkage — per-column L2
            # strength scaled up for rarely-seen columns (rho -> 1 as
            # col_counts -> 0), mirroring apply_freq_agg_stats' additive-
            # smoothing formula but applied to shrinking embedding
            # parameters toward zero rather than smoothing a target-encoded
            # scalar. l2_eff_w/l2_eff_v stay the bare scalar l2_reg (no
            # allocation, byte-identical to the pre-existing update) unless
            # freq_shrink_alpha>0 — this is the disabled-by-default path, so
            # every existing fm/sgd_logreg run pays no extra cost for a
            # feature it never opts into. l2_eff_v is reshaped to
            # (n_active, 1) only in the array case, since a scalar already
            # broadcasts against v_active's (n_active, k) with no reshape.
            l2_eff_w = l2_reg
            l2_eff_v = l2_reg
            if col_counts is not None and freq_shrink_alpha > 0:
                rho_shrink = _pseudo_count_weight(col_counts[active_cols], freq_shrink_m)
                l2_eff_w = l2_reg * (1.0 + freq_shrink_alpha * rho_shrink)
                l2_eff_v = l2_eff_w[:, None]

            # Technique B: hierarchical back-off — pulls a rare child
            # column's v toward its (better-trained) parent's current v
            # instead of toward zero, weighted by how little training data
            # supports that specific child (same rarity-smoothing form as
            # Technique A, its own independent hyperparameters). One-
            # directional: no gradient flows back into the parent's own v
            # row (v_ is read here, before this batch's in-place update
            # below, so this is always a pre-update snapshot) — a rare
            # child's noise can never perturb its well-trained parent.
            # Stays the scalar 0.0 (no allocation) unless hierarchy_beta>0
            # and at least one active column in this batch has a parent.
            grad_hier_active = 0.0
            if hierarchy_parent_idx is not None and hierarchy_beta > 0 and col_counts is not None:
                parent_active = hierarchy_parent_idx[active_cols]
                has_parent = parent_active != -1
                if has_parent.any():
                    grad_hier_active = np.zeros_like(v_active)
                    rho_hier = _pseudo_count_weight(col_counts[active_cols][has_parent], hierarchy_m)
                    grad_hier_active[has_parent] = hierarchy_beta * rho_hier[:, None] * (
                        v_active[has_parent] - v_[parent_active[has_parent]]
                    )

            w0 -= learning_rate * (err.sum() / denom)
            w_[active_cols] -= learning_rate * (g_active / denom + l2_eff_w * w_active)
            v_[active_cols] -= learning_rate * (grad_v_active + l2_eff_v * v_active + grad_hier_active)

        if not epoch_losses:
            # Every batch this epoch had zero total sample weight (only
            # possible with a pathological custom class_weight) — nothing
            # to log or evaluate for early stopping, move on.
            continue
        # Sample-weighted average across batches, not a plain mean of
        # per-batch means — the last batch of an epoch is usually a
        # different size than the rest (n not evenly divisible by
        # batch_size), so an unweighted mean would misrepresent the true
        # epoch loss. This only affects the printed diagnostic, not
        # training itself (each batch's gradient step already uses that
        # batch's own correctly-normalized mean).
        losses, sizes = zip(*epoch_losses, strict=True)
        train_loss = np.average(losses, weights=sizes)

        if use_val:
            val_loss = _fm_eval_loss(X_val, y_val, w0, w_, v_, sw_val)
            monitored_loss = val_loss
            print(f"[{log_prefix}] epoch {epoch + 1}/{n_epochs} train logloss: {train_loss:.4f}  val logloss: {val_loss:.4f}")
        else:
            monitored_loss = train_loss
            print(f"[{log_prefix}] epoch {epoch + 1}/{n_epochs} train logloss: {train_loss:.4f}")

        # Check train_loss too, not just monitored_loss: with an internal
        # val split, a divergent column that only appears in the training
        # portion (e.g. a rare one-hot category absent from the
        # chronologically-last val_frac slice) could blow up train_loss
        # while leaving val_loss — computed on different rows — untouched,
        # letting NaN-corrupted weights slip through the snapshot check
        # below undetected.
        if not np.isfinite(train_loss) or not np.isfinite(monitored_loss):
            raise FloatingPointError(
                f"[{log_prefix}] training diverged (non-finite loss) at epoch {epoch + 1}/{n_epochs} — "
                "try a lower learning_rate or higher l2_reg."
            )

        # Plateau-based early stopping: guards against silently wasting
        # compute (or, if a future n_epochs bump pushes well past
        # convergence, overfitting) when nobody's watching the printed
        # per-epoch loss. Judges "improvement" against the best monitored
        # loss seen so far (not just the previous epoch), since the loss
        # can wobble slightly upward between epochs near convergence
        # without that meaning training has actually plateaued. Whenever a
        # new best is reached, snapshot the weights that achieved it — the
        # function returns this snapshot, not whatever epoch training
        # happens to end on.
        if monitored_loss < best_loss - early_stopping_tol:
            best_loss = monitored_loss
            best_w0, best_w, best_v = w0, w_.copy(), v_.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"[{log_prefix}] early stopping at epoch {epoch + 1}/{n_epochs} "
                    f"(no improvement > {early_stopping_tol} for {early_stopping_patience} epochs)"
                )
                break
    return best_w0, best_w, best_v


class FMEmbeddingEncoder(BaseEstimator, TransformerMixin):
    """Degree-2 Factorization Machine encoder (Rendle, "Factorization
    Machines", 2010): trains a lightweight FM classifier via minibatch SGD
    directly on sparse one-hot input (see `_fit_fm_sgd`), then exposes each
    row's `n_factors`-dim latent projection `S = X @ v` (the sum of its
    active one-hot categories' latent vectors) as induced features for a
    downstream linear model — the FM analog of `GBDTLeafEncoder`'s leaf
    embeddings. For a variant that uses the FM as a standalone classifier
    in its own right (exposing the full `w0 + Xw + interaction` logit
    instead of only `v`), see `FMClassifier` below.

    No external FM library is used (pyfm/fastFM/xlearn are largely
    unmaintained C-extension packages — real risk of build failure on modern
    Python/ARM) — this is a from-scratch, dependency-free implementation
    sized for this project's scale (up to ~600K raw one-hot columns from
    `ALL_CAT_FEATURE_COLS`, up to a few million training rows).

    Only `fit(X_train, y_train)` ever sees labels — `transform` reuses the
    already-fit `w0_`/`w_`/`v_`, so this satisfies CLAUDE.md's no-leakage
    rule the same way the rest of this module's fit-on-train-only
    transformers do.

    `n_epochs` is an upper bound, not a target — `fit` stops early once the
    plateau-detection in `early_stopping_patience`/`early_stopping_tol`
    triggers, judged against an internal, chronologically-held-out `val_frac`
    slice of the training data rather than training loss itself (see
    `_fit_fm_sgd`'s docstring for why a time-based, not random, internal
    split matters here) — so raising `n_epochs` to explore whether more
    training helps is safe by default rather than risking silent
    overfitting on an unattended run.

    This class and `FMClassifier` intentionally duplicate the same SGD
    hyperparameters in their `__init__` signatures rather than sharing one
    via inheritance — sklearn's `get_params()`/`clone()` introspect each
    estimator's own `__init__` signature, so keeping every parameter
    literal and explicit here (the sklearn-recommended pattern) is more
    reliable than a shared base `__init__`. If you change a default here,
    check whether `FMClassifier`'s should change too.
    """

    def __init__(
        self,
        n_factors: int = 8,
        n_epochs: int = 5,
        batch_size: int = 4096,
        learning_rate: float = 0.05,
        l2_reg: float = 1e-5,
        early_stopping_patience: int = 3,
        early_stopping_tol: float = 1e-4,
        val_frac: float = 0.1,
        random_state: int | None = None,
        freq_shrink_alpha: float = 0.0,
        freq_shrink_m: float = DEFAULT_FREQ_SHRINK_M,
        hierarchy_parent_idx: np.ndarray | None = None,
        hierarchy_beta: float = 0.0,
        hierarchy_m: float = DEFAULT_HIERARCHY_M,
    ):
        """
        Args:
            n_factors: Scalar int — latent dimension `k`; `self.v_` will
                have shape `(d, k)` after fitting.
            n_epochs: Scalar int — upper bound on training epochs.
            batch_size: Scalar int — target rows per SGD minibatch.
            learning_rate: Scalar float — SGD step size.
            l2_reg: Scalar float — base uniform L2 penalty.
            early_stopping_patience: Scalar int — epochs without
                improvement tolerated before stopping.
            early_stopping_tol: Scalar float — minimum loss decrease
                counted as improvement.
            val_frac: Scalar float in `[0, 1]` — fraction of training rows
                (chronologically last) held out internally for early
                stopping (see `_fit_fm_sgd`).
            random_state: Scalar int or `None` — seed for weight init and
                minibatch shuffling.
            freq_shrink_alpha: Scalar float — frequency-adaptive shrinkage
                strength (`0` disables).
            freq_shrink_m: Scalar float — pseudo-count smoothing constant
                for shrinkage.
            hierarchy_parent_idx: `(d,)` int array or `None` — see
                `build_hierarchy_parent_map`.
            hierarchy_beta: Scalar float — hierarchical back-off strength
                (`0` disables).
            hierarchy_m: Scalar float — pseudo-count smoothing constant for
                back-off.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tol = early_stopping_tol
        self.val_frac = val_frac
        self.random_state = random_state
        self.freq_shrink_alpha = freq_shrink_alpha
        self.freq_shrink_m = freq_shrink_m
        self.hierarchy_parent_idx = hierarchy_parent_idx
        self.hierarchy_beta = hierarchy_beta
        self.hierarchy_m = hierarchy_m

    def fit(self, X, y):
        """Fit the FM via `_fit_fm_sgd` and keep only `w0_`/`w_`/`v_`.

        Args:
            X: One-hot input matrix, shape `(n, d)` — coerced to
                `csr_matrix` internally.
            y: Labels, shape `(n,)`.

        Returns:
            self, fitted — sets `self.w0_` (scalar), `self.w_` (shape
            `(d,)`), `self.v_` (shape `(d, n_factors)`).
        """
        X = sp.csr_matrix(X)
        y = np.asarray(y, dtype=np.float64)
        self.w0_, self.w_, self.v_ = _fit_fm_sgd(
            X,
            y,
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            l2_reg=self.l2_reg,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_tol=self.early_stopping_tol,
            random_state=self.random_state,
            sample_weight=None,
            val_frac=self.val_frac,
            log_prefix="FMEmbeddingEncoder",
            freq_shrink_alpha=self.freq_shrink_alpha,
            freq_shrink_m=self.freq_shrink_m,
            hierarchy_parent_idx=self.hierarchy_parent_idx,
            hierarchy_beta=self.hierarchy_beta,
            hierarchy_m=self.hierarchy_m,
        )
        return self

    def transform(self, X):
        """Project `X` onto the fitted latent factors.

        Args:
            X: One-hot input matrix, shape `(n, d)`, matching the `d` used
                at fit time.

        Returns:
            Dense `np.ndarray` (or matrix, per `X @ v_`'s dtype), shape
            `(n, n_factors)` — `X @ self.v_`.
        """
        X = sp.csr_matrix(X)
        return X @ self.v_


class FMClassifier(BaseEstimator, ClassifierMixin):
    """Degree-2 Factorization Machine (Rendle, "Factorization Machines",
    2010) as a genuine standalone classifier — literally the textbook FM,
    `ŷ(x) = w0 + Σ w_i x_i + Σ_{i<j} ⟨v_i, v_j⟩ x_i x_j`, trained end-to-end
    as a single jointly-optimized model. Unlike `FMEmbeddingEncoder` — which
    fits the exact same objective via the same shared `_fit_fm_sgd` training
    loop, but discards `w0`/`w` and exposes only the latent projection `v`
    as induced features for a *separately*-trained `LogisticRegression` —
    this class exposes the already-computed full FM logit
    `z = w0 + Xw + interaction` directly as `predict_proba`, so there is no
    second, independently-trained linear model layered on top.

    Only usable with pure one-hot input (e.g. the `baseline_ohe` feature
    set): the `x_i^2 = x_i` identity the training/prediction math relies on
    does not hold for continuous features like `freq_agg`'s `_ctr`/`_count`
    columns — see `trainer.FEATURE_SET_COMPATIBLE_MODELS`, which restricts
    `"fm"` to `"baseline_ohe"` only, for this reason.

    `class_weight` mirrors `sklearn.linear_model.LogisticRegression`'s
    parameter of the same name (`trainer.get_model` passes
    `class_weight="balanced"` for this model, matching the `"logreg"`
    baseline, since Avazu's ~17% base CTR means an unweighted BCE loss
    under-attends to clicks) — `"balanced"` reweights each training row by
    `n_samples / (n_classes * n_samples_of_its_class)` via
    `sklearn.utils.class_weight.compute_sample_weight`, computed once over
    the full training set and threaded through every average/gradient
    normalization in `_fit_fm_sgd`. Defaults to `None` (unweighted), same as
    `LogisticRegression`'s own default.

    Binary classification only (the model and its BCE loss are inherently
    binary) — `classes_` is set from the training labels per
    `sklearn.base.ClassifierMixin` convention, and `fit` raises if more than
    two classes are present.
    """

    def __init__(
        self,
        n_factors: int = 8,
        n_epochs: int = 5,
        batch_size: int = 4096,
        learning_rate: float = 0.05,
        l2_reg: float = 1e-5,
        early_stopping_patience: int = 3,
        early_stopping_tol: float = 1e-4,
        val_frac: float = 0.1,
        class_weight: str | dict | None = None,
        random_state: int | None = None,
        freq_shrink_alpha: float = 0.0,
        freq_shrink_m: float = DEFAULT_FREQ_SHRINK_M,
        hierarchy_parent_idx: np.ndarray | None = None,
        hierarchy_beta: float = 0.0,
        hierarchy_m: float = DEFAULT_HIERARCHY_M,
    ):
        """
        Args:
            n_factors: Scalar int — latent dimension `k`; `self.v_` will
                have shape `(d, k)` after fitting.
            n_epochs: Scalar int — upper bound on training epochs.
            batch_size: Scalar int — target rows per SGD minibatch.
            learning_rate: Scalar float — SGD step size.
            l2_reg: Scalar float — base uniform L2 penalty.
            early_stopping_patience: Scalar int — epochs without
                improvement tolerated before stopping.
            early_stopping_tol: Scalar float — minimum loss decrease
                counted as improvement.
            val_frac: Scalar float in `[0, 1]` — fraction of training rows
                (chronologically last) held out internally for early
                stopping (see `_fit_fm_sgd`).
            class_weight: `"balanced"`, a `{class_label: weight}` dict, or
                `None` — mirrors `sklearn.linear_model.LogisticRegression`'s
                parameter of the same name.
            random_state: Scalar int or `None` — seed for weight init and
                minibatch shuffling.
            freq_shrink_alpha: Scalar float — frequency-adaptive shrinkage
                strength (`0` disables).
            freq_shrink_m: Scalar float — pseudo-count smoothing constant
                for shrinkage.
            hierarchy_parent_idx: `(d,)` int array or `None` — see
                `build_hierarchy_parent_map`.
            hierarchy_beta: Scalar float — hierarchical back-off strength
                (`0` disables; also requires `n_factors > 0`).
            hierarchy_m: Scalar float — pseudo-count smoothing constant for
                back-off.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tol = early_stopping_tol
        self.val_frac = val_frac
        self.class_weight = class_weight
        self.random_state = random_state
        self.freq_shrink_alpha = freq_shrink_alpha
        self.freq_shrink_m = freq_shrink_m
        self.hierarchy_parent_idx = hierarchy_parent_idx
        self.hierarchy_beta = hierarchy_beta
        self.hierarchy_m = hierarchy_m

    def fit(self, X, y):
        """Fit the FM end-to-end via `_fit_fm_sgd`, keeping `w0_`/`w_`/`v_` as this classifier's own weights.

        Args:
            X: One-hot input matrix, shape `(n, d)` — coerced to
                `csr_matrix` internally.
            y: Labels, shape `(n,)` — any two distinct values (mapped
                internally to canonical `{0, 1}` via `classes_`).

        Returns:
            self, fitted — sets `self.classes_` (shape `(2,)`), `self.w0_`
            (scalar), `self.w_` (shape `(d,)`), `self.v_` (shape
            `(d, n_factors)`).
        """
        X = sp.csr_matrix(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(f"FMClassifier only supports binary classification, got classes={self.classes_}")
        # Map to canonical {0.0, 1.0} via classes_ rather than assuming labels
        # are already literally 0/1 — classes_[1] (the larger of the two sorted
        # unique values) is "positive", matching sklearn's own convention that
        # predict_proba's column 1 corresponds to classes_[1].
        y_float = (y_arr == self.classes_[1]).astype(np.float64)
        sample_weight = compute_sample_weight(self.class_weight, y_arr)
        self.w0_, self.w_, self.v_ = _fit_fm_sgd(
            X,
            y_float,
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            l2_reg=self.l2_reg,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_tol=self.early_stopping_tol,
            random_state=self.random_state,
            sample_weight=sample_weight,
            val_frac=self.val_frac,
            log_prefix="FMClassifier",
            freq_shrink_alpha=self.freq_shrink_alpha,
            freq_shrink_m=self.freq_shrink_m,
            hierarchy_parent_idx=self.hierarchy_parent_idx,
            hierarchy_beta=self.hierarchy_beta,
            hierarchy_m=self.hierarchy_m,
        )
        return self

    def predict_proba(self, X):
        """Compute the FM's predicted class probabilities.

        Args:
            X: One-hot input matrix, shape `(n, d)`, matching the `d` used
                at fit time.

        Returns:
            `np.ndarray` of shape `(n, 2)` — column 0 is `P(classes_[0])`,
            column 1 is `P(classes_[1])` (`= expit(z)`, `z` from
            `_fm_forward`).
        """
        X = sp.csr_matrix(X)
        z, _s = _fm_forward(X, self.w0_, self.w_, self.v_)
        p = expit(z)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Compute the FM's predicted class labels via a 0.5 threshold on `predict_proba`.

        Args:
            X: One-hot input matrix, shape `(n, d)`.

        Returns:
            `np.ndarray` of shape `(n,)` — predicted label per row, drawn
            from `self.classes_`.
        """
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]


def build_fm_embed_pipeline(cat_cols: list[str], fm_params: dict | None = None) -> Pipeline:
    """One-hot -> `FMEmbeddingEncoder`, mirroring `build_gbdt_leaves_ohe_pipeline`'s
    shape: the FM trains on raw uncapped one-hot vectors (same columns as
    `baseline_ohe`), and each row's latent projection is exposed as induced
    features. Reused by `build_freq_agg_fm_concat_pipeline` below.

    Args:
        cat_cols: List of categorical column names to one-hot encode before
            the FM.
        fm_params: dict forwarded to `FMEmbeddingEncoder`'s constructor.

    Returns:
        An unfitted `Pipeline`; once fit+transformed, produces a dense
        matrix of shape `(n_rows, n_factors)`.
    """
    return Pipeline(
        [
            ("ohe", build_ohe_only_pipeline(cat_cols)),
            ("fm", FMEmbeddingEncoder(**(fm_params or {}))),
        ]
    )


def build_freq_agg_fm_concat_pipeline(
    freq_cat_cols: list[str],
    freq_num_cols: list[str],
    fm_cat_cols: list[str],
    fm_params: dict | None = None,
) -> FeatureUnion:
    """`freq_agg_fm_concat` feature set: concatenates `freq_agg`'s engineered
    features (capped context one-hot + smoothed target-encoded user/ad
    aggregates + hour) with `FMEmbeddingEncoder`'s latent projection features
    (via `build_fm_embed_pipeline`, trained on raw uncapped one-hot vectors)
    — mirrors `freq_agg_leaves_concat`'s design (GBDT trained on plain
    one-hot, not on freq_agg's own LOO-adjusted `_ctr`/`_count` columns), for
    the same reason: those columns carry a small train-only leave-one-out
    artifact that a flexible second model can exploit, so the induced-feature
    model should train on plain one-hot instead.

    Args:
        freq_cat_cols: Capped categorical column names for the freq_agg
            branch.
        freq_num_cols: Numeric column names for the freq_agg branch.
        fm_cat_cols: Raw categorical columns fed to the FM branch.
        fm_params: dict forwarded to `FMEmbeddingEncoder` (via
            `build_fm_embed_pipeline`).

    Returns:
        An unfitted `FeatureUnion`; once fit+transformed, produces a dense
        matrix of shape `(n_rows, d_freq + n_factors)`.
    """
    return FeatureUnion(
        [
            ("raw_freq_agg", build_preprocessing_pipeline(freq_cat_cols, freq_num_cols)),
            ("fm_embed", build_fm_embed_pipeline(fm_cat_cols, fm_params=fm_params)),
        ]
    )


def add_hour_features(df: pd.DataFrame, hour_col: str = HOUR_COL) -> pd.DataFrame:
    """Parse Avazu's YYMMDDHH `hour` column into `hour_of_day` and `day_of_week`.

    Args:
        df: DataFrame, shape `(n, n_cols)` — must contain `hour_col` with
            values in `YYMMDDHH` integer/string format.
        hour_col: Name of the column to parse.

    Returns:
        DataFrame, shape `(n, n_cols + 2)` — original columns plus
        `hour_of_day` (int, 0-23) and `day_of_week` (int, 0-6).
    """
    df = df.copy()
    hour_str = df[hour_col].astype(str)
    hour_of_day = hour_str.str[6:8].astype(int)
    dt = pd.to_datetime(
        {
            "year": 2000 + hour_str.str[0:2].astype(int),
            "month": hour_str.str[2:4].astype(int),
            "day": hour_str.str[4:6].astype(int),
        }
    )
    df["hour_of_day"] = hour_of_day
    df["day_of_week"] = dt.dt.dayofweek
    return df


DEFAULT_SMOOTHING = 20.0


def fit_freq_agg_stats(
    train_df: pd.DataFrame, group_cols: list[str], label_col: str = LABEL_COL
) -> dict[str, pd.DataFrame]:
    """Fit per-category click sum/count stats on `train_df` only.

    Returns one small DataFrame per group column (indexed by category, columns
    `sum`/`count`) — this is the explicit fit-state CLAUDE.md's no-leakage rule
    calls for: it can be persisted alongside a trained model and reapplied to
    new raw data without needing the original train_df around. Raw sum/count
    (rather than a precomputed mean) is kept so smoothing strength can be
    chosen at apply time.

    Args:
        train_df: Training DataFrame, shape `(n_train, n_cols)` — must
            contain `label_col` and every column in `group_cols`.
        group_cols: List of column names to compute per-category click
            sum/count for.
        label_col: Name of the binary label column.

    Returns:
        dict mapping each entry of `group_cols` to a DataFrame indexed by
        that column's distinct training-set values (row count = that
        column's cardinality, varies per column), with columns `"sum"`/
        `"count"`.
    """
    return {col: train_df.groupby(col)[label_col].agg(sum="sum", count="count") for col in group_cols}


def apply_freq_agg_stats(
    df: pd.DataFrame,
    fit_stats: dict[str, pd.DataFrame],
    global_ctr: float,
    smoothing: float = DEFAULT_SMOOTHING,
) -> pd.DataFrame:
    """Apply previously-fit per-category stats to `df` as a smoothed CTR:
    `(sum + smoothing * global_ctr) / (count + smoothing)`. Categories unseen
    in the fit have sum=count=0, which this formula naturally resolves to
    `global_ctr`. Smoothing pulls low-count categories toward the global rate
    instead of trusting a handful of observations outright.

    Args:
        df: DataFrame to apply stats to, shape `(n, n_cols)` — must contain
            every key of `fit_stats` as a column.
        fit_stats: dict as returned by `fit_freq_agg_stats`.
        global_ctr: Scalar float — the training-set overall click rate,
            used as the smoothing prior.
        smoothing: Scalar float — pseudo-observation count controlling how
            strongly low-count categories are pulled toward `global_ctr`.

    Returns:
        DataFrame, shape `(n, n_cols + 2 * len(fit_stats))` — original
        columns plus one `_ctr`/`_count` column pair per key in `fit_stats`.
    """
    df = df.copy()
    for col, stats in fit_stats.items():
        ctr_col, count_col = f"{col}_ctr", f"{col}_count"
        merged = df[[col]].merge(stats, on=col, how="left")
        sum_ = merged["sum"].fillna(0.0).to_numpy()
        count_ = merged["count"].fillna(0.0).to_numpy()
        df[ctr_col] = (sum_ + smoothing * global_ctr) / (count_ + smoothing)
        df[count_col] = count_
    return df


def add_freq_agg_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_cols: list[str],
    label_col: str = LABEL_COL,
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Add per-category historical CTR + count features, fit on `train_df` only.

    `val_df` gets the stats applied as-fit (its rows never contributed to them,
    so no leakage risk there). `train_df` gets a **leave-one-out + smoothed**
    application instead: each row's own label is excluded from its own group's
    sum/count, and the result is blended toward `global_ctr` by `smoothing`
    pseudo-observations. Plain leave-one-out (no smoothing) isn't enough on its
    own — for a group of size 2, one row's "leave-one-out" CTR is just the
    *other* row's raw label, which a flexible model (e.g. a GBDT) can still
    exploit to memorize training rows via high-cardinality columns like
    device_id/device_ip (confirmed empirically: unsmoothed LOO gave
    hist_gbdt train AUC 1.0 but val AUC ~0.51 — pure overfitting to
    near-unique small groups). Smoothing bounds how much any single group,
    however small, can shift its members' feature value.

    Returns `(train_df, val_df, fit_stats)` — `fit_stats` is the explicit
    fit-state from `fit_freq_agg_stats`, suitable for persisting alongside a
    trained model (see `trainer.save_model`).

    Args:
        train_df: Raw training DataFrame, shape `(n_train, n_cols)`.
        val_df: Raw validation DataFrame, shape `(n_val, n_cols)`.
        group_cols: List of column names to target-encode.
        label_col: Name of the binary label column.
        smoothing: Scalar float smoothing strength (see
            `apply_freq_agg_stats`).

    Returns:
        `(train_df, val_df, fit_stats)` — `train_df`/`val_df` are shape
        `(n_train, n_cols + 2 * len(group_cols))` /
        `(n_val, n_cols + 2 * len(group_cols))`; `fit_stats` is the dict
        from `fit_freq_agg_stats`.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    global_ctr = train_df[label_col].mean()

    fit_stats = fit_freq_agg_stats(train_df, group_cols, label_col)
    val_df = apply_freq_agg_stats(val_df, fit_stats, global_ctr, smoothing)

    for col in group_cols:
        ctr_col, count_col = f"{col}_ctr", f"{col}_count"
        grouped = train_df.groupby(col)[label_col]
        group_sum = grouped.transform("sum")
        group_count = grouped.transform("count")

        loo_sum = group_sum - train_df[label_col]
        loo_count = group_count - 1
        train_df[ctr_col] = (loo_sum + smoothing * global_ctr) / (loo_count + smoothing)
        train_df[count_col] = group_count

    return train_df, val_df, fit_stats
