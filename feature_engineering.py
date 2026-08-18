import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from consts import HOUR_COL, LABEL_COL


def build_preprocessing_pipeline(
    cat_cols: list[str], num_cols: list[str], max_categories: int = 50
) -> ColumnTransformer:
    # max_categories caps cardinality on the anonymized high-cardinality context
    # columns (C14-C21 etc.) so one-hot encoding stays a manageable size. Dense
    # output (sparse_output=False) because HistGradientBoostingClassifier in the
    # installed scikit-learn version requires dense X.
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
        self.gbdt_params = gbdt_params

    def fit(self, X, y):
        import lightgbm as lgb

        self.gbdt_ = lgb.LGBMClassifier(**(self.gbdt_params or {}))
        self.gbdt_.fit(X, y)
        leaves = self.gbdt_.predict(X, pred_leaf=True)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.ohe_.fit(leaves)
        return self

    def transform(self, X):
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
    """
    return FeatureUnion(
        [
            ("raw_freq_agg", build_preprocessing_pipeline(freq_cat_cols, freq_num_cols)),
            ("gbdt_leaves", build_gbdt_leaves_ohe_pipeline(gbdt_cat_cols, gbdt_params=gbdt_params)),
        ]
    )


class FMEmbeddingEncoder(BaseEstimator, TransformerMixin):
    """Degree-2 Factorization Machine encoder (Rendle, "Factorization
    Machines", 2010): trains a lightweight FM classifier via minibatch SGD
    directly on sparse one-hot input, then exposes each row's `n_factors`-dim
    latent projection `S = X @ v` (the sum of its active one-hot categories'
    latent vectors) as induced features for a downstream linear model — the
    FM analog of `GBDTLeafEncoder`'s leaf embeddings.

    No external FM library is used (pyfm/fastFM/xlearn are largely
    unmaintained C-extension packages — real risk of build failure on modern
    Python/ARM) — this is a from-scratch, dependency-free implementation
    sized for this project's scale (up to ~600K raw one-hot columns from
    `ALL_CAT_FEATURE_COLS`, up to a few million training rows). Update math
    exploits that one-hot entries are binary (x_i^2 = x_i), which is what
    lets the sum-of-squared-sums trick collapse the pairwise interaction term
    (and its gradient) to two sparse matrix multiplies per batch, without
    ever materializing the full pairwise interaction matrix.

    Only `fit(X_train, y_train)` ever sees labels — `transform` reuses the
    already-fit `w0_`/`w_`/`v_`, so this satisfies CLAUDE.md's no-leakage
    rule the same way the rest of this module's fit-on-train-only
    transformers do.

    `n_epochs` is an upper bound, not a target — `fit` stops early once the
    (training-loss) plateau-detection in `early_stopping_patience`/
    `early_stopping_tol` triggers, so raising `n_epochs` to explore whether
    more training helps is safe by default rather than risking silent
    overfitting on an unattended run.
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
        random_state: int | None = None,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tol = early_stopping_tol
        self.random_state = random_state

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        self.w0_ = 0.0
        self.w_ = np.zeros(d, dtype=np.float64)
        # Small random init (not zero) is required: with v=0, the latent
        # projection S=Xv is 0, which makes grad_v identically 0 too (see
        # below) — a genuine degenerate fixed point that zero-init can never
        # escape. Small random values break that symmetry.
        self.v_ = rng.normal(scale=0.01, size=(d, self.n_factors))

        n_batches = max(1, math.ceil(n / self.batch_size))
        best_loss = np.inf
        epochs_without_improvement = 0
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            epoch_losses = []
            for batch_idx in np.array_split(order, n_batches):
                Xb = X[batch_idx]
                yb = y[batch_idx]
                nb = len(batch_idx)

                # Restrict all dense (rows x n_factors) work to the columns
                # actually active in this batch, rather than the full d rows
                # (up to ~600K) — a typical row only activates ~len(cat_cols)
                # one-hot columns, so this keeps each step's cost proportional
                # to the batch's actual sparsity instead of d.
                active_cols = np.unique(Xb.indices)
                Xb_active = Xb[:, active_cols]
                v_active = self.v_[active_cols]
                w_active = self.w_[active_cols]

                s = Xb_active @ v_active  # (nb, k): sum of active categories' latent vectors
                sq_sum = Xb_active @ (v_active**2)  # (nb, k): x_i^2 == x_i for binary one-hot input
                interaction = 0.5 * np.sum(s**2 - sq_sum, axis=1)  # (nb,)
                z = self.w0_ + Xb_active @ w_active + interaction
                p = expit(z)
                p_clipped = np.clip(p, 1e-12, 1 - 1e-12)
                batch_loss = -np.mean(yb * np.log(p_clipped) + (1 - yb) * np.log(1 - p_clipped))
                epoch_losses.append((batch_loss, nb))

                err = p - yb  # (nb,) == dL/dz for mean binary cross-entropy
                g_active = Xb_active.T @ err  # (n_active,)
                # grad_v[j,f] = mean_i[ err_i * x_i,j * (s_i,f - v_j,f * x_i,j) ], vectorized via
                # the same x_i^2=x_i trick used for `interaction` above.
                grad_v_active = (Xb_active.T @ (err[:, None] * s)) / nb - v_active * (g_active / nb)[:, None]

                self.w0_ -= self.learning_rate * err.mean()
                self.w_[active_cols] -= self.learning_rate * (g_active / nb + self.l2_reg * w_active)
                self.v_[active_cols] -= self.learning_rate * (grad_v_active + self.l2_reg * v_active)
            # Sample-weighted average across batches, not a plain mean of
            # per-batch means — the last batch of an epoch is usually a
            # different size than the rest (n not evenly divisible by
            # batch_size), so an unweighted mean would misrepresent the true
            # epoch loss. This only affects the printed diagnostic, not
            # training itself (each batch's gradient step already uses that
            # batch's own correctly-normalized mean).
            losses, sizes = zip(*epoch_losses, strict=True)
            weighted_loss = np.average(losses, weights=sizes)
            print(f"[FMEmbeddingEncoder] epoch {epoch + 1}/{self.n_epochs} mean logloss: {weighted_loss:.4f}")

            # Plateau-based early stopping on training loss: guards against
            # silently wasting compute (or, if a future n_epochs bump pushes
            # well past convergence, overfitting) when nobody's watching the
            # printed per-epoch loss. Judges "improvement" against the best
            # loss seen so far (not just the previous epoch), since the loss
            # can wobble slightly upward between epochs near convergence
            # without that meaning training has actually plateaued.
            if weighted_loss < best_loss - self.early_stopping_tol:
                best_loss = weighted_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    print(
                        f"[FMEmbeddingEncoder] early stopping at epoch {epoch + 1}/{self.n_epochs} "
                        f"(no improvement > {self.early_stopping_tol} for {self.early_stopping_patience} epochs)"
                    )
                    break
        return self

    def transform(self, X):
        X = sp.csr_matrix(X)
        return X @ self.v_


def build_fm_embed_pipeline(cat_cols: list[str], fm_params: dict | None = None) -> Pipeline:
    """One-hot -> `FMEmbeddingEncoder`, mirroring `build_gbdt_leaves_ohe_pipeline`'s
    shape: the FM trains on raw uncapped one-hot vectors (same columns as
    `baseline_ohe`), and each row's latent projection is exposed as induced
    features. Reused by `build_freq_agg_fm_concat_pipeline` below.
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
    """
    return FeatureUnion(
        [
            ("raw_freq_agg", build_preprocessing_pipeline(freq_cat_cols, freq_num_cols)),
            ("fm_embed", build_fm_embed_pipeline(fm_cat_cols, fm_params=fm_params)),
        ]
    )


def add_hour_features(df: pd.DataFrame, hour_col: str = HOUR_COL) -> pd.DataFrame:
    """Parse Avazu's YYMMDDHH `hour` column into `hour_of_day` and `day_of_week`."""
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
