import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from consts import (
    AD_FEATURE_COLS,
    AD_HIERARCHY_PAIRS,
    ALL_CAT_FEATURE_COLS,
    CONTEXT_FEATURE_COLS,
    FM_CLASSIFIER_PARAMS,
    FM_ENCODER_PARAMS,
    GBDT_LEAF_ENCODER_PARAMS,
    HOUR_DERIVED_COLS,
    LABEL_COL,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    SAMPLE_N_ROWS,
    SGD_LOGREG_PARAMS,
    USER_FEATURE_COLS,
    VAL_FRAC,
)
from data_loader import load_sample, time_based_split, validate_schema
from evaluator import evaluate, print_metrics
from feature_engineering import (
    DEFAULT_FREQ_SHRINK_M,
    DEFAULT_HIERARCHY_M,
    FMClassifier,
    GBDTLeafEncoder,
    add_freq_agg_features,
    add_hour_features,
    build_freq_agg_fm_concat_pipeline,
    build_freq_agg_leaves_concat_pipeline,
    build_gbdt_leaves_concat_pipeline,
    build_gbdt_leaves_ohe_pipeline,
    build_hierarchy_parent_map,
    build_ohe_only_pipeline,
    build_preprocessing_pipeline,
    to_lgbm_categoricals,
)

GROUP_COLS = USER_FEATURE_COLS + AD_FEATURE_COLS


def get_model(model_name: str, **kwargs):
    """Construct an unfitted classifier for `model_name`, applying each model's tuned defaults.

    Args:
        model_name: Which model to build — one of `"logreg"`, `"hist_gbdt"`,
            `"lightgbm"`, `"fm"`, `"sgd_logreg"`.
        **kwargs: Forwarded straight to the underlying estimator's
            constructor (e.g. `random_state`, or FM-only params like
            `freq_shrink_alpha`), overriding any conflicting key in that
            model's tuned defaults dict (`FM_CLASSIFIER_PARAMS`/
            `SGD_LOGREG_PARAMS`).

    Returns:
        An unfitted sklearn-compatible classifier instance (exposes
        `.fit(X, y)`/`.predict_proba(X)`).
    """
    if model_name == "logreg":
        return LogisticRegression(max_iter=10_000, class_weight="balanced", **kwargs)
    if model_name == "hist_gbdt":
        return HistGradientBoostingClassifier(**kwargs)
    if model_name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm is not installed; pip install lightgbm to use this model") from e
        return lgb.LGBMClassifier(**kwargs)
    if model_name == "fm":
        return FMClassifier(class_weight="balanced", **{**FM_CLASSIFIER_PARAMS, **kwargs})
    if model_name == "sgd_logreg":
        # Same FMClassifier/_fit_fm_sgd machinery as "fm", with n_factors
        # pinned to 0 — the pairwise interaction term sums to 0 over an
        # empty (d, 0) v, so this is exactly first-order-only logistic
        # regression, trained via this project's own SGD loop instead of
        # sklearn's L-BFGS. Exists specifically so it's directly comparable
        # to "fm" with no hyperparameter confound between them (see
        # SGD_LOGREG_PARAMS' docstring in consts.py) — n_factors is not a
        # key in SGD_LOGREG_PARAMS, so passing it explicitly here can't
        # collide with **SGD_LOGREG_PARAMS below.
        return FMClassifier(n_factors=0, class_weight="balanced", **{**SGD_LOGREG_PARAMS, **kwargs})
    raise ValueError(f"Unknown model_name: {model_name!r}")


def train_model(model, X_train, y_train):
    """Fit `model` on `(X_train, y_train)` in place and return it.

    Args:
        model: Any sklearn-compatible estimator exposing `.fit(X, y)` — a
            bare classifier from `get_model`, or a `Pipeline` wrapping one.
        X_train: Training feature matrix, shape `(n_train, d)` — dense
            `np.ndarray` or sparse `scipy.sparse.csr_matrix` depending on
            the feature_set's preprocessor.
        y_train: Training labels, shape `(n_train,)`.

    Returns:
        The same `model` instance passed in, now fitted (`.fit` mutates it
        in place; the return value is purely for call-site convenience).
    """
    model.fit(X_train, y_train)
    return model


def save_model(artifact, path: Path) -> None:
    """Persist `artifact` to `path` via joblib, creating parent directories as needed.

    Args:
        artifact: Any joblib-picklable object — here always a dict bundling
            a fitted `Pipeline` plus `feature_set` metadata (see `main`).
        path: Destination file path.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path: Path):
    """Load an artifact previously written by `save_model`.

    Args:
        path: Path to a joblib file written by `save_model`.

    Returns:
        The deserialized object (the same dict structure `save_model` was
        given).
    """
    return joblib.load(path)


def build_features(train_df, val_df, group_cols: list[str] = GROUP_COLS):
    """Engineer features. `add_freq_agg_features` fits aggregates on `train_df`
    only (leave-one-out for train itself, straight application for val) — see
    CLAUDE.md's no-leakage rule. Returns `fit_stats`/`global_ctr` too, so they
    can be persisted alongside a trained model for scoring new raw data later.

    Args:
        train_df: Raw training split, shape `(n_train, n_raw_cols)` — must
            contain `LABEL_COL`, `HOUR_COL`, and every column in `group_cols`.
        val_df: Raw validation split, shape `(n_val, n_raw_cols)`, same
            schema as `train_df`.
        group_cols: List of categorical column names (user + ad columns) to
            target-encode via `add_freq_agg_features`.

    Returns:
        `(train_df, val_df, cat_cols, num_cols, feature_cols, fit_stats,
        global_ctr)` — `train_df`/`val_df` are shape
        `(n_train, n_raw_cols + 2 + 2 * len(group_cols))` /
        `(n_val, n_raw_cols + 2 + 2 * len(group_cols))`: `+2` for
        `hour_of_day`/`day_of_week`, `+2 * len(group_cols)` for one `_ctr`/
        `_count` column pair per entry in `group_cols`; `cat_cols`/
        `num_cols`/`feature_cols` are lists of column names; `fit_stats` is
        the dict from `feature_engineering.fit_freq_agg_stats`; `global_ctr`
        is a scalar float.
    """
    train_df = add_hour_features(train_df)
    val_df = add_hour_features(val_df)
    global_ctr = train_df[LABEL_COL].mean()
    train_df, val_df, fit_stats = add_freq_agg_features(train_df, val_df, group_cols)

    num_cols = [f"{c}_ctr" for c in group_cols] + [f"{c}_count" for c in group_cols] + HOUR_DERIVED_COLS
    cat_cols = CONTEXT_FEATURE_COLS
    feature_cols = cat_cols + num_cols
    return train_df, val_df, cat_cols, num_cols, feature_cols, fit_stats, global_ctr


def build_baseline_ohe_features(train_df, val_df):
    """`baseline_ohe` feature set: one-hot ALL raw categorical columns
    (uncapped) plus hour_of_day/day_of_week treated as categorical. No
    target-encoding, so no fit_stats/global_ctr — the fitted OneHotEncoder
    inside the ColumnTransformer (fit via Pipeline.fit on train_df only) IS
    the fit-state, same as CONTEXT_FEATURE_COLS' OHE already works today.

    Args:
        train_df: Raw training split, shape `(n_train, n_raw_cols)`.
        val_df: Raw validation split, shape `(n_val, n_raw_cols)`.

    Returns:
        `(train_df, val_df, cat_cols, feature_cols)` — `train_df`/`val_df`
        are shape `(n_train, n_raw_cols + 2)` / `(n_val, n_raw_cols + 2)`
        after adding `hour_of_day`/`day_of_week`; `cat_cols == feature_cols`,
        the list of every raw categorical column name to one-hot encode
        (length `len(ALL_CAT_FEATURE_COLS) + 2`).
    """
    train_df = add_hour_features(train_df)
    val_df = add_hour_features(val_df)
    cat_cols = ALL_CAT_FEATURE_COLS + HOUR_DERIVED_COLS
    return train_df, val_df, cat_cols, cat_cols  # cat_cols == feature_cols here


def build_gbdt_leaf_features(train_df, val_df):
    """`gbdt_leaves` feature set (the Facebook GBDT+LR technique): the GBDT
    trains directly on raw categorical columns (cast to pandas 'category'
    dtype for lightgbm's native categorical handling, categories fixed from
    train only) plus hour_of_day/day_of_week as plain numeric — letting the
    trees learn their own binning/interactions instead of reusing the
    hand-engineered freq-agg aggregates. See feature_engineering.GBDTLeafEncoder
    for the leaf-extraction + one-hot step that turns the fitted GBDT into
    input features for a downstream linear model.

    Args:
        train_df: Raw training split, shape `(n_train, n_raw_cols)`.
        val_df: Raw validation split, shape `(n_val, n_raw_cols)`.

    Returns:
        `(train_df, val_df, feature_cols)` — `train_df`/`val_df` are shape
        `(n_train, n_raw_cols + 2)` / `(n_val, n_raw_cols + 2)` after adding
        `hour_of_day`/`day_of_week` and casting `ALL_CAT_FEATURE_COLS` to
        'category' dtype (vocabulary fixed from `train_df`); `feature_cols`
        is a list of column names, length `len(ALL_CAT_FEATURE_COLS) + 2`.
    """
    train_df = add_hour_features(train_df)
    val_df = add_hour_features(val_df)
    cat_cols = ALL_CAT_FEATURE_COLS
    num_cols = HOUR_DERIVED_COLS
    train_df, val_df = to_lgbm_categoricals(train_df, val_df, cat_cols)
    feature_cols = cat_cols + num_cols
    return train_df, val_df, feature_cols


# Which --model choices each --feature-set can pair with. HistGradientBoostingClassifier
# requires dense X and raises a TypeError on sparse input, so hist_gbdt is
# excluded from every feature_set whose preprocessor can produce sparse
# output — logreg and lightgbm both accept sparse input natively, so they're
# always compatible. freq_agg_fm_concat is included with hist_gbdt despite
# its GBDT-leaf-shaped structure: FMEmbeddingEncoder.transform returns
# `X @ v_` (sparse @ dense = dense ndarray), and freq_agg's own branch is
# already dense, so the whole FeatureUnion output is dense — unlike
# gbdt_leaves_concat/freq_agg_leaves_concat, whose GBDT-leaf branch really is
# sparse one-hot leaves.
FEATURE_SET_COMPATIBLE_MODELS = {
    "freq_agg": {"logreg", "hist_gbdt", "lightgbm"},
    # "fm"/"sgd_logreg" (both FMClassifier) are restricted to baseline_ohe
    # specifically — their training/prediction math relies on x_i^2 = x_i,
    # which only holds for pure one-hot columns, not e.g. freq_agg's
    # continuous _ctr/_count columns or the concatenated/leaf-derived
    # feature sets below.
    "baseline_ohe": {"logreg", "lightgbm", "fm", "sgd_logreg"},
    "gbdt_leaves": {"logreg", "lightgbm"},
    "gbdt_leaves_ohe": {"logreg", "lightgbm"},
    "gbdt_leaves_concat": {"logreg", "lightgbm"},
    "freq_agg_leaves_concat": {"logreg", "lightgbm"},
    "freq_agg_fm_concat": {"logreg", "hist_gbdt", "lightgbm"},
}


def check_feature_set_model_compatible(feature_set: str, model_name: str) -> None:
    """Raise if `model_name` isn't a supported pairing for `feature_set`.

    Args:
        feature_set: One of `FEATURE_SET_COMPATIBLE_MODELS`' keys (str).
        model_name: One of the `--model` choices (str) to check.

    Returns:
        None. Raises `ValueError` (and `KeyError` if `feature_set` is
        unknown) rather than returning a boolean.
    """
    compatible = FEATURE_SET_COMPATIBLE_MODELS[feature_set]
    if model_name not in compatible:
        raise ValueError(
            f"--model {model_name!r} is not compatible with --feature-set {feature_set!r} "
            f"(needs dense input; compatible models: {sorted(compatible)})"
        )


def feature_set_engineering_key(feature_set: str) -> str:
    """Groups feature_sets that produce identical `engineer_features` output
    for the same (train_df, val_df) — i.e. they resolve to the same
    underlying `build_*_features` call. baseline_ohe/gbdt_leaves_ohe/
    gbdt_leaves_concat all call `build_baseline_ohe_features` with identical
    arguments (see `engineer_features` below), so they share a key. Callers
    that engineer features for multiple feature_sets (e.g. run_pipeline.py)
    can cache by this key instead of by feature_set to avoid redundant work.
    Co-located with `engineer_features` so the two can't drift out of sync.

    Args:
        feature_set: One of the `--feature-set` choices (str).

    Returns:
        str cache key — either `"baseline_ohe_family"`, `"freq_agg_family"`,
        or `feature_set` itself unchanged.
    """
    if feature_set in ("baseline_ohe", "gbdt_leaves_ohe", "gbdt_leaves_concat"):
        return "baseline_ohe_family"
    if feature_set in ("freq_agg", "freq_agg_leaves_concat", "freq_agg_fm_concat"):
        return "freq_agg_family"
    return feature_set


def engineer_features(feature_set: str, train_df, val_df):
    """Run the (seed-independent, potentially expensive) feature engineering
    for `feature_set`. Returns `(train_df, val_df, feature_cols, state)`,
    where `state` is an opaque dict of feature_set-specific data consumed by
    `build_preprocessor` and, for `freq_agg`, by the saved model artifact.
    Split out from `build_preprocessor` so callers that train multiple models
    on the same feature_set (e.g. run_pipeline.py) can engineer features once
    and build a fresh preprocessor per model.

    Args:
        feature_set: Which feature engineering path to run (str, one of the
            `--feature-set` choices).
        train_df: Raw training split, shape `(n_train, n_raw_cols)`.
        val_df: Raw validation split, shape `(n_val, n_raw_cols)`.

    Returns:
        `(train_df, val_df, feature_cols, state)` — engineered DataFrames
        (row count unchanged from the inputs; column count extended
        per-feature_set, see the `build_*_features` helper each branch
        calls); `feature_cols` is the list of column names the preprocessor/
        model should consume; `state` is a feature_set-specific dict of
        fit-state (fitted-encoder inputs, aggregate stats, etc.) consumed by
        `build_preprocessor`.
    """
    if feature_set in ("freq_agg", "freq_agg_leaves_concat", "freq_agg_fm_concat"):
        train_df, val_df, freq_cat_cols, freq_num_cols, freq_feature_cols, fit_stats, global_ctr = build_features(
            train_df, val_df
        )
        # Always compute the widest (freq_agg_leaves_concat/freq_agg_fm_concat
        # -sized) feature_cols and state here, regardless of which of the
        # three feature_sets was actually requested — build_features only
        # ADDS the _ctr/_count columns, never drops the original raw
        # categorical columns, so the raw baseline_ohe-style columns (needed
        # only by freq_agg_leaves_concat's GBDT branch / freq_agg_fm_concat's
        # FM branch, both of which train on plain one-hot rather than
        # freq_agg's own LOO-adjusted features — see
        # feature_engineering.build_freq_agg_leaves_concat_pipeline /
        # build_freq_agg_fm_concat_pipeline) are already present in
        # train_df/val_df alongside freq_agg's engineered ones, and a
        # downstream ColumnTransformer/Pipeline simply ignores any columns it
        # doesn't select by name — so plain "freq_agg" is unaffected by the
        # extra columns/state entries. This makes all three feature_sets'
        # engineer_features output byte-identical, which is what lets
        # feature_set_engineering_key group them and avoid running this
        # (expensive, groupby-heavy) build_features call more than once when
        # several of them appear in the same run (e.g. run_pipeline.py's
        # RUNS). ohe_cat_cols doubles as both the GBDT's and the FM's raw
        # one-hot input columns — same underlying column set either way.
        ohe_cat_cols = ALL_CAT_FEATURE_COLS + HOUR_DERIVED_COLS
        feature_cols = list(dict.fromkeys(freq_feature_cols + ohe_cat_cols))
        state = {
            "cat_cols": freq_cat_cols,
            "num_cols": freq_num_cols,
            "freq_cat_cols": freq_cat_cols,
            "freq_num_cols": freq_num_cols,
            "ohe_cat_cols": ohe_cat_cols,
            "fit_stats": fit_stats,
            "global_ctr": global_ctr,
        }
    elif feature_set == "baseline_ohe":
        train_df, val_df, cat_cols, feature_cols = build_baseline_ohe_features(train_df, val_df)
        state = {"cat_cols": cat_cols}
    elif feature_set == "gbdt_leaves":
        train_df, val_df, feature_cols = build_gbdt_leaf_features(train_df, val_df)
        state = {}
    elif feature_set in ("gbdt_leaves_ohe", "gbdt_leaves_concat"):
        train_df, val_df, cat_cols, feature_cols = build_baseline_ohe_features(train_df, val_df)
        state = {"cat_cols": cat_cols}
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")
    return train_df, val_df, feature_cols, state


def build_preprocessor(feature_set: str, state: dict, seed: int):
    """Build a fresh (unfit) preprocessor for `feature_set` from the `state`
    returned by `engineer_features`. Always constructs a new instance rather
    than reusing one across calls, so training multiple models on the same
    engineered DataFrames (e.g. run_pipeline.py) doesn't have them share a
    preprocessor object mutated by a prior model's `Pipeline.fit`.

    Args:
        feature_set: Which feature_set's preprocessing pipeline to build
            (str, one of the `--feature-set` choices).
        state: The state dict returned by `engineer_features` for the same
            `feature_set`.
        seed: Random seed (scalar int) forwarded to any preprocessing step
            with its own randomness — currently only the internal GBDT in
            the `gbdt_leaves*`/`freq_agg_leaves_concat` branches.

    Returns:
        An unfitted sklearn-compatible transformer (`ColumnTransformer`/
        `Pipeline`/`FeatureUnion`/`GBDTLeafEncoder`) — once fit+transformed
        on a DataFrame of `feature_cols` columns, shape `(n_rows,
        n_feature_cols)`, it produces a 2D feature matrix of shape
        `(n_rows, d)` where `d` depends on `feature_set` (dense `np.ndarray`
        or sparse `csr_matrix`; see each `build_*_pipeline` docstring in
        `feature_engineering.py` for its exact `d`).
    """
    if feature_set == "freq_agg":
        return build_preprocessing_pipeline(state["cat_cols"], state["num_cols"])
    if feature_set == "baseline_ohe":
        return build_ohe_only_pipeline(state["cat_cols"])
    if feature_set == "gbdt_leaves":
        return GBDTLeafEncoder(gbdt_params={**GBDT_LEAF_ENCODER_PARAMS, "random_state": seed})
    if feature_set == "gbdt_leaves_ohe":
        return build_gbdt_leaves_ohe_pipeline(
            state["cat_cols"], gbdt_params={**GBDT_LEAF_ENCODER_PARAMS, "random_state": seed}
        )
    if feature_set == "gbdt_leaves_concat":
        return build_gbdt_leaves_concat_pipeline(
            state["cat_cols"], gbdt_params={**GBDT_LEAF_ENCODER_PARAMS, "random_state": seed}
        )
    if feature_set == "freq_agg_leaves_concat":
        return build_freq_agg_leaves_concat_pipeline(
            state["freq_cat_cols"],
            state["freq_num_cols"],
            state["ohe_cat_cols"],
            gbdt_params={**GBDT_LEAF_ENCODER_PARAMS, "random_state": seed},
        )
    if feature_set == "freq_agg_fm_concat":
        return build_freq_agg_fm_concat_pipeline(
            state["freq_cat_cols"],
            state["freq_num_cols"],
            state["ohe_cat_cols"],
            fm_params={**FM_ENCODER_PARAMS, "random_state": seed},
        )
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def main():
    """CLI entrypoint: train one `(--model, --feature-set)` pair and persist metrics + the fitted pipeline.

    Args:
        None as Python parameters — all inputs come from `sys.argv`, parsed
        by the `argparse.ArgumentParser` below: `--data-path`, `--n-rows`
        (scalar int), `--model`, `--feature-set`, `--val-frac` (scalar float
        in `[0, 1]`), `--output-dir`, `--seed` (scalar int),
        `--freq-shrink-alpha`/`--freq-shrink-m` (scalar floats, `fm`/
        `sgd_logreg` only), `--hierarchy-beta`/`--hierarchy-m` (scalar
        floats, `fm` + `baseline_ohe` only).

    Returns:
        The `metrics` dict from `evaluator.evaluate` (also printed and
        written to `{output_dir}/metrics_{stem}.json`); the fitted pipeline
        is additionally saved to `{MODELS_DIR}/{stem}.joblib`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--n-rows", type=int, default=SAMPLE_N_ROWS)
    parser.add_argument(
        "--model", default="hist_gbdt", choices=["logreg", "hist_gbdt", "lightgbm", "fm", "sgd_logreg"]
    )
    parser.add_argument(
        "--feature-set",
        default="freq_agg",
        choices=[
            "freq_agg",
            "baseline_ohe",
            "gbdt_leaves",
            "gbdt_leaves_ohe",
            "gbdt_leaves_concat",
            "freq_agg_leaves_concat",
            "freq_agg_fm_concat",
        ],
    )
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--freq-shrink-alpha",
        type=float,
        default=0.0,
        help="fm/sgd_logreg only: frequency-adaptive L2 shrinkage strength for rare one-hot "
        "columns (0 = disabled, matches current behavior)",
    )
    parser.add_argument("--freq-shrink-m", type=float, default=DEFAULT_FREQ_SHRINK_M)
    parser.add_argument(
        "--hierarchy-beta",
        type=float,
        default=0.0,
        help="fm/sgd_logreg + baseline_ohe only: hierarchical back-off strength pulling rare "
        "site_id/app_id embeddings toward their site_domain/app_domain embedding "
        "(0 = disabled, matches current behavior)",
    )
    parser.add_argument("--hierarchy-m", type=float, default=DEFAULT_HIERARCHY_M)
    args = parser.parse_args()

    check_feature_set_model_compatible(args.feature_set, args.model)
    if args.freq_shrink_alpha > 0 and args.model not in ("fm", "sgd_logreg"):
        raise ValueError("--freq-shrink-alpha > 0 requires --model fm or sgd_logreg")
    hierarchy_enabled = args.hierarchy_beta > 0
    if hierarchy_enabled and not (args.model == "fm" and args.feature_set == "baseline_ohe"):
        # sgd_logreg is deliberately excluded here, not just baseline_ohe/fm-
        # only: it forces n_factors=0 (trainer.get_model), so the hierarchy
        # term — which only ever touches v_ (shape (d, 0) there) — would be a
        # silent no-op rather than a real regularizer. _fit_fm_sgd itself
        # also raises if this is ever bypassed via a direct FMClassifier call.
        raise ValueError("--hierarchy-beta > 0 requires --model fm (not sgd_logreg) and --feature-set baseline_ohe")

    df = load_sample(path=args.data_path, n_rows=args.n_rows)
    validate_schema(df)
    print(f"Loaded {len(df)} rows, overall CTR = {df[LABEL_COL].mean():.4f}")

    train_df, val_df = time_based_split(df, val_frac=args.val_frac)
    train_df, val_df, feature_cols, state = engineer_features(args.feature_set, train_df, val_df)
    preprocessor = build_preprocessor(args.feature_set, state, args.seed)

    model_kwargs = {"random_state": args.seed}
    if args.model in ("fm", "sgd_logreg"):
        model_kwargs["freq_shrink_alpha"] = args.freq_shrink_alpha
        model_kwargs["freq_shrink_m"] = args.freq_shrink_m

    if hierarchy_enabled:
        # Hierarchical back-off needs the fitted OneHotEncoder's categories_
        # to build the child->parent column-index map, which a single
        # Pipeline.fit() call never exposes a seam for between the
        # "preprocess" and "model" steps — so unroll them manually here.
        # Every other case (the vast majority of runs) keeps the original
        # single-pipeline.fit() path in the else branch below untouched.
        # Pipeline([...]) is reassembled from the already-fitted steps
        # afterward purely for save_model's artifact bundling below —
        # Pipeline.predict_proba/.transform only delegate to each step's own
        # fitted state, with no bookkeeping tied to whether .fit() was
        # called on the Pipeline object itself, so this is safe.
        X_train = preprocessor.fit_transform(train_df[feature_cols])
        ohe = preprocessor.named_transformers_["cat"]
        parent_idx = build_hierarchy_parent_map(train_df, state["cat_cols"], ohe, AD_HIERARCHY_PAIRS)
        model_kwargs["hierarchy_parent_idx"] = parent_idx
        model_kwargs["hierarchy_beta"] = args.hierarchy_beta
        model_kwargs["hierarchy_m"] = args.hierarchy_m
        model = get_model(args.model, **model_kwargs)
        train_model(model, X_train, train_df[LABEL_COL])
        X_val = preprocessor.transform(val_df[feature_cols])
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    else:
        model = get_model(args.model, **model_kwargs)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        train_model(pipeline, train_df[feature_cols], train_df[LABEL_COL])
        val_pred_proba = pipeline.predict_proba(val_df[feature_cols])[:, 1]

    metrics = evaluate(val_df[LABEL_COL], val_pred_proba)
    print_metrics(metrics)

    # freq_agg's filenames stay unsuffixed (backward compatible with existing
    # results/); every other feature_set gets a disambiguating suffix.
    stem = args.model if args.feature_set == "freq_agg" else f"{args.model}_{args.feature_set}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / f"metrics_{stem}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Bundle the pipeline with the fitted aggregate stats it depends on, so a
    # loaded artifact can score fresh raw data without needing the original
    # train_df around (see feature_engineering.apply_freq_agg_stats). Only
    # the freq_agg family has such aggregate fit-state — the other feature
    # sets' fitted pipeline steps (OneHotEncoder/GBDTLeafEncoder/
    # FMEmbeddingEncoder) are their own fit-state.
    artifact = {"pipeline": pipeline, "feature_set": args.feature_set}
    if args.feature_set in ("freq_agg", "freq_agg_leaves_concat", "freq_agg_fm_concat"):
        artifact.update({"fit_stats": state["fit_stats"], "global_ctr": state["global_ctr"], "group_cols": GROUP_COLS})
    save_model(artifact, MODELS_DIR / f"{stem}.joblib")
    return metrics


if __name__ == "__main__":
    main()
