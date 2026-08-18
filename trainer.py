import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from consts import (
    AD_FEATURE_COLS,
    ALL_CAT_FEATURE_COLS,
    CONTEXT_FEATURE_COLS,
    GBDT_LEAF_ENCODER_PARAMS,
    HOUR_DERIVED_COLS,
    LABEL_COL,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    SAMPLE_N_ROWS,
    USER_FEATURE_COLS,
    VAL_FRAC,
)
from data_loader import load_sample, time_based_split, validate_schema
from evaluator import evaluate, print_metrics
from feature_engineering import (
    GBDTLeafEncoder,
    add_freq_agg_features,
    add_hour_features,
    build_gbdt_leaves_concat_pipeline,
    build_gbdt_leaves_ohe_pipeline,
    build_ohe_only_pipeline,
    build_preprocessing_pipeline,
    to_lgbm_categoricals,
)

GROUP_COLS = USER_FEATURE_COLS + AD_FEATURE_COLS


def get_model(model_name: str, **kwargs):
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
    raise ValueError(f"Unknown model_name: {model_name!r}")


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def save_model(artifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path: Path):
    return joblib.load(path)


def build_features(train_df, val_df, group_cols: list[str] = GROUP_COLS):
    """Engineer features. `add_freq_agg_features` fits aggregates on `train_df`
    only (leave-one-out for train itself, straight application for val) — see
    CLAUDE.md's no-leakage rule. Returns `fit_stats`/`global_ctr` too, so they
    can be persisted alongside a trained model for scoring new raw data later."""
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
    """
    train_df = add_hour_features(train_df)
    val_df = add_hour_features(val_df)
    cat_cols = ALL_CAT_FEATURE_COLS
    num_cols = HOUR_DERIVED_COLS
    train_df, val_df = to_lgbm_categoricals(train_df, val_df, cat_cols)
    feature_cols = cat_cols + num_cols
    return train_df, val_df, feature_cols


# Which --model choices each --feature-set can pair with. baseline_ohe/
# gbdt_leaves/gbdt_leaves_ohe/gbdt_leaves_concat all produce sparse output;
# HistGradientBoostingClassifier requires dense X and raises a TypeError on
# sparse input, so hist_gbdt is excluded from all four. logreg and lightgbm
# both accept sparse input natively.
FEATURE_SET_COMPATIBLE_MODELS = {
    "freq_agg": {"logreg", "hist_gbdt", "lightgbm"},
    "baseline_ohe": {"logreg", "lightgbm"},
    "gbdt_leaves": {"logreg", "lightgbm"},
    "gbdt_leaves_ohe": {"logreg", "lightgbm"},
    "gbdt_leaves_concat": {"logreg", "lightgbm"},
}


def check_feature_set_model_compatible(feature_set: str, model_name: str) -> None:
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
    """
    if feature_set in ("baseline_ohe", "gbdt_leaves_ohe", "gbdt_leaves_concat"):
        return "baseline_ohe_family"
    return feature_set


def engineer_features(feature_set: str, train_df, val_df):
    """Run the (seed-independent, potentially expensive) feature engineering
    for `feature_set`. Returns `(train_df, val_df, feature_cols, state)`,
    where `state` is an opaque dict of feature_set-specific data consumed by
    `build_preprocessor` and, for `freq_agg`, by the saved model artifact.
    Split out from `build_preprocessor` so callers that train multiple models
    on the same feature_set (e.g. run_pipeline.py) can engineer features once
    and build a fresh preprocessor per model.
    """
    if feature_set == "freq_agg":
        train_df, val_df, cat_cols, num_cols, feature_cols, fit_stats, global_ctr = build_features(train_df, val_df)
        state = {"cat_cols": cat_cols, "num_cols": num_cols, "fit_stats": fit_stats, "global_ctr": global_ctr}
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
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--n-rows", type=int, default=SAMPLE_N_ROWS)
    parser.add_argument("--model", default="hist_gbdt", choices=["logreg", "hist_gbdt", "lightgbm"])
    parser.add_argument(
        "--feature-set",
        default="freq_agg",
        choices=["freq_agg", "baseline_ohe", "gbdt_leaves", "gbdt_leaves_ohe", "gbdt_leaves_concat"],
    )
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    check_feature_set_model_compatible(args.feature_set, args.model)

    df = load_sample(path=args.data_path, n_rows=args.n_rows)
    validate_schema(df)
    print(f"Loaded {len(df)} rows, overall CTR = {df[LABEL_COL].mean():.4f}")

    train_df, val_df = time_based_split(df, val_frac=args.val_frac)
    train_df, val_df, feature_cols, state = engineer_features(args.feature_set, train_df, val_df)
    preprocessor = build_preprocessor(args.feature_set, state, args.seed)

    model = get_model(args.model, random_state=args.seed)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    train_model(pipeline, train_df[feature_cols], train_df[LABEL_COL])

    val_pred_proba = pipeline.predict_proba(val_df[feature_cols])[:, 1]
    metrics = evaluate(val_df[LABEL_COL], val_pred_proba)
    print_metrics(metrics)

    # freq_agg's filenames stay unsuffixed (backward compatible with existing
    # results/); every other feature_set (baseline_ohe, gbdt_leaves,
    # gbdt_leaves_ohe) gets a disambiguating suffix.
    stem = args.model if args.feature_set == "freq_agg" else f"{args.model}_{args.feature_set}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / f"metrics_{stem}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Bundle the pipeline with the fitted aggregate stats it depends on, so a
    # loaded artifact can score fresh raw data without needing the original
    # train_df around (see feature_engineering.apply_freq_agg_stats). Only
    # freq_agg has such aggregate fit-state — the other feature sets' fitted
    # pipeline steps (OneHotEncoder/GBDTLeafEncoder) are their own fit-state.
    artifact = {"pipeline": pipeline, "feature_set": args.feature_set}
    if args.feature_set == "freq_agg":
        artifact.update({"fit_stats": state["fit_stats"], "global_ctr": state["global_ctr"], "group_cols": GROUP_COLS})
    save_model(artifact, MODELS_DIR / f"{stem}.joblib")
    return metrics


if __name__ == "__main__":
    main()
