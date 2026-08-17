import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from consts import (
    AD_FEATURE_COLS,
    CONTEXT_FEATURE_COLS,
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
from feature_engineering import add_freq_agg_features, add_hour_features, build_preprocessing_pipeline

GROUP_COLS = USER_FEATURE_COLS + AD_FEATURE_COLS


def get_model(model_name: str, **kwargs):
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", **kwargs)
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

    num_cols = [f"{c}_ctr" for c in group_cols] + [f"{c}_count" for c in group_cols] + [
        "hour_of_day",
        "day_of_week",
    ]
    cat_cols = CONTEXT_FEATURE_COLS
    feature_cols = cat_cols + num_cols
    return train_df, val_df, cat_cols, num_cols, feature_cols, fit_stats, global_ctr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--n-rows", type=int, default=SAMPLE_N_ROWS)
    parser.add_argument("--model", default="hist_gbdt", choices=["logreg", "hist_gbdt", "lightgbm"])
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = load_sample(path=args.data_path, n_rows=args.n_rows)
    validate_schema(df)
    print(f"Loaded {len(df)} rows, overall CTR = {df[LABEL_COL].mean():.4f}")

    train_df, val_df = time_based_split(df, val_frac=args.val_frac)
    train_df, val_df, cat_cols, num_cols, feature_cols, fit_stats, global_ctr = build_features(train_df, val_df)

    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    model = get_model(args.model, random_state=args.seed)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    train_model(pipeline, train_df[feature_cols], train_df[LABEL_COL])

    val_pred_proba = pipeline.predict_proba(val_df[feature_cols])[:, 1]
    metrics = evaluate(val_df[LABEL_COL], val_pred_proba)
    print_metrics(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / f"metrics_{args.model}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Bundle the pipeline with the fitted aggregate stats it depends on, so a
    # loaded artifact can score fresh raw data without needing the original
    # train_df around (see feature_engineering.apply_freq_agg_stats).
    artifact = {
        "pipeline": pipeline,
        "fit_stats": fit_stats,
        "global_ctr": global_ctr,
        "group_cols": GROUP_COLS,
    }
    save_model(artifact, MODELS_DIR / f"{args.model}.joblib")
    return metrics


if __name__ == "__main__":
    main()
