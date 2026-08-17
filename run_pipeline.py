"""One-command end-to-end run: load -> time-split -> engineer features -> train -> evaluate.

Trains both `logreg` and `hist_gbdt` on the same sample/split and prints a
comparison table. This is the scaffold's smoke test against the real data.
"""

import argparse
from pathlib import Path

from sklearn.pipeline import Pipeline

from consts import LABEL_COL, RANDOM_SEED, RAW_DATA_PATH, SAMPLE_N_ROWS, VAL_FRAC
from data_loader import load_sample, time_based_split, validate_schema
from evaluator import evaluate
from feature_engineering import build_preprocessing_pipeline
from trainer import build_features, get_model, train_model

MODEL_NAMES = ["logreg", "hist_gbdt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--n-rows", type=int, default=SAMPLE_N_ROWS)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = load_sample(path=args.data_path, n_rows=args.n_rows)
    validate_schema(df)
    print(f"Loaded {len(df)} rows, overall CTR = {df[LABEL_COL].mean():.4f}\n")

    train_df, val_df = time_based_split(df, val_frac=args.val_frac)
    # Feature engineering doesn't depend on model_name, so it's done once and
    # reused across models rather than redundantly recomputed per model.
    feat_train_df, feat_val_df, cat_cols, num_cols, feature_cols, _fit_stats, _global_ctr = build_features(
        train_df, val_df
    )

    results = {}
    for model_name in MODEL_NAMES:
        preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", get_model(model_name, random_state=args.seed))])
        train_model(pipeline, feat_train_df[feature_cols], feat_train_df[LABEL_COL])
        val_pred_proba = pipeline.predict_proba(feat_val_df[feature_cols])[:, 1]
        results[model_name] = evaluate(feat_val_df[LABEL_COL], val_pred_proba)

    print(f"{'model':<12}{'auc':>10}{'logloss':>10}{'pr_auc':>10}")
    for model_name, metrics in results.items():
        print(f"{model_name:<12}{metrics['auc']:>10.4f}{metrics['logloss']:>10.4f}{metrics['pr_auc']:>10.4f}")


if __name__ == "__main__":
    main()
