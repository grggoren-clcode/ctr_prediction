"""One-command end-to-end run: load -> time-split -> engineer features -> train -> evaluate.

Trains `logreg`/`hist_gbdt` on the `freq_agg` feature set, `logreg` on the
naive `baseline_ohe` feature set, and `logreg` on the GBDT-leaf-embedding
`gbdt_leaves`/`gbdt_leaves_ohe`/`gbdt_leaves_concat`/`freq_agg_leaves_concat`
feature sets (see trainer.py), all on the same sample/split, and prints a
comparison table. This is the scaffold's smoke test against the real data.
"""

import argparse
from pathlib import Path

from sklearn.pipeline import Pipeline

from consts import LABEL_COL, RANDOM_SEED, RAW_DATA_PATH, SAMPLE_N_ROWS, VAL_FRAC
from data_loader import load_sample, time_based_split, validate_schema
from evaluator import evaluate
from trainer import (
    build_preprocessor,
    check_feature_set_model_compatible,
    engineer_features,
    feature_set_engineering_key,
    get_model,
    train_model,
)

# (model_name, feature_set) pairs to compare. hist_gbdt is deliberately not
# paired with the sparse-output feature sets (baseline_ohe/gbdt_leaves/
# gbdt_leaves_ohe/gbdt_leaves_concat/freq_agg_leaves_concat): it needs dense
# input, and densifying an uncapped, high-cardinality one-hot matrix would be
# infeasible (enforced by trainer.check_feature_set_model_compatible below).
RUNS = [
    ("logreg", "freq_agg"),
    ("hist_gbdt", "freq_agg"),
    ("logreg", "baseline_ohe"),
    ("logreg", "gbdt_leaves"),
    ("logreg", "gbdt_leaves_ohe"),
    ("logreg", "gbdt_leaves_concat"),
    ("logreg", "freq_agg_leaves_concat"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--n-rows", type=int, default=SAMPLE_N_ROWS)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    for model_name, feature_set in RUNS:
        check_feature_set_model_compatible(feature_set, model_name)

    df = load_sample(path=args.data_path, n_rows=args.n_rows)
    validate_schema(df)
    print(f"Loaded {len(df)} rows, overall CTR = {df[LABEL_COL].mean():.4f}\n")

    train_df, val_df = time_based_split(df, val_frac=args.val_frac)

    # Feature engineering (the expensive part) depends only on feature_set,
    # not model_name, so each distinct feature_set's engineered DataFrames are
    # built once and cached via trainer.engineer_features — keyed by
    # trainer.feature_set_engineering_key rather than feature_set itself, since
    # baseline_ohe/gbdt_leaves_ohe/gbdt_leaves_concat all resolve to the same
    # underlying build_baseline_ohe_features call and would otherwise redo that
    # work (add_hour_features' full-DataFrame parsing) once per feature_set. A
    # fresh preprocessor is then built per run via trainer.build_preprocessor,
    # so each model gets its own unfitted transformer rather than reusing one
    # already fitted by a prior run on the same feature_set.
    engineered_by_key = {}

    def get_engineered(feature_set):
        key = feature_set_engineering_key(feature_set)
        if key not in engineered_by_key:
            engineered_by_key[key] = engineer_features(feature_set, train_df, val_df)
        return engineered_by_key[key]

    results = {}
    for model_name, feature_set in RUNS:
        feat_train_df, feat_val_df, feature_cols, state = get_engineered(feature_set)
        preprocessor = build_preprocessor(feature_set, state, args.seed)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", get_model(model_name, random_state=args.seed))])
        train_model(pipeline, feat_train_df[feature_cols], feat_train_df[LABEL_COL])
        val_pred_proba = pipeline.predict_proba(feat_val_df[feature_cols])[:, 1]
        results[f"{model_name}/{feature_set}"] = evaluate(feat_val_df[LABEL_COL], val_pred_proba)

    print(f"{'model/feature_set':<26}{'auc':>10}{'logloss':>10}{'pr_auc':>10}")
    for run_key, metrics in results.items():
        print(f"{run_key:<26}{metrics['auc']:>10.4f}{metrics['logloss']:>10.4f}{metrics['pr_auc']:>10.4f}")


if __name__ == "__main__":
    main()
