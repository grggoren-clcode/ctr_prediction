from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
OUTPUTS_DIR = RESULTS_DIR / "outputs"

# The real Avazu dataset lives outside the repo.
RAW_DATA_PATH = Path("/Users/gregorygoren/Documents/research/avazu-ctr-prediction/train.gz")

ID_COL = "id"
LABEL_COL = "click"
HOUR_COL = "hour"

RANDOM_SEED = 42
SAMPLE_N_ROWS = 2_000_000
VAL_FRAC = 0.2

# device_id/device_ip/device_model/device_type/device_conn_type are proxies for "user" —
# Avazu has no true user_id.
USER_FEATURE_COLS = ["device_id", "device_ip", "device_model", "device_type", "device_conn_type"]
AD_FEATURE_COLS = ["site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category"]
# hour is handled separately (drives the split + derived hour_of_day/day_of_week), so it's
# excluded here to avoid double-listing.
CONTEXT_FEATURE_COLS = ["C1", "banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]

# All raw categorical columns across the three feature groups above — used by
# the `baseline_ohe` feature set (see trainer.build_baseline_ohe_features),
# which one-hot encodes everything with no target-encoding/aggregation.
ALL_CAT_FEATURE_COLS = CONTEXT_FEATURE_COLS + USER_FEATURE_COLS + AD_FEATURE_COLS

# hour_of_day/day_of_week, as added by feature_engineering.add_hour_features.
# Named as a constant since build_features (as numeric), build_baseline_ohe_features
# (as categorical), and build_gbdt_leaf_features (as numeric) all reference them.
HOUR_DERIVED_COLS = ["hour_of_day", "day_of_week"]

# Defaults for the internal GBDT that feature_engineering.GBDTLeafEncoder fits
# to derive leaf-index features (the `gbdt_leaves`/`gbdt_leaves_ohe`/
# `gbdt_leaves_concat` feature sets). random_state is deliberately excluded
# here — trainer.py merges in args.seed at construction time, same as it does
# for get_model's random_state kwarg.
#
# Shrunk back to 100 trees/31 leaves (from a 500/64 setting that empirically
# overfit under this project's time-based split — see gbdt_leaves' AUC
# regressing from 0.7385 to 0.7096 on the full 2M-row sample when pushed to
# 500/64) plus explicit anti-overfitting regularization: min_child_samples
# raises the minimum leaf size well above lightgbm's default (20), and
# feature_fraction/bagging_fraction+bagging_freq decorrelate trees by
# subsampling columns/rows per tree, so no single tree can lean too heavily
# on a memorization-prone near-unique column (device_id/device_ip).
GBDT_LEAF_ENCODER_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}
