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

# Defaults for feature_engineering.FMEmbeddingEncoder (the `freq_agg_fm_concat`
# feature set's internal Factorization Machine, used as a feature-inducer for
# a downstream LR — NOT the final prediction itself). random_state is
# deliberately excluded — trainer.py merges in args.seed at construction
# time, same as GBDT_LEAF_ENCODER_PARAMS above.
#
# Tuned up from an initial 8-factor/5-epoch/lr=0.05 pass, which undershot —
# the printed per-epoch logloss was still dropping at epoch 5, meaning SGD
# hadn't converged. A sweep on a 100K-row loaded sample (80K-row train split
# after the usual 80/20 time-based split; measuring the FM's own raw
# prediction, not yet the downstream concat+LR result) found learning_rate
# and n_epochs to be the highest-leverage knobs: lr=0.15 with enough epochs
# to let it settle roughly doubled the FM's own val AUC (0.67 -> 0.74), while
# l2_reg and n_factors beyond ~16 gave comparatively little. n_epochs is set
# lower here (30) than the ~120 that helped on that 80K-row train split,
# since the full ~2M-row sample's ~1.6M-row train split gives roughly 20x
# more gradient updates per epoch (1.6M / 80K) — confirmed at full scale: the
# printed per-epoch loss had already flattened out well before epoch 30.
# n_epochs is only an upper bound in any case — FMEmbeddingEncoder.fit now
# stops early once its plateau-detection (early_stopping_patience/_tol)
# triggers, so this doesn't need to be retuned precisely.
#
# NOT used by FMClassifier ("fm"/"sgd_logreg" --model choices) — those have
# their own separately-tuned FM_CLASSIFIER_PARAMS/SGD_LOGREG_PARAMS below,
# since a standalone classifier's own predictive accuracy is a different
# optimization target than "produces good induced features for a downstream
# LR", and empirically needs different hyperparameters (much smaller
# batch_size in particular — see the comment below).
FM_ENCODER_PARAMS = {
    "n_factors": 16,
    "n_epochs": 30,
    "batch_size": 4096,
    "learning_rate": 0.15,
    "l2_reg": 1e-5,
}

# Defaults for feature_engineering.FMClassifier used as the standalone `fm`
# --model (full degree-2 FM: first-order + pairwise interaction, jointly
# trained) and, with n_factors forced to 0 in trainer.get_model, the
# `sgd_logreg` --model (first-order only — i.e. plain logistic regression,
# but trained via this project's own SGD loop instead of sklearn's L-BFGS,
# added specifically so it's directly comparable to "fm" with no
# hyperparameter confound between them).
#
# A /btw diagnostic found FM_ENCODER_PARAMS' batch_size=4096 (tuned for a
# different role, see above) was badly mismatched for training a standalone
# classifier: a staged hyperparameter sweep on a 100K-row sample (measuring
# real held-out AUC/logloss, not internal training loss) found smaller
# batch_size to be the single highest-leverage knob for both models — AUC
# improved substantially and monotonically as batch_size shrank from 4096
# down to roughly 128-512, before returns diminished/became noisy at the
# small-sample scale tested. Higher learning_rate also helped once batch_size
# was small enough to keep gradient steps well-behaved; l2_reg had almost no
# effect anywhere in the range tested (1e-6 to 1e-3) and is left at
# FM_ENCODER_PARAMS' existing value. Confirmed stable across 3 random seeds
# before promoting to these defaults.
#
# fm's batch_size (512) is deliberately larger than sgd_logreg's (128): the
# same sweep found fm diverges (non-finite loss) at batch_size=128 combined
# with a high learning_rate — the pairwise interaction term amplifies
# per-batch gradient noise quadratically (through v), so smaller batches
# destabilize it well before they destabilize the linear-only sgd_logreg.
FM_CLASSIFIER_PARAMS = {
    "n_factors": 16,
    "n_epochs": 60,
    "batch_size": 512,
    "learning_rate": 0.4,
    "l2_reg": 1e-5,
}
SGD_LOGREG_PARAMS = {
    "n_epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.2,
    "l2_reg": 1e-5,
}
