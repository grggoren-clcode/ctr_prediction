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
