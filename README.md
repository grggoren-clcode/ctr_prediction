# ctr_prediction

Research/experimentation project for click-through-rate (CTR) prediction on the
[Kaggle Avazu CTR dataset](https://www.kaggle.com/c/avazu-ctr-prediction).

## Data

The real dataset lives outside this repo, at
`/Users/gregorygoren/Documents/research/avazu-ctr-prediction/` (`train.gz`, ~40M rows,
sorted by `hour`). `data_loader.load_sample()` reads a configurable leading subset
(`SAMPLE_N_ROWS` in `consts.py`, default 2,000,000 rows) for fast, notebook-driven
iteration — full-file/chunked training over all ~40M rows is a possible later step,
not part of this scaffold.

Columns: `id, click, hour, C1, banner_pos, site_id, site_domain, site_category, app_id,
app_domain, app_category, device_id, device_ip, device_model, device_type,
device_conn_type, C14-C21`. `click` is the label. `hour` is `YYMMDDHH`.

Avazu has no explicit user or ad ID, so feature groups (in `consts.py`) are proxies:

- **User-proxy**: `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type`
- **Ad/item**: `site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`
- **Context**: `C1`, `banner_pos`, `C14`-`C21`, plus derived `hour_of_day`/`day_of_week`

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```
python run_pipeline.py
```

Loads the configured sample, does a **time-based split** (train on earlier hours,
validate on later hours — this both simulates real deployment and avoids
future-into-past leakage), engineers features, trains both `logreg` and `hist_gbdt`,
and prints an AUC / LogLoss / PR-AUC comparison table.

To train and persist a single model:

```
python trainer.py --model hist_gbdt --n-rows 500000
```

This saves metrics to `results/outputs/` and the fitted pipeline to `results/models/`.

Options for both scripts: `--n-rows` (sample size), `--val-frac` (validation
fraction of the sample), `--data-path` (override the raw data location).

## Exploration

`notebooks/01_eda.ipynb` — loads a sample, checks click-rate imbalance
(Avazu's overall CTR is ~17%), and eyeballs feature distributions/signal per
candidate column.

## Extending

- New models: add a branch in `trainer.get_model`.
- New features: add functions to `feature_engineering.py`, following the
  `add_freq_agg_features` pattern — **fit on the train split only**, apply to val/test.

## Rules

See `CLAUDE.md`: code review is required before any GitHub commit, and feature
engineering must never leak train/test — see `add_freq_agg_features` in
`feature_engineering.py` for the canonical fit-on-train-only pattern.
