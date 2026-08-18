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
future-into-past leakage), engineers features, trains every `(model, feature_set)`
pair in `run_pipeline.RUNS`, and prints an AUC / LogLoss / PR-AUC comparison table.

To train and persist a single model:

```
python trainer.py --model hist_gbdt --feature-set freq_agg --n-rows 500000
```

This saves metrics to `results/outputs/` and the fitted pipeline to `results/models/`
(filenames are `{model}.joblib`/`metrics_{model}.json` for the default `freq_agg`
feature set, `{model}_{feature_set}.joblib`/`metrics_{model}_{feature_set}.json`
otherwise).

Options for both scripts: `--n-rows` (sample size), `--val-frac` (validation
fraction of the sample), `--data-path` (override the raw data location).
`trainer.py` additionally takes `--model` and `--feature-set`:

- **`--feature-set freq_agg`** (default): context columns one-hot (capped at 50
  categories) + leakage-safe smoothed target-encoding for the high-cardinality
  user/ad columns. Pairs with any `--model`.
- **`--feature-set baseline_ohe`**: every raw categorical column (including
  `device_id`/`device_ip`/`site_id`/`app_id`) one-hot encoded, uncapped, sparse —
  a naive baseline with no target-encoding. Intended for `--model logreg` (sparse
  input; `hist_gbdt` needs dense and this matrix is too wide to densify).
- **`--feature-set gbdt_leaves`**: the Facebook GBDT+LR technique — a lightgbm
  GBDT trains on the raw categorical + hour columns (via lightgbm's native
  categorical handling), then each row is re-encoded as the one-hot
  concatenation of which leaf it landed in per tree
  (`feature_engineering.GBDTLeafEncoder`), and a linear model trains on those
  induced features instead of the raw columns. Also intended for `--model logreg`.
  Note: lightgbm's native categorical handling has a `max_bin` cap, so very
  high-cardinality columns (`device_id`/`device_ip`) lose some resolution —
  see `gbdt_leaves_ohe` below for a variant that avoids this.
- **`--feature-set gbdt_leaves_ohe`**: same GBDT+LR leaf-embedding technique,
  but the internal GBDT trains on the same uncapped one-hot vectors as
  `baseline_ohe` instead of raw categorical columns
  (`feature_engineering.build_gbdt_leaves_ohe_pipeline`) — every input is
  already a binary indicator, so there's no categorical `max_bin` cap to lose
  resolution to, at the cost of a much wider/sparser GBDT input. Also intended
  for `--model logreg`.
- **`--feature-set gbdt_leaves_concat`**: concatenates the raw `baseline_ohe`
  one-hot vectors with the `gbdt_leaves_ohe` leaf features
  (`feature_engineering.build_gbdt_leaves_concat_pipeline`, via
  `sklearn.pipeline.FeatureUnion`), so the linear model sees both the
  original linear signal and the GBDT's induced non-linear signal — this
  matches the Facebook paper's actual design (leaves *augment* the base
  features rather than replacing them), unlike the leaf-only
  `gbdt_leaves_ohe`. Also intended for `--model logreg`.
- **`--feature-set freq_agg_leaves_concat`**: concatenates `freq_agg`'s
  engineered features (capped context one-hot + smoothed target-encoded
  user/ad aggregates + hour) with GBDT-leaf one-hot features from the same
  GBDT `gbdt_leaves_ohe` already trains — on raw uncapped one-hot, **not**
  on freq_agg's own `_ctr`/`_count` columns
  (`feature_engineering.build_freq_agg_leaves_concat_pipeline`). This is
  deliberate: `_ctr` is leave-one-out encoded for train_df (each row's own
  label excluded from its own category's stats — see `add_freq_agg_features`),
  which introduces a small train-only artifact that a linear model barely
  notices but a second GBDT can and empirically does exploit — a GBDT trained
  directly on freq_agg's dense matrix scored AUC ~0.48 (worse than random) on
  held-out val, because its leaf splits encoded that train-only artifact
  rather than transferable signal. Also intended for `--model logreg`.
- **`--feature-set freq_agg_fm_concat`**: same idea as `freq_agg_leaves_concat`,
  but the induced features come from a Factorization Machine
  (`feature_engineering.FMEmbeddingEncoder`) instead of a GBDT — a lightweight,
  dependency-free degree-2 FM (Rendle, 2010) trained via minibatch SGD
  directly on raw uncapped one-hot vectors, exposing each row's `n_factors`-dim
  latent projection (the sum of its active categories' learned latent
  vectors) as concatenated features alongside `freq_agg`'s. No external FM
  library is used (pyfm/fastFM/xlearn are largely unmaintained C-extension
  packages, high risk of build failure on modern Python/ARM). Also intended
  for `--model logreg`.

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
