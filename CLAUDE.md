# ctr_prediction

CTR (click-through rate) prediction research project. See `README.md` (once created) for setup and usage.

## Project locations

- Project dir: `/Users/gregorygoren/PycharmProjects/ctr_prediction`
- Dataset dir: `/Users/gregorygoren/Documents/research/avazu-ctr-prediction/`
  (`train.gz`, `test.gz`, `sampleSubmission.gz` — full canonical Avazu Kaggle
  CTR dataset, referenced via `consts.RAW_DATA_PATH`)

## Rules

1. **Code review before any GitHub commit.** Before committing changes to this repo, run a code review (the `/code-review` skill) and address findings first. Never commit straight from a diff without reviewing it.
2. **No train/test leakage in feature engineering.** Any statistic computed from the data (per-user CTR, per-ad CTR, other aggregates/encodings) must be fit on the training fold only and then applied to validation/test — never computed on the full dataset before splitting. Functions that compute such aggregates must take/return explicit fit-state (e.g. `fit_stats`) so it's obvious what was learned from train and reused elsewhere, rather than silently recomputing from whatever DataFrame is passed in.
