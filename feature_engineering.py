import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from consts import HOUR_COL, LABEL_COL


def build_preprocessing_pipeline(
    cat_cols: list[str], num_cols: list[str], max_categories: int = 50
) -> ColumnTransformer:
    # max_categories caps cardinality on the anonymized high-cardinality context
    # columns (C14-C21 etc.) so one-hot encoding stays a manageable size. Dense
    # output (sparse_output=False) because HistGradientBoostingClassifier in the
    # installed scikit-learn version requires dense X.
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", max_categories=max_categories, sparse_output=False),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ]
    )


def build_ohe_only_pipeline(cat_cols: list[str]) -> ColumnTransformer:
    """Cat-only, UNCAPPED one-hot encoding for the `baseline_ohe` feature set —
    a naive baseline that one-hots every raw categorical column including
    high-cardinality device_id/device_ip/site_id/app_id, with no
    max_categories cap and no numeric branch. sparse_output=True is required
    here — dense would be infeasible at this cardinality; LogisticRegression
    (unlike HistGradientBoostingClassifier) accepts sparse input natively.
    handle_unknown="ignore" is expected to produce many all-zero blocks in val
    for the highest-cardinality columns under a time-based split — a known
    limitation of this baseline, not a leakage issue.
    """
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)],
        sparse_threshold=1.0,
    )


def to_lgbm_categoricals(
    train_df: pd.DataFrame, val_df: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast `cat_cols` to pandas 'category' dtype for lightgbm's native
    categorical handling, with each column's categories fixed from
    `train_df` only (per CLAUDE.md's no-leakage rule) — values in `val_df`
    that never appeared in train become NaN, which lightgbm treats as
    missing rather than raising.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    for col in cat_cols:
        categories = train_df[col].astype("category").cat.categories
        train_df[col] = pd.Categorical(train_df[col], categories=categories)
        val_df[col] = pd.Categorical(val_df[col], categories=categories)
    return train_df, val_df


class GBDTLeafEncoder(BaseEstimator, TransformerMixin):
    """GBDT + LR leaf-embedding transformer (the Facebook 2014 ads-CTR
    technique): fits a lightgbm GBDT on X/y, then re-encodes every row as the
    one-hot concatenation of which leaf it landed in, per tree — the GBDT
    does its own feature engineering, and a downstream linear model (e.g.
    LogisticRegression) learns weights over the induced leaf features rather
    than the raw columns. `gbdt_params` are passed straight to
    `lightgbm.LGBMClassifier` (e.g. n_estimators, num_leaves).

    Only `fit(X_train, y_train)` sees labels/leaf assignments derived from
    training data — `transform` on val/test reuses the already-fit GBDT and
    OneHotEncoder, so this satisfies CLAUDE.md's no-leakage rule the same way
    the rest of this module's fit-on-train-only functions do.
    """

    def __init__(self, gbdt_params: dict | None = None):
        self.gbdt_params = gbdt_params

    def fit(self, X, y):
        import lightgbm as lgb

        self.gbdt_ = lgb.LGBMClassifier(**(self.gbdt_params or {}))
        self.gbdt_.fit(X, y)
        leaves = self.gbdt_.predict(X, pred_leaf=True)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.ohe_.fit(leaves)
        return self

    def transform(self, X):
        leaves = self.gbdt_.predict(X, pred_leaf=True)
        return self.ohe_.transform(leaves)


def build_gbdt_leaves_ohe_pipeline(cat_cols: list[str], gbdt_params: dict | None = None) -> Pipeline:
    """`gbdt_leaves_ohe` feature set: like `GBDTLeafEncoder`, but the internal
    GBDT trains on the same uncapped one-hot vectors as `baseline_ohe`
    (via `build_ohe_only_pipeline`) instead of raw categorical columns with
    lightgbm's native categorical handling. This sidesteps lightgbm's
    max_bin cap on high-cardinality categoricals (device_id/device_ip etc.
    silently lose resolution under native handling — see gbdt_leaves'
    training warning) at the cost of a much wider, sparser GBDT input.
    """
    return Pipeline(
        [
            ("ohe", build_ohe_only_pipeline(cat_cols)),
            ("gbdt_leaves", GBDTLeafEncoder(gbdt_params=gbdt_params)),
        ]
    )


def add_hour_features(df: pd.DataFrame, hour_col: str = HOUR_COL) -> pd.DataFrame:
    """Parse Avazu's YYMMDDHH `hour` column into `hour_of_day` and `day_of_week`."""
    df = df.copy()
    hour_str = df[hour_col].astype(str)
    hour_of_day = hour_str.str[6:8].astype(int)
    dt = pd.to_datetime(
        {
            "year": 2000 + hour_str.str[0:2].astype(int),
            "month": hour_str.str[2:4].astype(int),
            "day": hour_str.str[4:6].astype(int),
        }
    )
    df["hour_of_day"] = hour_of_day
    df["day_of_week"] = dt.dt.dayofweek
    return df


DEFAULT_SMOOTHING = 20.0


def fit_freq_agg_stats(
    train_df: pd.DataFrame, group_cols: list[str], label_col: str = LABEL_COL
) -> dict[str, pd.DataFrame]:
    """Fit per-category click sum/count stats on `train_df` only.

    Returns one small DataFrame per group column (indexed by category, columns
    `sum`/`count`) — this is the explicit fit-state CLAUDE.md's no-leakage rule
    calls for: it can be persisted alongside a trained model and reapplied to
    new raw data without needing the original train_df around. Raw sum/count
    (rather than a precomputed mean) is kept so smoothing strength can be
    chosen at apply time.
    """
    return {col: train_df.groupby(col)[label_col].agg(sum="sum", count="count") for col in group_cols}


def apply_freq_agg_stats(
    df: pd.DataFrame,
    fit_stats: dict[str, pd.DataFrame],
    global_ctr: float,
    smoothing: float = DEFAULT_SMOOTHING,
) -> pd.DataFrame:
    """Apply previously-fit per-category stats to `df` as a smoothed CTR:
    `(sum + smoothing * global_ctr) / (count + smoothing)`. Categories unseen
    in the fit have sum=count=0, which this formula naturally resolves to
    `global_ctr`. Smoothing pulls low-count categories toward the global rate
    instead of trusting a handful of observations outright.
    """
    df = df.copy()
    for col, stats in fit_stats.items():
        ctr_col, count_col = f"{col}_ctr", f"{col}_count"
        merged = df[[col]].merge(stats, on=col, how="left")
        sum_ = merged["sum"].fillna(0.0).to_numpy()
        count_ = merged["count"].fillna(0.0).to_numpy()
        df[ctr_col] = (sum_ + smoothing * global_ctr) / (count_ + smoothing)
        df[count_col] = count_
    return df


def add_freq_agg_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_cols: list[str],
    label_col: str = LABEL_COL,
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Add per-category historical CTR + count features, fit on `train_df` only.

    `val_df` gets the stats applied as-fit (its rows never contributed to them,
    so no leakage risk there). `train_df` gets a **leave-one-out + smoothed**
    application instead: each row's own label is excluded from its own group's
    sum/count, and the result is blended toward `global_ctr` by `smoothing`
    pseudo-observations. Plain leave-one-out (no smoothing) isn't enough on its
    own — for a group of size 2, one row's "leave-one-out" CTR is just the
    *other* row's raw label, which a flexible model (e.g. a GBDT) can still
    exploit to memorize training rows via high-cardinality columns like
    device_id/device_ip (confirmed empirically: unsmoothed LOO gave
    hist_gbdt train AUC 1.0 but val AUC ~0.51 — pure overfitting to
    near-unique small groups). Smoothing bounds how much any single group,
    however small, can shift its members' feature value.

    Returns `(train_df, val_df, fit_stats)` — `fit_stats` is the explicit
    fit-state from `fit_freq_agg_stats`, suitable for persisting alongside a
    trained model (see `trainer.save_model`).
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    global_ctr = train_df[label_col].mean()

    fit_stats = fit_freq_agg_stats(train_df, group_cols, label_col)
    val_df = apply_freq_agg_stats(val_df, fit_stats, global_ctr, smoothing)

    for col in group_cols:
        ctr_col, count_col = f"{col}_ctr", f"{col}_count"
        grouped = train_df.groupby(col)[label_col]
        group_sum = grouped.transform("sum")
        group_count = grouped.transform("count")

        loo_sum = group_sum - train_df[label_col]
        loo_count = group_count - 1
        train_df[ctr_col] = (loo_sum + smoothing * global_ctr) / (loo_count + smoothing)
        train_df[count_col] = group_count

    return train_df, val_df, fit_stats
