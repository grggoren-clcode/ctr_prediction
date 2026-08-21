from pathlib import Path

import pandas as pd

from consts import HOUR_COL, LABEL_COL, RAW_DATA_PATH, SAMPLE_N_ROWS, VAL_FRAC


def load_sample(
    path: Path = RAW_DATA_PATH,
    n_rows: int = SAMPLE_N_ROWS,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Load the leading `n_rows` of the (gzip-compressed) raw CSV.

    train.gz is already sorted by `hour`, so this yields a contiguous,
    chronologically-ordered sample without decompressing the full file.

    Args:
        path: Filesystem path to the gzip-compressed raw CSV (Avazu's
            `train.gz` schema — `id, click, hour, C1, banner_pos, site_*,
            app_*, device_*, C14-C21`).
        n_rows: Number of leading rows to read (scalar int).
        label_col: Name of the binary label column whose presence is
            validated after loading.

    Returns:
        pd.DataFrame of shape `(min(n_rows, file_row_count), n_raw_cols)`,
        one row per event and one column per raw CSV field.
    """
    df = pd.read_csv(path, compression="infer", nrows=n_rows)
    if label_col not in df.columns:
        raise ValueError(f"Expected label column {label_col!r} not found in {list(df.columns)}")
    return df


def validate_schema(df: pd.DataFrame, label_col: str = LABEL_COL) -> None:
    """Check that `label_col` is strictly binary and report any columns with nulls.

    Args:
        df: Sample DataFrame to validate, shape `(n_rows, n_raw_cols)` (as
            returned by `load_sample`) — must contain `label_col`.
        label_col: Name of the column expected to hold only 0/1 values.

    Returns:
        None. Raises `ValueError` if `label_col` contains any value outside
        `{0, 1}`; prints (does not raise on) any column-wise null rates > 0.
    """
    labels = set(df[label_col].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Label column {label_col!r} must be binary (0/1), found: {labels}")

    null_rates = df.isnull().mean()
    nonzero_nulls = null_rates[null_rates > 0]
    if not nonzero_nulls.empty:
        print("Columns with nulls:")
        print(nonzero_nulls.sort_values(ascending=False))


def time_based_split(
    df: pd.DataFrame,
    hour_col: str = HOUR_COL,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: train on earlier hours, validate on later hours.

    No shuffling — validation is strictly later in time than train, which both
    simulates real deployment and avoids future-into-past leakage.

    Args:
        df: Full loaded sample, shape `(n_rows, n_cols)` — must contain
            `hour_col`.
        hour_col: Name of the `YYMMDDHH`-formatted column used to sort
            chronologically before splitting.
        val_frac: Fraction (scalar float in `[0, 1]`) of rows, by
            chronological position (not random), held out for validation.

    Returns:
        `(train_df, val_df)` — DataFrames of shape
        `(round(n_rows * (1 - val_frac)), n_cols)` and
        `(n_rows - round(n_rows * (1 - val_frac)), n_cols)` respectively (row
        counts from `int()` truncation of `len(df) * (1 - val_frac)`), each
        with a fresh `0..len-1` index.
    """
    df_sorted = df.sort_values(hour_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_frac))
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    val_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df
