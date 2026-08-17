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
    """
    df = pd.read_csv(path, compression="infer", nrows=n_rows)
    if label_col not in df.columns:
        raise ValueError(f"Expected label column {label_col!r} not found in {list(df.columns)}")
    return df


def validate_schema(df: pd.DataFrame, label_col: str = LABEL_COL) -> None:
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
    """
    df_sorted = df.sort_values(hour_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_frac))
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    val_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df
