import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def evaluate(y_true, y_pred_proba) -> dict:
    """Compute the standard CTR-prediction metric trio for one set of predictions.

    Args:
        y_true: True binary labels, array-like of shape `(n,)`, values in `{0, 1}`.
        y_pred_proba: Predicted probability of the positive class, array-like
            of shape `(n,)`, floats in `[0, 1]`.

    Returns:
        dict with keys `"auc"`, `"logloss"`, `"pr_auc"`, each a scalar float.
    """
    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "logloss": log_loss(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }


def print_metrics(metrics: dict) -> None:
    """Print each metric as `name: value` at 4 decimal places.

    Args:
        metrics: Mapping of metric name (str) to scalar float value, e.g.
            `evaluate`'s return value.

    Returns:
        None (prints to stdout).
    """
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def calibration_table(y_true, y_pred_proba, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into `n_bins` quantiles and compare mean predicted vs. actual rate per bucket.

    Args:
        y_true: True binary labels, array-like of shape `(n,)`.
        y_pred_proba: Predicted probabilities, array-like of shape `(n,)`.
        n_bins: Number of equal-frequency (quantile) buckets to split
            `y_pred_proba` into.

    Returns:
        pd.DataFrame with up to `n_bins` rows (fewer if `pd.qcut` drops
        duplicate bin edges) and columns `bucket`, `mean_predicted`,
        `mean_actual`, `n`.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred_proba": np.asarray(y_pred_proba)})
    df["bucket"] = pd.qcut(df["y_pred_proba"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bucket", observed=True)
        .agg(mean_predicted=("y_pred_proba", "mean"), mean_actual=("y_true", "mean"), n=("y_true", "size"))
        .reset_index()
    )


def lift_at_k(y_true, y_pred_proba, k_fractions: list[float] = [0.05, 0.1, 0.2]) -> dict:
    """Compute click-rate lift (top-k CTR / overall CTR) at each fraction in `k_fractions`.

    Args:
        y_true: True binary labels, array-like of shape `(n,)`.
        y_pred_proba: Predicted probabilities, array-like of shape `(n,)`,
            used to rank rows before taking each top-k slice.
        k_fractions: List of top-fraction cutoffs, each a scalar float in
            `(0, 1]` (e.g. `0.05` = top 5% of rows by predicted probability).

    Returns:
        dict mapping each entry of `k_fractions` to its lift ratio (scalar
        float; `nan` if the overall CTR is 0).
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred_proba": np.asarray(y_pred_proba)})
    df = df.sort_values("y_pred_proba", ascending=False).reset_index(drop=True)
    baseline_ctr = df["y_true"].mean()

    lifts = {}
    for k in k_fractions:
        n = max(1, int(len(df) * k))
        top_k_ctr = df.iloc[:n]["y_true"].mean()
        lifts[k] = top_k_ctr / baseline_ctr if baseline_ctr > 0 else float("nan")
    return lifts
