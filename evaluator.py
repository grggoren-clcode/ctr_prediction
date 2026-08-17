import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def evaluate(y_true, y_pred_proba) -> dict:
    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "logloss": log_loss(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }


def print_metrics(metrics: dict) -> None:
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def calibration_table(y_true, y_pred_proba, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred_proba": np.asarray(y_pred_proba)})
    df["bucket"] = pd.qcut(df["y_pred_proba"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bucket", observed=True)
        .agg(mean_predicted=("y_pred_proba", "mean"), mean_actual=("y_true", "mean"), n=("y_true", "size"))
        .reset_index()
    )


def lift_at_k(y_true, y_pred_proba, k_fractions: list[float] = [0.05, 0.1, 0.2]) -> dict:
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred_proba": np.asarray(y_pred_proba)})
    df = df.sort_values("y_pred_proba", ascending=False).reset_index(drop=True)
    baseline_ctr = df["y_true"].mean()

    lifts = {}
    for k in k_fractions:
        n = max(1, int(len(df) * k))
        top_k_ctr = df.iloc[:n]["y_true"].mean()
        lifts[k] = top_k_ctr / baseline_ctr if baseline_ctr > 0 else float("nan")
    return lifts
