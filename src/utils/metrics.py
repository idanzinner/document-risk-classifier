"""
metrics.py — Classification and calibration metric utilities.

Computes threshold-based ternary metrics (F1, false-safe rate, review rate),
ROC/PR AUC, and Expected Calibration Error (ECE).
Supports both aggregate and per-institution breakdowns.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


def _apply_thresholds(y_prob: np.ndarray, thresholds: dict) -> np.ndarray:
    """
    Maps probabilities to ternary labels using thresholds.

    Returns:
        Array with values 0 (safe), 1 (risky), or 2 (review).
    """
    t_low = thresholds["T_low"]
    t_high = thresholds["T_high"]
    pred = np.where(y_prob < t_low, 0, np.where(y_prob > t_high, 1, 2))
    return pred.astype(int)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict,
) -> dict:
    """
    Computes all reported evaluation metrics for the binary classifier.

    Threshold-based ternary mapping:
      prob < T_low          → 0 (safe_for_extraction)
      T_low <= prob <= T_high → 2 (review)
      prob > T_high         → 1 (high_hallucination_risk)

    For safety-critical binary metrics, review (2) is treated as risky (1):
      - f1: computed on binary predictions (safe=0, not-safe=1)
      - precision_safe: fraction of predicted-safe that are truly safe
      - recall_risky: fraction of truly risky predicted as risky (NOT including review)

    Args:
        y_true:     np.ndarray [N], binary int labels (0=safe, 1=risky).
        y_prob:     np.ndarray [N], calibrated probabilities.
        thresholds: Dict {'T_low': float, 'T_high': float}.

    Returns:
        Dict with keys:
          f1, precision_safe, recall_risky, false_safe_rate,
          review_rate, roc_auc, pr_auc, ece
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    if n == 0:
        return {
            "f1": float("nan"),
            "precision_safe": float("nan"),
            "recall_risky": float("nan"),
            "false_safe_rate": float("nan"),
            "review_rate": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "ece": float("nan"),
        }

    pred_ternary = _apply_thresholds(y_prob, thresholds)

    # Binary predictions: review → risky (conservative for safety)
    pred_binary = np.where(pred_ternary == 0, 0, 1)

    n_classes = len(np.unique(y_true))

    # F1 score (treating risky=1 as positive, review counted as risky)
    f1 = f1_score(y_true, pred_binary, pos_label=1, zero_division=0)

    # precision_safe: among predicted-safe, fraction that are truly safe
    safe_mask = pred_ternary == 0
    if safe_mask.sum() == 0:
        precision_safe = float("nan")
    else:
        precision_safe = float((y_true[safe_mask] == 0).sum()) / float(safe_mask.sum())

    # recall_risky: among truly risky, fraction predicted strictly risky (not review)
    risky_mask = y_true == 1
    if risky_mask.sum() == 0:
        recall_risky = float("nan")
    else:
        strictly_risky_pred = pred_ternary == 1
        recall_risky = float((strictly_risky_pred & risky_mask).sum()) / float(risky_mask.sum())

    # false_safe_rate: fraction of truly risky docs predicted safe
    if risky_mask.sum() == 0:
        false_safe_rate = float("nan")
    else:
        false_safe_rate = float(((pred_ternary == 0) & risky_mask).sum()) / float(risky_mask.sum())

    # review_rate: fraction routed to review
    review_rate = float((pred_ternary == 2).sum()) / float(n)

    # ROC AUC — requires both classes present
    if n_classes < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float("nan")

    ece = compute_ece(y_true, y_prob)

    return {
        "f1": float(f1),
        "precision_safe": precision_safe,
        "recall_risky": recall_risky,
        "false_safe_rate": false_safe_rate,
        "review_rate": review_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": ece,
    }


def compute_per_institution_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    institutions: np.ndarray,
    thresholds: dict,
) -> pd.DataFrame:
    """
    Computes per-institution breakdown of key metrics.

    Args:
        y_true:       np.ndarray [N], binary int labels (0=safe, 1=risky).
        y_prob:       np.ndarray [N], calibrated probabilities.
        institutions: np.ndarray [N], institution string per sample.
        thresholds:   Dict {'T_low': float, 'T_high': float}.

    Returns:
        DataFrame with columns:
          institution, n_samples, f1, recall_risky, false_safe_rate, review_rate
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    institutions = np.asarray(institutions)

    records = []
    for inst in np.unique(institutions):
        mask = institutions == inst
        subset_true = y_true[mask]
        subset_prob = y_prob[mask]
        m = compute_metrics(subset_true, subset_prob, thresholds)
        records.append(
            {
                "institution": inst,
                "n_samples": int(mask.sum()),
                "f1": m["f1"],
                "recall_risky": m["recall_risky"],
                "false_safe_rate": m["false_safe_rate"],
                "review_rate": m["review_rate"],
            }
        )

    return pd.DataFrame(records, columns=["institution", "n_samples", "f1", "recall_risky", "false_safe_rate", "review_rate"])


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    Bins predictions by confidence and measures the mean absolute difference
    between mean predicted probability and observed accuracy within each bin,
    weighted by bin size.

    Args:
        y_true:  np.ndarray [N], binary int labels.
        y_prob:  np.ndarray [N], predicted probabilities.
        n_bins:  Number of equally-spaced bins in [0, 1].

    Returns:
        Scalar ECE value.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    if n == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge only for the last bin
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        if mask.sum() == 0:
            continue

        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_conf - bin_acc)

    return float(ece)
