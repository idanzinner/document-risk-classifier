"""
visualization.py — Training diagnostics and result visualizations.

Provides utilities for plotting calibration curves, ROC/PR curves,
confusion matrices, confidence histograms, per-institution metrics,
error slice summaries, and score distributions.

All functions save to output_path and close the figure without displaying.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


_TERNARY_LABELS = {0: "safe_for_extraction", 1: "high_hallucination_risk", 2: "review"}

ERROR_SLICES = [
    "printed_report",
    "complete_form",
    "partial_form",
    "empty_questionnaire",
    "marked_questionnaire",
    "handwriting_dominant",
    "poor_scan",
]


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_ternary: np.ndarray,
    output_path: str,
    title: str = "",
) -> None:
    """
    Plots a confusion matrix for ternary predictions.

    Args:
        y_true:         np.ndarray [N], binary int labels (0=safe, 1=risky).
        y_pred_ternary: np.ndarray [N], ternary predictions
                        (0=safe, 1=risky, 2=review).
        output_path:    Path to save the figure.
        title:          Optional plot title.
    """
    y_true = np.asarray(y_true)
    y_pred_ternary = np.asarray(y_pred_ternary)

    # Rows = true binary, cols = predicted ternary
    pred_labels = sorted(np.unique(y_pred_ternary).tolist())
    true_labels = [0, 1]
    matrix = np.zeros((2, 3), dtype=int)
    for pred_val in range(3):
        for true_val in range(2):
            matrix[true_val, pred_val] = int(((y_true == true_val) & (y_pred_ternary == pred_val)).sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)

    col_labels = [_TERNARY_LABELS[i] for i in range(3)]
    row_labels = ["true_safe", "true_risky"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(2))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title or "Confusion Matrix", fontsize=11)

    for row in range(2):
        for col in range(3):
            val = matrix[row, col]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(col, row, str(val), ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    title: str = "",
) -> None:
    """
    Plots the ROC curve with AUC annotation.

    Args:
        y_true:      np.ndarray [N], binary int labels.
        y_prob:      np.ndarray [N], predicted probabilities.
        output_path: Path to save the figure.
        title:       Optional plot title.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))

    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "Single class — ROC undefined", ha="center", va="center", transform=ax.transAxes)
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=9)

    ax.set_title(title or "ROC Curve", fontsize=11)
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    title: str = "",
) -> None:
    """
    Plots the Precision-Recall curve with average precision annotation.

    Args:
        y_true:      np.ndarray [N], binary int labels.
        y_prob:      np.ndarray [N], predicted probabilities.
        output_path: Path to save the figure.
        title:       Optional plot title.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))

    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "Single class — PR curve undefined", ha="center", va="center", transform=ax.transAxes)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        baseline = y_true.mean()
        ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline = {baseline:.3f}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title(title or "Precision-Recall Curve", fontsize=11)
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    n_bins: int = 10,
) -> None:
    """
    Plots a reliability diagram (calibration curve).

    Args:
        y_true:      np.ndarray [N], binary int labels.
        y_prob:      np.ndarray [N], predicted probabilities.
        output_path: Path to save the figure.
        n_bins:      Number of calibration bins.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))

    if len(np.unique(y_true)) >= 2:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", lw=2, label="Model")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_title("Calibration Curve", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_institution_metrics(
    institution_df: pd.DataFrame,
    metric: str,
    output_path: str,
) -> None:
    """
    Plots a bar chart of a single metric across institutions.

    Args:
        institution_df: DataFrame from compute_per_institution_metrics(),
                        columns: institution, n_samples, f1, recall_risky,
                                 false_safe_rate, review_rate.
        metric:         Column name to plot (e.g. 'false_safe_rate').
        output_path:    Path to save the figure.
    """
    df = institution_df.copy()
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in DataFrame columns: {list(df.columns)}")

    df = df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.7 + 1), 5))

    bars = ax.bar(range(len(df)), df[metric].values, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["institution"].tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
    ax.set_title(f"{metric.replace('_', ' ').title()} by Institution", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for bar, val in zip(bars, df[metric].values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error_slice_summary(
    error_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plots a grouped bar chart of false-safe and false-risky counts per error slice.

    Expects error_df to have a column 'predicted_category' and 'true_label', plus
    a column that identifies which of the 7 error slices the sample belongs to.
    Slice membership is inferred from 'template_family' if present, otherwise
    tries a column named 'error_slice'; if neither exists, falls back to
    grouping by whatever categorical column is available.

    The 7 canonical slices:
      printed_report, complete_form, partial_form, empty_questionnaire,
      marked_questionnaire, handwriting_dominant, poor_scan

    False-safe: true_label==1 AND predicted_category=='safe_for_extraction'
    False-risky: true_label==0 AND predicted_category=='high_hallucination_risk'

    Args:
        error_df:    DataFrame from ErrorAnalysisLogger records.
        output_path: Path to save the figure.
    """
    df = error_df.copy()

    # Determine slice column
    if "error_slice" in df.columns:
        slice_col = "error_slice"
    elif "template_family" in df.columns:
        slice_col = "template_family"
    else:
        available_cat = [c for c in df.columns if df[c].dtype == object and c not in ("file_path", "scan_quality_note", "handwriting_note")]
        slice_col = available_cat[0] if available_cat else None

    false_safe_counts = []
    false_risky_counts = []
    slice_names = []

    candidate_slices = ERROR_SLICES if slice_col is not None else []

    if slice_col is not None:
        all_slices = set(df[slice_col].dropna().unique())
        for sl in ERROR_SLICES:
            slice_names.append(sl)
            mask = df[slice_col] == sl
            sub = df[mask]
            fs = int(((sub["true_label"] == 1) & (sub["predicted_category"] == "safe_for_extraction")).sum())
            fr = int(((sub["true_label"] == 0) & (sub["predicted_category"] == "high_hallucination_risk")).sum())
            false_safe_counts.append(fs)
            false_risky_counts.append(fr)
        # Add any slices present in data but not in canonical list
        for sl in sorted(all_slices - set(ERROR_SLICES)):
            slice_names.append(sl)
            mask = df[slice_col] == sl
            sub = df[mask]
            fs = int(((sub["true_label"] == 1) & (sub["predicted_category"] == "safe_for_extraction")).sum())
            fr = int(((sub["true_label"] == 0) & (sub["predicted_category"] == "high_hallucination_risk")).sum())
            false_safe_counts.append(fs)
            false_risky_counts.append(fr)
    else:
        # Fallback: aggregate
        slice_names = ["all"]
        false_safe_counts = [int(((df["true_label"] == 1) & (df["predicted_category"] == "safe_for_extraction")).sum())]
        false_risky_counts = [int(((df["true_label"] == 0) & (df["predicted_category"] == "high_hallucination_risk")).sum())]

    x = np.arange(len(slice_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(slice_names) * 1.2 + 1), 5))
    bars1 = ax.bar(x - width / 2, false_safe_counts, width, label="False-Safe", color="#e05c5c", edgecolor="white")
    bars2 = ax.bar(x + width / 2, false_risky_counts, width, label="False-Risky", color="#5c8ae0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(slice_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Error Counts by Slice Category", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)), ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_score_distribution(
    risk_scores: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """
    Plots overlapping histograms of risk scores split by true label.

    Args:
        risk_scores: np.ndarray [N], predicted probabilities or risk scores.
        labels:      np.ndarray [N], binary labels (0=safe, 1=risky).
        output_path: Path to save the figure.
    """
    risk_scores = np.asarray(risk_scores, dtype=float)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 4))

    safe_scores = risk_scores[labels == 0]
    risky_scores = risk_scores[labels == 1]

    bins = np.linspace(0, 1, 30)
    if len(safe_scores) > 0:
        ax.hist(safe_scores, bins=bins, alpha=0.6, color="#4caf50", label=f"Safe (n={len(safe_scores)})", density=True)
    if len(risky_scores) > 0:
        ax.hist(risky_scores, bins=bins, alpha=0.6, color="#f44336", label=f"Risky (n={len(risky_scores)})", density=True)

    ax.set_xlabel("Risk Score / Predicted Probability", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Score Distribution by True Label", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])

    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
