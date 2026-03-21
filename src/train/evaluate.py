"""
evaluate.py — Evaluation and error-analysis utilities.

Runs a trained + calibrated model on the held-out test split (or any split),
computes all reported metrics, generates per-slice error analysis, and
produces an eval_summary.json for downstream inspection.

Error analysis row schema:
  file_path, page_num, true_label, predicted_category, confidence,
  institution, template_family, risk_score, D, H, S, L,
  scan_quality_note, handwriting_note
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import HallucinationRiskDataset
from src.models.calibrator import TemperatureCalibrator
from src.utils.logging import ErrorAnalysisLogger
from src.utils.metrics import compute_metrics, compute_per_institution_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(
    model_type: str,
    checkpoint_path: str,
    device: torch.device,
    cfg: Optional[dict] = None,
) -> nn.Module:
    """
    Load a trained model from a checkpoint.

    Args:
        model_type: One of 'resnet50', 'efficientnet_b0', 'vit', 'dit' (or full
                    timm/HF model names).
        checkpoint_path: Path to a .pt checkpoint saved by the training scripts.
        device: Target device.
        cfg: Optional config dict; used to infer model_name if available.

    Returns:
        nn.Module in eval mode on *device*.
    """
    # Resolve the model name (prefer cfg, fall back to model_type)
    if cfg is not None:
        model_name = cfg.get("model", {}).get("name", model_type)
        num_classes = cfg.get("model", {}).get("num_classes", 1)
        pretrained = cfg.get("model", {}).get("pretrained", False)
    else:
        model_name = model_type
        num_classes = 1
        pretrained = False

    # Instantiate the right class
    if "dit" in model_type.lower() or "dit" in model_name.lower():
        from src.models.dit_classifier import DiTClassifier
        model = DiTClassifier(model_name=model_name, num_classes=num_classes)
    elif model_type.lower().startswith("vit"):
        from src.models.vit_baseline import ViTClassifier
        model = ViTClassifier(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
    else:
        from src.models.resnet_baseline import ResNetClassifier
        model = ResNetClassifier(model_name=model_name, pretrained=pretrained, num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded %s checkpoint from %s", model_type, checkpoint_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect raw logits and labels from the entire loader.

    Returns:
        (logits, labels) — both float32 np.ndarray of shape [N].
    """
    all_logits: List[float] = []
    all_labels: List[float] = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="inference", leave=True):
            images = images.to(device)
            logits = model(images)  # [B, 1]
            all_logits.extend(logits.cpu().squeeze(1).tolist())
            all_labels.extend(labels.cpu().squeeze(1).tolist())

    return (
        np.array(all_logits, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error-slice definitions
# ─────────────────────────────────────────────────────────────────────────────

def _get_slice_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build boolean masks for each of the 7 predefined error slices.

    Columns required (if absent the slice mask will be all-False):
      template_family, risk_score, H, L
    """

    def _col(name: str) -> Optional[pd.Series]:
        return df[name] if name in df.columns else None

    tf = _col("template_family")
    rs = _col("risk_score")
    h_col = _col("H")
    l_col = _col("L")

    n = len(df)
    false_mask = np.zeros(n, dtype=bool)

    def _tf_contains(keyword: str) -> np.ndarray:
        if tf is None:
            return false_mask.copy()
        return tf.str.lower().str.contains(keyword, na=False).values

    def _rs_range(lo: Optional[float], hi: Optional[float]) -> np.ndarray:
        if rs is None:
            return np.ones(n, dtype=bool)
        arr = rs.values.astype(float)
        mask = np.ones(n, dtype=bool)
        if lo is not None:
            mask &= arr >= lo
        if hi is not None:
            mask &= arr <= hi
        return mask

    def _hval(threshold: float, op: str) -> np.ndarray:
        if h_col is None:
            return false_mask.copy()
        arr = h_col.values.astype(float)
        return (arr >= threshold) if op == ">=" else (arr <= threshold)

    def _lval(threshold: float, op: str) -> np.ndarray:
        if l_col is None:
            return false_mask.copy()
        arr = l_col.values.astype(float)
        return (arr >= threshold) if op == ">=" else (arr <= threshold)

    slices: Dict[str, np.ndarray] = {
        "printed_report": (
            _tf_contains("report") | _tf_contains("letter") | _tf_contains("summary")
        ),
        "complete_form": (
            _tf_contains("form") & _rs_range(None, 3)
        ),
        "partial_form": (
            _tf_contains("form") & _rs_range(4, 6)
        ),
        "empty_questionnaire": (
            _tf_contains("questionnaire") & _rs_range(7, None)
        ),
        "marked_questionnaire": (
            _tf_contains("questionnaire") & _hval(1, "<=") & _rs_range(4, 6)
        ),
        "handwriting_dominant": _hval(2, ">="),
        "poor_scan": _lval(2, ">="),
    }
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Category mapping
# ─────────────────────────────────────────────────────────────────────────────

def _probs_to_categories(probs: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Map calibrated probabilities to ternary category strings."""
    T_low = thresholds["T_low"]
    T_high = thresholds["T_high"]
    cats = np.where(
        probs < T_low,
        "safe_for_extraction",
        np.where(probs >= T_high, "high_hallucination_risk", "review"),
    )
    return cats


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_error_analysis_df(
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    predicted_categories: np.ndarray,
) -> pd.DataFrame:
    """
    Builds a DataFrame of per-page predictions joined with metadata.

    Returns:
        DataFrame with columns matching the error analysis log schema:
          file_path, page_num, true_label, predicted_category, confidence,
          institution, template_family, risk_score, D, H, S, L,
          scan_quality_note, handwriting_note
    """
    optional_cols = [
        "file_path", "page_num", "institution", "template_family",
        "risk_score", "D", "H", "S", "L",
        "scan_quality_note", "handwriting_note",
    ]

    result = pd.DataFrame({
        "true_label": y_true.astype(int),
        "predicted_category": predicted_categories,
        "confidence": y_prob,
    })

    for col in optional_cols:
        if col in metadata_df.columns:
            result[col] = metadata_df[col].values
        else:
            result[col] = None

    # Classify errors
    result["is_false_safe"] = (
        (result["true_label"] == 1)
        & (result["predicted_category"] == "safe_for_extraction")
    )
    result["is_false_risky"] = (
        (result["true_label"] == 0)
        & (result["predicted_category"] == "high_hallucination_risk")
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluate function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    calibrator: TemperatureCalibrator,
    metadata_csv: str,
    rendered_dir: str,
    thresholds: Dict[str, float],
    device: torch.device,
    output_dir: str,
    split: str = "test",
) -> dict:
    """
    Evaluates model + calibrator on the specified split.

    Args:
        model: Trained nn.Module.
        calibrator: Fitted TemperatureCalibrator.
        metadata_csv: Path to data/metadata.csv.
        rendered_dir: Path to rendered page PNGs.
        thresholds: Dict {'T_low': float, 'T_high': float}.
        device: torch.device.
        output_dir: Directory where eval_summary.json and error_analysis.csv are saved.
        split: One of 'train', 'val', 'test'.

    Returns:
        Dict of aggregated test metrics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset & loader ──────────────────────────────────────────────────
    dataset = HallucinationRiskDataset(
        metadata_csv=metadata_csv,
        split=split,
        rendered_dir=rendered_dir,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logger.info("Evaluating on split='%s': %d samples", split, len(dataset))

    # ── Inference ─────────────────────────────────────────────────────────
    logits, y_true = _run_inference(model, loader, device)
    y_prob = calibrator.predict(logits)
    predicted_categories = _probs_to_categories(y_prob, thresholds)

    # ── Primary metrics ───────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_prob, thresholds)
    logger.info("Primary metrics: %s", metrics)

    # ── Per-institution metrics ───────────────────────────────────────────
    institutions: np.ndarray
    if "institution" in dataset.df.columns:
        institutions = dataset.df["institution"].values
    else:
        institutions = np.array(["unknown"] * len(dataset))

    per_inst_df = compute_per_institution_metrics(y_true, y_prob, institutions, thresholds)

    # ── Error slices ──────────────────────────────────────────────────────
    slice_masks = _get_slice_masks(dataset.df)
    slice_metrics: Dict[str, dict] = {}

    for slice_name, mask in slice_masks.items():
        n_slice = mask.sum()
        if n_slice == 0:
            logger.info("Slice '%s': 0 samples — skipping", slice_name)
            slice_metrics[slice_name] = {"n_samples": 0}
            continue

        s_metrics = compute_metrics(y_true[mask], y_prob[mask], thresholds)
        s_metrics["n_samples"] = int(n_slice)
        slice_metrics[slice_name] = s_metrics
        logger.info("Slice '%s' (n=%d): F1=%.4f, false_safe_rate=%.4f",
                    slice_name, n_slice,
                    s_metrics.get("f1", float("nan")),
                    s_metrics.get("false_safe_rate", float("nan")))

    # ── Error analysis DataFrame ──────────────────────────────────────────
    error_df = build_error_analysis_df(
        metadata_df=dataset.df.reset_index(drop=True),
        y_true=y_true,
        y_prob=y_prob,
        predicted_categories=predicted_categories,
    )

    # Log false-safe and false-risky samples
    false_safe = error_df[error_df["is_false_safe"]]
    false_risky = error_df[error_df["is_false_risky"]]
    logger.info("False-safe samples: %d", len(false_safe))
    logger.info("False-risky samples: %d", len(false_risky))

    error_logger = ErrorAnalysisLogger(str(out_dir / "error_analysis_log.csv"))
    for _, row in false_safe.iterrows():
        logger.warning(
            "[FALSE-SAFE] file=%s page=%s conf=%.3f institution=%s",
            row.get("file_path"), row.get("page_num"),
            row["confidence"], row.get("institution"),
        )
        error_logger.log_sample(
            file_path=str(row.get("file_path", "")),
            page_num=int(row.get("page_num", 1)),
            true_label=int(row.get("true_label", -1)),
            predicted_category=str(row.get("predicted_category", "")),
            confidence=float(row.get("confidence", 0.0)),
            institution=str(row.get("institution", "")),
            template_family=str(row.get("template_family", "")),
            risk_score=int(row.get("risk_score", -1)),
            D=int(row.get("D", -1)),
            H=int(row.get("H", -1)),
            S=int(row.get("S", -1)),
            L=int(row.get("L", -1)),
        )

    for _, row in false_risky.iterrows():
        logger.warning(
            "[FALSE-RISKY] file=%s page=%s conf=%.3f institution=%s",
            row.get("file_path"), row.get("page_num"),
            row["confidence"], row.get("institution"),
        )
        error_logger.log_sample(
            file_path=str(row.get("file_path", "")),
            page_num=int(row.get("page_num", 1)),
            true_label=int(row.get("true_label", -1)),
            predicted_category=str(row.get("predicted_category", "")),
            confidence=float(row.get("confidence", 0.0)),
            institution=str(row.get("institution", "")),
            template_family=str(row.get("template_family", "")),
            risk_score=int(row.get("risk_score", -1)),
            D=int(row.get("D", -1)),
            H=int(row.get("H", -1)),
            S=int(row.get("S", -1)),
            L=int(row.get("L", -1)),
        )
    error_logger.save()

    # Save error analysis CSV
    error_csv_path = out_dir / "error_analysis.csv"
    error_df.to_csv(error_csv_path, index=False)
    logger.info("Error analysis saved to %s", error_csv_path)

    # ── Assemble summary ──────────────────────────────────────────────────
    summary = {
        "split": split,
        "n_samples": len(dataset),
        "thresholds": thresholds,
        "primary_metrics": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        },
        "per_institution": per_inst_df.to_dict(orient="records"),
        "error_slices": {
            name: {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in s.items()
            }
            for name, s in slice_metrics.items()
        },
        "false_safe_count": int(len(false_safe)),
        "false_risky_count": int(len(false_risky)),
    }

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation summary saved to %s", summary_path)

    # ── Print clean summary table ─────────────────────────────────────────
    _print_summary(summary)

    return summary


def _print_summary(summary: dict) -> None:
    """Print a clean evaluation summary table to stdout."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY  (split={summary['split']}, n={summary['n_samples']})")
    print("=" * 60)

    thresholds = summary["thresholds"]
    print(f"\n  Thresholds: T_low={thresholds['T_low']:.4f}  T_high={thresholds['T_high']:.4f}")

    print("\n  Primary Metrics:")
    print(f"  {'Metric':<28} {'Value':>8}")
    print(f"  {'-'*38}")
    for k, v in summary["primary_metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:>8.4f}")
        else:
            print(f"  {k:<28} {str(v):>8}")

    print(f"\n  False-safe count:  {summary['false_safe_count']}")
    print(f"  False-risky count: {summary['false_risky_count']}")

    print("\n  Error Slice Metrics (F1 / false_safe_rate / n):")
    print(f"  {'Slice':<28} {'F1':>7}  {'FSR':>7}  {'N':>6}")
    print(f"  {'-'*52}")
    for name, sm in summary["error_slices"].items():
        n = sm.get("n_samples", 0)
        if n == 0:
            print(f"  {name:<28} {'—':>7}  {'—':>7}  {0:>6}")
        else:
            f1 = sm.get("f1", float("nan"))
            fsr = sm.get("false_safe_rate", float("nan"))
            print(f"  {name:<28} {f1:>7.4f}  {fsr:>7.4f}  {n:>6}")

    if summary.get("per_institution"):
        print("\n  Per-Institution Breakdown:")
        inst_df = pd.DataFrame(summary["per_institution"])
        print(inst_df.to_string(index=False))

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained hallucination-risk classifier"
    )
    p.add_argument("--checkpoint", required=True, help="Path to model .pt checkpoint")
    p.add_argument("--calibrator", required=True, help="Path to calibrator .pkl file")
    p.add_argument("--metadata_csv", required=True, help="Path to data/metadata.csv")
    p.add_argument("--rendered_dir", required=True, help="Path to rendered PNG directory")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--model_type",
        default="resnet50",
        help="Model type: resnet50, efficientnet_b0, vit, dit (or full HF/timm name)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional YAML config (used to infer model architecture)",
    )
    p.add_argument("--output_dir", default="eval_output", help="Directory for output files")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = _parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # Load config if provided
    cfg: Optional[dict] = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # Load model
    model = _load_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        device=device,
        cfg=cfg,
    )

    # Load calibrator
    with open(args.calibrator, "rb") as fh:
        calibrator: TemperatureCalibrator = pickle.load(fh)

    # Derive thresholds from calibrator state if they were stored, otherwise
    # re-derive on the evaluation split (use val for threshold derivation in practice).
    # Here we load thresholds from the checkpoint if available.
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "thresholds" in ckpt:
        thresholds = ckpt["thresholds"]
        logger.info("Using stored thresholds: %s", thresholds)
    else:
        # Fall back: run a quick pass on the eval split to derive thresholds
        logger.warning(
            "No thresholds in checkpoint; deriving from the evaluation split itself. "
            "For production use, thresholds should come from the validation set."
        )
        tmp_ds = HallucinationRiskDataset(
            metadata_csv=args.metadata_csv,
            split=args.split,
            rendered_dir=args.rendered_dir,
            augment=False,
        )
        tmp_loader = DataLoader(tmp_ds, batch_size=32, shuffle=False, num_workers=0)
        tmp_logits, tmp_labels = _run_inference(model, tmp_loader, device)
        tmp_probs = calibrator.predict(tmp_logits)
        thresholds = calibrator.get_thresholds(
            tmp_probs, tmp_labels, target_false_safe_rate=0.05
        )
        logger.info("Derived thresholds: %s", thresholds)

    # Run evaluation
    evaluate(
        model=model,
        calibrator=calibrator,
        metadata_csv=args.metadata_csv,
        rendered_dir=args.rendered_dir,
        thresholds=thresholds,
        device=device,
        output_dir=args.output_dir,
        split=args.split,
    )
