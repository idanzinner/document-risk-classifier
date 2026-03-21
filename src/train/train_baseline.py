"""
train_baseline.py — Training loop for ResNet and ViT baseline models.

Supports:
  - BCEWithLogitsLoss binary training
  - Grouped split data loading from metadata.csv
  - Early stopping on validation F1
  - Checkpoint saving (best model by val F1)
  - Config-driven via configs/baseline.yaml
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import HallucinationRiskDataset
from src.models.calibrator import TemperatureCalibrator
from src.models.resnet_baseline import ResNetClassifier
from src.models.vit_baseline import ViTClassifier
from src.utils.logging import TrainingLogger
from src.utils.metrics import compute_metrics, compute_per_institution_metrics

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: str, run_name: str = "train_baseline") -> None:
    """Configure root logger with console and file handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    fh = logging.FileHandler(log_path / f"{run_name}.log")
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(console)
        root.addHandler(fh)
    else:
        root.addHandler(fh)


def _build_model(cfg: dict) -> nn.Module:
    """Instantiate the correct classifier from config."""
    model_name: str = cfg["model"]["name"]
    pretrained: bool = cfg["model"].get("pretrained", True)
    num_classes: int = cfg["model"].get("num_classes", 1)

    if model_name.startswith("vit"):
        return ViTClassifier(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    else:
        return ResNetClassifier(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Runs a single training epoch.

    Returns:
        Dict with keys: loss, and raw logits/labels for downstream calibration.
    """
    model.train()
    total_loss = 0.0
    all_logits: List[float] = []
    all_labels: List[float] = []

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)          # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_logits.extend(logits.detach().cpu().squeeze(1).tolist())
        all_labels.extend(labels.cpu().squeeze(1).tolist())

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "logits": np.array(all_logits, dtype=np.float32),
        "labels": np.array(all_labels, dtype=np.float32),
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """
    Evaluates model on validation loader.

    Returns:
        Dict with keys: loss, logits (np.ndarray), labels (np.ndarray).
    """
    model.eval()
    total_loss = 0.0
    all_logits: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            all_logits.extend(logits.cpu().squeeze(1).tolist())
            all_labels.extend(labels.cpu().squeeze(1).tolist())

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "logits": np.array(all_logits, dtype=np.float32),
        "labels": np.array(all_labels, dtype=np.float32),
    }


def _get_institutions(dataset: HallucinationRiskDataset) -> np.ndarray:
    """Extract institution column from the dataset's underlying DataFrame."""
    if "institution" in dataset.df.columns:
        return dataset.df["institution"].values
    return np.array(["unknown"] * len(dataset))


def train(config_path: str) -> None:
    """
    Runs the full baseline training pipeline from a YAML config.

    Args:
        config_path: Path to configs/baseline.yaml.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── paths ──────────────────────────────────────────────────────────────
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(str(log_dir))
    logger.info("Config loaded from %s", config_path)
    run_name = f"baseline_{cfg['model']['name']}"
    training_logger = TrainingLogger(log_dir=str(log_dir), run_name=run_name)
    training_logger.log_hparams(cfg)

    # ── device ─────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # ── data ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    metadata_csv = cfg.get("data", {}).get("metadata_csv", "data/metadata.csv")
    rendered_dir = cfg.get("data", {}).get("rendered_dir", "data/rendered")

    train_ds = HallucinationRiskDataset(
        metadata_csv=metadata_csv,
        split="train",
        rendered_dir=rendered_dir,
        augment=data_cfg.get("augmentation", False),
    )
    val_ds = HallucinationRiskDataset(
        metadata_csv=metadata_csv,
        split="val",
        rendered_dir=rendered_dir,
        augment=False,
    )

    batch_size: int = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # ── model ──────────────────────────────────────────────────────────────
    model = _build_model(cfg).to(device)
    logger.info("Model: %s", cfg["model"]["name"])

    # ── loss / optimizer ───────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    # ── training loop ──────────────────────────────────────────────────────
    n_epochs: int = cfg["training"]["epochs"]
    patience: int = cfg["training"]["early_stopping_patience"]

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    best_calibrator_path = checkpoint_dir / "calibrator.pkl"

    # Institutions for per-institution metrics
    val_institutions = _get_institutions(val_ds)

    for epoch in range(1, n_epochs + 1):
        logger.info("Epoch %d / %d", epoch, n_epochs)

        train_out = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_out = validate(model, val_loader, criterion, device)

        # ── calibration on this epoch's val logits ──────────────────────
        calibrator = TemperatureCalibrator()
        calibrator.calibrate(val_out["logits"], val_out["labels"])
        val_probs = calibrator.predict(val_out["logits"])
        thresholds = calibrator.get_thresholds(
            val_probs, val_out["labels"], target_false_safe_rate=0.05
        )

        metrics = compute_metrics(val_out["labels"], val_probs, thresholds)
        val_f1 = metrics.get("f1", 0.0)

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | "
            "val_roc_auc=%.4f | false_safe_rate=%.4f | review_rate=%.4f | "
            "T_low=%.4f | T_high=%.4f",
            epoch,
            train_out["loss"],
            val_out["loss"],
            val_f1,
            metrics.get("roc_auc", float("nan")),
            metrics.get("false_safe_rate", float("nan")),
            metrics.get("review_rate", float("nan")),
            thresholds["T_low"],
            thresholds["T_high"],
        )
        training_logger.log_epoch(epoch, "train", {"loss": train_out["loss"]})
        training_logger.log_epoch(epoch, "val", {
            "loss": val_out["loss"],
            **{k: float(v) if v is not None else None for k, v in metrics.items()},
            "T_low": thresholds["T_low"],
            "T_high": thresholds["T_high"],
        })

        # ── checkpoint ──────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": val_f1,
                    "metrics": metrics,
                    "thresholds": thresholds,
                    "config": cfg,
                },
                best_checkpoint_path,
            )
            with open(best_calibrator_path, "wb") as fh:
                pickle.dump(calibrator, fh)

            logger.info("  ✓ New best val F1=%.4f — checkpoint saved", val_f1)
        else:
            epochs_without_improvement += 1
            logger.info(
                "  No improvement (%d / %d)", epochs_without_improvement, patience
            )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # ── final evaluation on val using best checkpoint ──────────────────────
    logger.info("Loading best checkpoint for final evaluation …")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(best_calibrator_path, "rb") as fh:
        best_calibrator = pickle.load(fh)

    val_out_final = validate(model, val_loader, criterion, device)
    final_probs = best_calibrator.predict(val_out_final["logits"])
    final_thresholds = best_calibrator.get_thresholds(
        final_probs, val_out_final["labels"], target_false_safe_rate=0.05
    )
    final_metrics = compute_metrics(
        val_out_final["labels"], final_probs, final_thresholds
    )

    logger.info("=== Final Val Metrics (best checkpoint, epoch %d) ===", ckpt["epoch"])
    for k, v in final_metrics.items():
        logger.info("  %s: %s", k, v)

    # Per-institution breakdown
    per_inst_df = compute_per_institution_metrics(
        val_out_final["labels"], final_probs, val_institutions, final_thresholds
    )
    logger.info("Per-institution metrics:\n%s", per_inst_df.to_string())

    # Print clean summary
    print("\n=== Baseline Training Complete ===")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Best val F1:     {best_val_f1:.4f}")
    print(f"Thresholds:      T_low={final_thresholds['T_low']:.4f}  T_high={final_thresholds['T_high']:.4f}")
    print("\nFinal Val Metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.4f}")
        else:
            print(f"  {k:<25} {v}")
    print()

    training_logger.save()
    logger.info("Training log saved to %s", log_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline hallucination-risk classifier")
    p.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline.yaml config file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
