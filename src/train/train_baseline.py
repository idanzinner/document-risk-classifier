"""
train_baseline.py — Training loop for ResNet and ViT baseline models.

Supports:
  - BCEWithLogitsLoss binary training
  - Grouped split data loading from metadata.csv
  - Early stopping on validation F1
  - Checkpoint saving (best model by val F1)
  - Config-driven via configs/baseline.yaml
  - --model flag to override config model name at the CLI

Usage:
    python -m src.train.train_baseline --config configs/baseline.yaml
    python -m src.train.train_baseline --config configs/baseline.yaml --model vit_base_patch16_224
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.dataset import HallucinationRiskDataset
from src.models.calibrator import TemperatureCalibrator
from src.models.resnet_baseline import ResNetClassifier
from src.models.vit_baseline import ViTClassifier
from src.utils.device import get_device, prepare_model, prepare_input, mps_sync, mps_empty_cache
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
    MAX_GRAD_NORM = 1.0
    model.train()
    total_loss = 0.0
    all_logits: List[float] = []
    all_labels: List[float] = []

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = prepare_input(images, device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)          # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        mps_sync()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        batch_loss = loss.item()
        if not np.isfinite(batch_loss):
            logger.warning("NaN/Inf loss detected — skipping batch")
            continue
        total_loss += batch_loss * images.size(0)
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
    Evaluates model on a loader.

    Returns:
        Dict with keys: loss, logits (np.ndarray), labels (np.ndarray).
    """
    model.eval()
    total_loss = 0.0
    all_logits: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images = prepare_input(images, device)
            labels = labels.to(device)

            logits = model(images)
            mps_sync()
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


def train(config_path: str, model_override: str | None = None) -> None:
    """
    Runs the full baseline training pipeline from a YAML config.

    Args:
        config_path:    Path to configs/baseline.yaml.
        model_override: Optional timm model name that overrides cfg["model"]["name"].
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if model_override:
        cfg["model"]["name"] = model_override

    # ── paths ──────────────────────────────────────────────────────────────
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name: str = cfg["model"]["name"]
    run_name = f"baseline_{model_name.replace('/', '_')}"
    _setup_logging(str(log_dir), run_name=run_name)
    logger.info("Config loaded from %s", config_path)
    training_logger = TrainingLogger(log_dir=str(log_dir), run_name=run_name)
    training_logger.log_hparams(cfg)

    # ── device ─────────────────────────────────────────────────────────────
    device_pref = cfg["training"].get("device", "auto")
    device = get_device(device_pref)
    logger.info("Using device: %s  (preference=%s)", device, device_pref)

    # ── data ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    metadata_csv = data_cfg.get("metadata_csv", "data/metadata.csv")
    rendered_dir = data_cfg.get("rendered_dir", "data/rendered_pages")

    train_ds = HallucinationRiskDataset(
        metadata_csv=metadata_csv,
        split="train",
        rendered_dir=rendered_dir,
        augment=data_cfg.get("augmentation", False),
    )
    # No-augment train loader used for final inference pass
    train_ds_eval = HallucinationRiskDataset(
        metadata_csv=metadata_csv,
        split="train",
        rendered_dir=rendered_dir,
        augment=False,
    )
    val_ds = HallucinationRiskDataset(
        metadata_csv=metadata_csv,
        split="val",
        rendered_dir=rendered_dir,
        augment=False,
    )
    test_ds = HallucinationRiskDataset(
        metadata_csv=metadata_csv,
        split="test",
        rendered_dir=rendered_dir,
        augment=False,
    )

    batch_size: int = cfg["training"]["batch_size"]

    # ── class-imbalance: WeightedRandomSampler (optional) ──────────────────
    train_sampler = None
    use_weighted_sampler = cfg["training"].get("use_weighted_sampler", False)
    if use_weighted_sampler:
        train_labels = train_ds.df["label_binary"].values.astype(float)
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            class_weight = {0: 1.0 / n_neg, 1: 1.0 / n_pos}
            sample_weights = torch.tensor(
                [class_weight[int(lb)] for lb in train_labels], dtype=torch.float
            )
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            logger.info(
                "WeightedRandomSampler: n_neg=%d  n_pos=%d  weight_neg=%.4f  weight_pos=%.4f",
                int(n_neg), int(n_pos), class_weight[0], class_weight[1],
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
    )
    train_eval_loader = DataLoader(
        train_ds_eval, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    logger.info(
        "Train: %d | Val: %d | Test: %d samples",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # ── model ──────────────────────────────────────────────────────────────
    model = prepare_model(_build_model(cfg), device)
    logger.info("Model: %s", model_name)

    # ── loss / optimizer ───────────────────────────────────────────────────
    # pos_weight: upweights positive (risky) class in BCEWithLogitsLoss
    pos_weight_cfg = cfg["training"].get("pos_weight", None)
    pos_weight_tensor = None
    if pos_weight_cfg == "auto":
        train_labels = train_ds.df["label_binary"].values.astype(float)
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        if n_pos > 0:
            pw_val = n_neg / n_pos
            pos_weight_tensor = torch.tensor([pw_val], dtype=torch.float).to(device)
            logger.info("pos_weight=auto → %.4f  (n_neg=%d / n_pos=%d)", pw_val, int(n_neg), int(n_pos))
    elif pos_weight_cfg is not None:
        pos_weight_tensor = torch.tensor([float(pos_weight_cfg)], dtype=torch.float).to(device)
        logger.info("pos_weight=%.4f (from config)", float(pos_weight_cfg))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
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
    best_checkpoint_path = checkpoint_dir / f"best_{model_name.replace('/', '_')}.pt"
    best_calibrator_path = checkpoint_dir / f"calibrator_{model_name.replace('/', '_')}.pkl"

    val_institutions = _get_institutions(val_ds)

    for epoch in range(1, n_epochs + 1):
        logger.info("Epoch %d / %d", epoch, n_epochs)

        train_out = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_out = validate(model, val_loader, criterion, device)
        mps_empty_cache()

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

    training_logger.save()

    # ── final evaluation on all splits using best checkpoint ──────────────
    logger.info("Loading best checkpoint for final evaluation across all splits …")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(best_calibrator_path, "rb") as fh:
        best_calibrator = pickle.load(fh)

    logger.info("Running inference on train set (no augmentation) …")
    train_out_final = validate(model, train_eval_loader, criterion, device)
    logger.info("Running inference on val set …")
    val_out_final = validate(model, val_loader, criterion, device)
    logger.info("Running inference on test set …")
    test_out_final = validate(model, test_loader, criterion, device)
    mps_empty_cache()

    # Calibrate all splits with the val-fit calibrator
    train_probs = best_calibrator.predict(train_out_final["logits"])
    val_probs_final = best_calibrator.predict(val_out_final["logits"])
    test_probs = best_calibrator.predict(test_out_final["logits"])

    final_thresholds = best_calibrator.get_thresholds(
        val_probs_final, val_out_final["labels"], target_false_safe_rate=0.05
    )

    train_metrics = compute_metrics(train_out_final["labels"], train_probs, final_thresholds)
    val_metrics   = compute_metrics(val_out_final["labels"],   val_probs_final, final_thresholds)
    test_metrics  = compute_metrics(test_out_final["labels"],  test_probs,  final_thresholds)

    # Per-institution metrics on val and test
    val_institutions  = _get_institutions(val_ds)
    test_institutions = _get_institutions(test_ds)
    per_inst_val  = compute_per_institution_metrics(
        val_out_final["labels"],  val_probs_final, val_institutions,  final_thresholds
    )
    per_inst_test = compute_per_institution_metrics(
        test_out_final["labels"], test_probs,      test_institutions, final_thresholds
    )
    logger.info("Val per-institution metrics:\n%s", per_inst_val.to_string())
    logger.info("Test per-institution metrics:\n%s", per_inst_test.to_string())

    # Expand checkpoint with logits/labels for all splits
    torch.save(
        {
            **ckpt,
            "model_name": model_name,
            "temperature": best_calibrator.temperature,
            "thresholds": final_thresholds,
            # Train split
            "train_logits":  train_out_final["logits"],
            "train_labels":  train_out_final["labels"],
            "train_metrics": train_metrics,
            # Val split
            "val_logits":  val_out_final["logits"],
            "val_labels":  val_out_final["labels"],
            "val_metrics": val_metrics,
            # Test split
            "test_logits":  test_out_final["logits"],
            "test_labels":  test_out_final["labels"],
            "test_metrics": test_metrics,
        },
        best_checkpoint_path,
    )
    logger.info("Expanded checkpoint saved → %s", best_checkpoint_path)

    # Print clean summary
    print(f"\n=== Baseline Training Complete — {model_name} ===")
    print(f"Best checkpoint : {best_checkpoint_path}")
    print(f"Best val F1     : {best_val_f1:.4f}")
    print(f"Thresholds      : T_low={final_thresholds['T_low']:.4f}  T_high={final_thresholds['T_high']:.4f}")
    print(f"Temperature     : {best_calibrator.temperature:.4f}")
    for split_name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"\n{split_name} Metrics:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:.4f}")

    logger.info("Training log saved to %s", log_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline hallucination-risk classifier")
    p.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline.yaml config file",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional timm model name to override cfg['model']['name'] "
             "(e.g. vit_base_patch16_224, efficientnet_b0)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config, model_override=args.model)
