"""
train_dit.py — Staged fine-tuning loop for the DiT classifier.

Implements the three-stage DiT training protocol:
  Stage 1 — frozen backbone, head-only training  (stage1_lr, stage1_epochs)
  Stage 2 — top-2 transformer blocks unfrozen    (stage2_lr, stage2_epochs)
  Stage 3 — full model fine-tuning (optional)    (stage3_lr, stage3_epochs)

Config-driven via configs/dit.yaml.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.dataset import HallucinationRiskDataset
from src.models.calibrator import TemperatureCalibrator
from src.models.dit_classifier import DiTClassifier
from src.utils.device import get_device, prepare_model, prepare_input, mps_sync, mps_empty_cache
from src.utils.logging import TrainingLogger
from src.utils.metrics import compute_metrics, compute_per_institution_metrics

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: str, run_name: str = "train_dit") -> None:
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


def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """Run one validation pass; return loss, logits, and labels."""
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


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run a single training epoch; return loss."""
    MAX_GRAD_NORM = 1.0
    model.train()
    total_loss = 0.0
    n = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = prepare_input(images, device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
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
        n += images.size(0)

    return {"loss": total_loss / max(n, 1)}


def _calibrate_and_score(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Tuple[TemperatureCalibrator, np.ndarray, dict, dict]:
    """Fit calibrator, compute probabilities, derive thresholds, return metrics."""
    calibrator = TemperatureCalibrator()
    calibrator.calibrate(logits, labels)
    probs = calibrator.predict(logits)
    thresholds = calibrator.get_thresholds(probs, labels, target_false_safe_rate=0.05)
    metrics = compute_metrics(labels, probs, thresholds)
    return calibrator, probs, thresholds, metrics


def _get_institutions(dataset: HallucinationRiskDataset) -> np.ndarray:
    """Extract institution column from the dataset's underlying DataFrame."""
    if "institution" in dataset.df.columns:
        return dataset.df["institution"].values
    return np.array(["unknown"] * len(dataset))


def run_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    n_epochs: int,
    patience: int,
    checkpoint_dir: Path,
    stage_name: str,
    training_logger=None,
) -> Dict[str, object]:
    """
    Runs one training stage with early stopping.

    Args:
        training_logger: Optional TrainingLogger; if provided, each epoch is
                         logged with phase=f'{stage_name}/train' and
                         phase=f'{stage_name}/val'.

    Returns:
        Dict with keys: best_val_f1, best_epoch, checkpoint_path, calibrator_path.
    """
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    ckpt_path = checkpoint_dir / f"best_{stage_name}.pt"
    cal_path = checkpoint_dir / f"calibrator_{stage_name}.pkl"

    for epoch in range(1, n_epochs + 1):
        logger.info("[%s] Epoch %d / %d", stage_name, epoch, n_epochs)

        train_out = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_out = _validate(model, val_loader, criterion, device)
        mps_empty_cache()

        calibrator, probs, thresholds, metrics = _calibrate_and_score(
            val_out["logits"], val_out["labels"]
        )
        val_f1 = metrics.get("f1", 0.0)

        logger.info(
            "[%s] Epoch %d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | "
            "roc_auc=%.4f | false_safe_rate=%.4f | T_low=%.4f | T_high=%.4f",
            stage_name,
            epoch,
            train_out["loss"],
            val_out["loss"],
            val_f1,
            metrics.get("roc_auc", float("nan")),
            metrics.get("false_safe_rate", float("nan")),
            thresholds["T_low"],
            thresholds["T_high"],
        )

        if training_logger is not None:
            training_logger.log_epoch(epoch, f"{stage_name}/train", {"loss": train_out["loss"]})
            training_logger.log_epoch(epoch, f"{stage_name}/val", {
                "loss": val_out["loss"],
                **{k: float(v) if v is not None else None for k, v in metrics.items()},
                "T_low": thresholds["T_low"],
                "T_high": thresholds["T_high"],
            })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "stage": stage_name,
                    "model_state_dict": model.state_dict(),
                    "val_f1": val_f1,
                    "metrics": metrics,
                    "thresholds": thresholds,
                },
                ckpt_path,
            )
            with open(cal_path, "wb") as fh:
                pickle.dump(calibrator, fh)

            logger.info("[%s] ✓ New best F1=%.4f — checkpoint saved", stage_name, val_f1)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info("[%s] Early stopping at epoch %d", stage_name, epoch)
            break

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
        "calibrator_path": str(cal_path),
    }


def train(config_path: str) -> None:
    """
    Runs the full DiT staged training pipeline from a YAML config.

    Args:
        config_path: Path to configs/dit.yaml.
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
    training_logger = TrainingLogger(log_dir=str(log_dir), run_name="dit")
    training_logger.log_hparams(cfg)

    # ── device ─────────────────────────────────────────────────────────────
    device_pref = cfg["training"].get("device", "auto")
    device = get_device(device_pref)
    logger.info("Using device: %s  (preference=%s)", device, device_pref)

    # ── data ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    metadata_csv = cfg.get("data", {}).get("metadata_csv", "data/metadata.csv")
    rendered_dir = cfg.get("data", {}).get("rendered_dir", "data/rendered_pages")
    batch_size: int = cfg["training"]["batch_size"]

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
    model_name: str = cfg["model"]["name"]
    num_classes: int = cfg["model"].get("num_classes", 1)
    model = prepare_model(DiTClassifier(model_name=model_name, num_classes=num_classes), device)
    logger.info("DiTClassifier loaded: %s", model_name)

    # ── pos_weight for BCEWithLogitsLoss ───────────────────────────────────
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
    patience: int = cfg["training"]["early_stopping_patience"]
    weight_decay: float = float(cfg["training"]["weight_decay"])

    # ── Stage 1: freeze backbone, train head only ──────────────────────────
    logger.info("=== Stage 1: head-only training ===")
    model.freeze_backbone()
    stage1_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["stage1_lr"]),
        weight_decay=weight_decay,
    )
    stage1_result = run_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=stage1_optimizer,
        criterion=criterion,
        device=device,
        n_epochs=cfg["training"]["stage1_epochs"],
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        stage_name="stage1",
        training_logger=training_logger,
    )
    logger.info("Stage 1 complete. Best F1=%.4f", stage1_result["best_val_f1"])

    # ── Stage 2: unfreeze top-2 blocks ────────────────────────────────────
    logger.info("=== Stage 2: top-2 blocks unfrozen ===")
    model.unfreeze_top_blocks(2)
    stage2_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["stage2_lr"]),
        weight_decay=weight_decay,
    )
    stage2_result = run_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=stage2_optimizer,
        criterion=criterion,
        device=device,
        n_epochs=cfg["training"]["stage2_epochs"],
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        stage_name="stage2",
        training_logger=training_logger,
    )
    logger.info("Stage 2 complete. Best F1=%.4f", stage2_result["best_val_f1"])

    # ── Stage 3: full model (only if val F1 improved) ─────────────────────
    best_f1_so_far = max(stage1_result["best_val_f1"], stage2_result["best_val_f1"])
    stage3_result: Optional[dict] = None

    logger.info("=== Stage 3: full model fine-tuning ===")
    model.unfreeze_all()
    stage3_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["stage3_lr"]),
        weight_decay=weight_decay,
    )
    stage3_result = run_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=stage3_optimizer,
        criterion=criterion,
        device=device,
        n_epochs=cfg["training"]["stage3_epochs"],
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        stage_name="stage3",
        training_logger=training_logger,
    )
    logger.info("Stage 3 complete. Best F1=%.4f", stage3_result["best_val_f1"])

    # ── Determine overall best stage ──────────────────────────────────────
    all_stages = {
        "stage1": stage1_result,
        "stage2": stage2_result,
    }
    if stage3_result is not None:
        all_stages["stage3"] = stage3_result

    best_stage = max(all_stages, key=lambda s: all_stages[s]["best_val_f1"])
    best_result = all_stages[best_stage]
    logger.info("Overall best stage: %s (F1=%.4f)", best_stage, best_result["best_val_f1"])

    # Copy best stage checkpoint as canonical best_model.pt
    best_model_path = checkpoint_dir / "best_model.pt"
    best_calibrator_path = checkpoint_dir / "calibrator.pkl"

    ckpt = torch.load(best_result["checkpoint_path"], map_location=device)
    torch.save(ckpt, best_model_path)

    with open(best_result["calibrator_path"], "rb") as fh:
        best_calibrator = pickle.load(fh)
    with open(best_calibrator_path, "wb") as fh:
        pickle.dump(best_calibrator, fh)

    # ── Final evaluation on all splits using best checkpoint ─────────────
    logger.info("Loading best checkpoint (%s) for final evaluation across all splits …", best_stage)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info("Running inference on train set (no augmentation) …")
    train_out_final = _validate(model, train_eval_loader, criterion, device)
    logger.info("Running inference on val set …")
    val_out_final = _validate(model, val_loader, criterion, device)
    logger.info("Running inference on test set …")
    test_out_final = _validate(model, test_loader, criterion, device)
    mps_empty_cache()

    # Calibrate all splits with val-fit calibrator
    train_probs = best_calibrator.predict(train_out_final["logits"])
    val_probs_final = best_calibrator.predict(val_out_final["logits"])
    test_probs = best_calibrator.predict(test_out_final["logits"])

    final_thresholds = best_calibrator.get_thresholds(
        val_probs_final, val_out_final["labels"], target_false_safe_rate=0.05
    )
    train_metrics = compute_metrics(train_out_final["labels"], train_probs, final_thresholds)
    val_metrics   = compute_metrics(val_out_final["labels"],   val_probs_final, final_thresholds)
    test_metrics  = compute_metrics(test_out_final["labels"],  test_probs,  final_thresholds)

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

    # Expand best_model.pt with logits/labels for all splits
    torch.save(
        {
            **ckpt,
            "model_name": cfg["model"]["name"],
            "best_stage": best_stage,
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
        best_model_path,
    )
    logger.info("Expanded checkpoint saved → %s", best_model_path)

    print("\n=== DiT Staged Training Complete ===")
    print(f"Best stage:      {best_stage}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Best val F1:     {best_result['best_val_f1']:.4f}")
    print(f"Temperature:     {best_calibrator.temperature:.4f}")
    print(f"Thresholds:      T_low={final_thresholds['T_low']:.4f}  T_high={final_thresholds['T_high']:.4f}")
    for split_name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"\n{split_name} Metrics:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:.4f}")
    print()

    training_logger.save()
    logger.info("Training log saved to %s", log_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged DiT training for hallucination-risk classifier")
    p.add_argument(
        "--config",
        type=str,
        default="configs/dit.yaml",
        help="Path to dit.yaml config file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
