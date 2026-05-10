#!/usr/bin/env python3
"""
Retune the decision thresholds of each trained model using a cost-weighted
criterion rather than the Phase-5 "FSR ≤ 5%" rule.

Cost model (default, matching the project's safety goal):

    cost(τ) = fn_cost · FN(τ) + fp_cost · FP(τ)

with fn_cost = 10, fp_cost = 1  — "missing a risky document is 10× worse
than flagging a safe one".

For each model we:
    1. Load the best checkpoint and the calibrator pkl.
    2. Read stored val_logits + val_labels.
    3. Recompute calibrated val_probs via the fitted temperature.
    4. Find τ* that minimises the cost on the val set.
    5. Update checkpoint["thresholds"] and calibrator.{t_low,t_high}
       so inference picks up the new policy immediately.
    6. Print a before/after comparison on test_* if present.
    7. Back up the originals so the change is trivially reversible.

Run from the repo root:
    python scripts/retune_thresholds.py
    python scripts/retune_thresholds.py --fn-cost 1 --fp-cost 10   # flipped
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.calibrator import TemperatureCalibrator  # noqa: E402


MODELS = [
    {
        "name":       "ResNet50",
        "checkpoint": ROOT / "checkpoints" / "baseline" / "best_resnet50.pt",
        "calibrator": ROOT / "checkpoints" / "baseline" / "calibrator_resnet50.pkl",
    },
    {
        "name":       "ViT-Base",
        "checkpoint": ROOT / "checkpoints" / "baseline" / "best_vit_base_patch16_224.pt",
        "calibrator": ROOT / "checkpoints" / "baseline" / "calibrator_vit_base_patch16_224.pkl",
    },
    {
        "name":       "DiT",
        "checkpoint": ROOT / "checkpoints" / "dit" / "best_model.pt",
        "calibrator": ROOT / "checkpoints" / "dit" / "calibrator.pkl",
    },
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _evaluate_at(threshold: float, probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n = tp + tn + fp + fn
    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float((tp + tn) / n) if n else float("nan"),
        "fsr":      float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "precision_risky": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "recall_risky":    float(tp / (tp + fn)) if (tp + fn) else float("nan"),
    }


def _load_calibrator(path: Path) -> TemperatureCalibrator:
    """Support both pkl formats (whole object or state dict)."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, TemperatureCalibrator):
        return obj
    cal = TemperatureCalibrator()
    cal.temperature = float(obj["temperature"])
    cal.t_low  = obj.get("t_low")
    cal.t_high = obj.get("t_high")
    return cal


def _save_calibrator(path: Path, cal: TemperatureCalibrator) -> None:
    """Persist calibrator as a whole object (matches load_pipeline's primary
    format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(cal, fh)


def retune(fn_cost: float, fp_cost: float, dry_run: bool) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = ROOT / "checkpoints" / f"thresholds_backup_{timestamp}"

    if not dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)

    out_records: list[dict] = []

    for m in MODELS:
        name = m["name"]
        print(f"\n=== {name} ===")
        ck   = torch.load(str(m["checkpoint"]), map_location="cpu",
                          weights_only=False)
        cal  = _load_calibrator(m["calibrator"])

        temp = float(cal.temperature)
        val_logits = np.asarray(ck["val_logits"]).ravel()
        val_labels = np.asarray(ck["val_labels"]).ravel().astype(int)
        val_probs  = _sigmoid(val_logits / temp)

        test_probs = test_labels = None
        if "test_logits" in ck and "test_labels" in ck:
            test_probs  = _sigmoid(np.asarray(ck["test_logits"]).ravel() / temp)
            test_labels = np.asarray(ck["test_labels"]).ravel().astype(int)

        # ---------- before ----------
        old_thresholds = dict(ck.get("thresholds", {}))
        old_t_low = float(old_thresholds.get("T_low", cal.t_low or 0.5))
        before_binary_val  = _evaluate_at(0.5, val_probs, val_labels)
        before_binary_test = _evaluate_at(0.5, test_probs, test_labels) if test_probs is not None else None
        before_tlow_val    = _evaluate_at(old_t_low, val_probs, val_labels)
        before_tlow_test   = _evaluate_at(old_t_low, test_probs, test_labels) if test_probs is not None else None

        # ---------- cost-weighted ----------
        result = cal.get_cost_weighted_thresholds(
            probs=val_probs, labels=val_labels,
            fn_cost=fn_cost, fp_cost=fp_cost,
        )
        tau = result["tau"]
        new_t_low  = result["T_low"]
        new_t_high = result["T_high"]

        after_val  = _evaluate_at(tau, val_probs, val_labels)
        after_test = _evaluate_at(tau, test_probs, test_labels) if test_probs is not None else None

        print(f"  temperature      = {temp:.4f}")
        print(f"  old T_low/T_high = {old_t_low:.4f} / "
              f"{old_thresholds.get('T_high', float('nan'))}")
        print(f"  NEW τ* (T_low)   = {new_t_low:.4f}   (T_high = {new_t_high:.4f})")
        print(f"  cost @ τ*        = {result['total_cost']:.1f}  "
              f"(fn={result['fn']}, fp={result['fp']})")

        def _pp(label: str, s: dict | None):
            if s is None: return
            print(f"  {label}: τ={s['threshold']:.3f}  "
                  f"acc={s['accuracy']:.1%}  FSR={s['fsr']:.1%}  "
                  f"TP={s['tp']} FN={s['fn']} FP={s['fp']} TN={s['tn']}")

        print("  -- before (0.5 binary) --")
        _pp("val ", before_binary_val)
        _pp("test", before_binary_test)
        print(f"  -- before (T_low={old_t_low:.3f}) --")
        _pp("val ", before_tlow_val)
        _pp("test", before_tlow_test)
        print("  -- after (cost-weighted τ*) --")
        _pp("val ", after_val)
        _pp("test", after_test)

        out_records.append({
            "model":           name,
            "temperature":     temp,
            "old_thresholds":  old_thresholds,
            "new_thresholds":  {"T_low": new_t_low, "T_high": new_t_high, "tau": tau},
            "cost_model":      {"fn_cost": fn_cost, "fp_cost": fp_cost,
                                "total_cost": result["total_cost"]},
            "before": {
                "binary_val":   before_binary_val,
                "binary_test":  before_binary_test,
                "tlow_val":     before_tlow_val,
                "tlow_test":    before_tlow_test,
            },
            "after": {
                "val":  after_val,
                "test": after_test,
            },
        })

        if dry_run:
            continue

        # Backup calibrator + checkpoint-thresholds JSON for easy rollback.
        shutil.copy2(m["calibrator"], backup_dir / m["calibrator"].name)
        with open(backup_dir / f"{m['calibrator'].stem}_thresholds.json", "w") as fh:
            json.dump({"old_thresholds": old_thresholds,
                       "temperature": temp}, fh, indent=2)

        # Update calibrator pkl and checkpoint in place.
        _save_calibrator(m["calibrator"], cal)
        ck["thresholds"] = {"T_low": float(new_t_low), "T_high": float(new_t_high)}
        ck["thresholds_cost_model"] = {
            "fn_cost":    float(fn_cost),
            "fp_cost":    float(fp_cost),
            "tau":        float(tau),
            "total_cost": float(result["total_cost"]),
            "retuned_at": timestamp,
        }
        torch.save(ck, m["checkpoint"])
        print(f"  ✓ wrote new thresholds to {m['checkpoint'].name} "
              f"and {m['calibrator'].name}")

    if not dry_run:
        summary_path = backup_dir / "retune_summary.json"
        with open(summary_path, "w") as fh:
            json.dump({
                "fn_cost":  fn_cost,
                "fp_cost":  fp_cost,
                "records":  out_records,
            }, fh, indent=2, default=float)
        print(f"\nBackups + summary → {backup_dir.relative_to(ROOT)}")

    return {"fn_cost": fn_cost, "fp_cost": fp_cost, "records": out_records}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fn-cost", type=float, default=10.0,
                   help="Penalty for one false-negative (default 10).")
    p.add_argument("--fp-cost", type=float, default=1.0,
                   help="Penalty for one false-positive (default 1).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute new thresholds but don't overwrite checkpoints.")
    args = p.parse_args()

    print(f"Cost model: FN cost = {args.fn_cost}, FP cost = {args.fp_cost}  "
          f"(FN is {args.fn_cost / args.fp_cost:.1f}× worse than FP)")
    if args.dry_run:
        print("DRY RUN — nothing will be written to disk.")
    retune(args.fn_cost, args.fp_cost, args.dry_run)


if __name__ == "__main__":
    main()
