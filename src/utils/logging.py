"""
logging.py — Structured logging utilities for training and error analysis.

Provides:
  - TrainingLogger: accumulates epoch metrics and saves to JSON.
  - ErrorAnalysisLogger: accumulates per-sample prediction records and saves to CSV.
  - get_logger: convenience logger factory for module-level logging.
  - log_metrics: emits a structured metrics log entry via a standard logger.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# TrainingLogger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Structured logger for training runs.

    Accumulates epoch-level metric dicts in memory and serialises the full
    history to ``{log_dir}/{run_name}.json`` on save().
    """

    def __init__(self, log_dir: str, run_name: str) -> None:
        """
        Args:
            log_dir:  Directory where log files are written.
            run_name: Identifier for this run (used as filename stem).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.history: list[dict] = []
        self._hparams: Optional[dict] = None

    def log_epoch(self, epoch: int, phase: str, metrics: dict) -> None:
        """
        Appends one epoch record to the history.

        Args:
            epoch:   Epoch number (0-indexed or 1-indexed — caller's choice).
            phase:   Phase identifier, e.g. 'train' or 'val'.
            metrics: Dict of metric name → scalar value.
        """
        record = {"epoch": epoch, "phase": phase, **metrics}
        self.history.append(record)

    def log_hparams(self, hparams: dict) -> None:
        """
        Saves hyperparameters to ``{log_dir}/{run_name}_hparams.json``.

        Args:
            hparams: Dict of hyperparameter name → value.
        """
        self._hparams = hparams
        hparams_path = self.log_dir / f"{self.run_name}_hparams.json"
        with open(hparams_path, "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2, ensure_ascii=False)

    def save(self) -> None:
        """Writes the accumulated history to ``{log_dir}/{run_name}.json``."""
        history_path = self.log_dir / f"{self.run_name}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ErrorAnalysisLogger
# ---------------------------------------------------------------------------

class ErrorAnalysisLogger:
    """
    Logs per-sample prediction details for error analysis.

    Accumulates records in memory and writes them as a CSV via pandas on
    save().  Each record corresponds to one page of a PDF document.
    """

    def __init__(self, output_path: str) -> None:
        """
        Args:
            output_path: Path for the output CSV file.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def log_sample(
        self,
        file_path: str,
        page_num: int,
        true_label: int,
        predicted_category: str,
        confidence: float,
        institution: str = "",
        template_family: str = "",
        risk_score: int = -1,
        D: int = -1,
        H: int = -1,
        S: int = -1,
        L: int = -1,
        scan_quality_note: str = "",
        handwriting_note: str = "",
    ) -> None:
        """
        Appends one sample record to the log.

        Args:
            file_path:           Source PDF file path.
            page_num:            Page number within the PDF (0-indexed).
            true_label:          Ground-truth binary label (0=safe, 1=risky).
            predicted_category:  One of 'safe_for_extraction',
                                 'high_hallucination_risk', or 'review'.
            confidence:          Model confidence score in [0, 1].
            institution:         Institution identifier (optional).
            template_family:     Template / document family (optional).
            risk_score:          Composite rubric risk score (optional, -1 = unknown).
            D:                   Rubric dimension D score (optional, -1 = unknown).
            H:                   Rubric dimension H score (optional, -1 = unknown).
            S:                   Rubric dimension S score (optional, -1 = unknown).
            L:                   Rubric dimension L score (optional, -1 = unknown).
            scan_quality_note:   Free-text scan quality note (optional).
            handwriting_note:    Free-text handwriting note (optional).
        """
        record = {
            "file_path": file_path,
            "page_num": page_num,
            "true_label": true_label,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "institution": institution,
            "template_family": template_family,
            "risk_score": risk_score,
            "D": D,
            "H": H,
            "S": S,
            "L": L,
            "scan_quality_note": scan_quality_note,
            "handwriting_note": handwriting_note,
        }
        self.records.append(record)

    def save(self) -> None:
        """Writes all accumulated records to the CSV at output_path."""
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Module-level logger utilities (kept for backward-compatibility)
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Args:
        name:    Logger name (typically __name__ of the calling module).
        log_dir: If provided, attaches a FileHandler writing to
                 ``{log_dir}/{name}_{timestamp}.log``.
        level:   Logging level (default: logging.INFO).

    Returns:
        logging.Logger instance.
    """
    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: dict,
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Emits a structured metrics log entry.

    Args:
        logger:  Logger returned by get_logger().
        metrics: Dict of metric name → value.
        step:    Optional training step or epoch number.
        prefix:  Optional prefix string (e.g. 'train/', 'val/').
    """
    parts = []
    if step is not None:
        parts.append(f"step={step}")
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, float):
            parts.append(f"{key}={v:.4f}")
        else:
            parts.append(f"{key}={v}")
    logger.info("  ".join(parts))
