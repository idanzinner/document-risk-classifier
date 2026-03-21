"""
calibrator.py — Post-hoc temperature scaling calibrator.

Fits a scalar temperature parameter on validation logits to minimize
negative log-likelihood, then derives safe/risky decision thresholds
such that the false-safe rate (safe predictions on truly risky pages)
does not exceed a user-specified tolerance.
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # numerically stable sigmoid


class TemperatureCalibrator:
    """
    Post-hoc temperature scaling for binary classifiers.

    Calibrated probability = sigmoid(logit / T), where T is found by
    minimising binary cross-entropy on a held-out validation set.

    Decision boundaries:
      - prob < T_low  → safe_for_extraction
      - T_low ≤ prob ≤ T_high → review
      - prob > T_high → high_hallucination_risk

    Usage:
        cal = TemperatureCalibrator()
        cal.calibrate(val_logits, val_labels)
        probs = cal.predict(test_logits)
        thresholds = cal.get_thresholds(probs, val_labels, target_false_safe_rate=0.05)
        cal.save('checkpoints/dit/calibrator.pt')
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.t_low: float | None = None
        self.t_high: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return expit(x)

    @staticmethod
    def _bce(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
        """Binary cross-entropy for given temperature (scalar)."""
        eps = 1e-7
        probs = expit(logits / temperature)
        probs = np.clip(probs, eps, 1.0 - eps)
        return -np.mean(
            labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Fits temperature parameter T via NLL minimisation on a validation set.

        Calibrated probability: sigmoid(logit / T)

        Args:
            logits: np.ndarray of shape [N] — raw model logits.
            labels: np.ndarray of shape [N] — binary int labels (0 or 1).
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        result = minimize_scalar(
            fun=lambda t: self._bce(logits, labels, t),
            bounds=(0.1, 10.0),
            method="bounded",
        )
        self.temperature = float(result.x)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """
        Returns calibrated probabilities using the fitted temperature.

        Args:
            logits: np.ndarray of shape [N].

        Returns:
            probs: np.ndarray of shape [N], values in [0, 1].
        """
        logits = np.asarray(logits, dtype=np.float64)
        return self._sigmoid(logits / self.temperature)

    def get_thresholds(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        target_false_safe_rate: float = 0.05,
    ) -> dict[str, float]:
        """
        Derives T_low and T_high for safe / review / risky routing.

        T_low is the highest threshold at which calling a page "safe"
        (prob < T_low) keeps the fraction of truly risky pages labelled safe
        at or below *target_false_safe_rate*.

        T_high is placed halfway between T_low and 1.0, keeping the review
        band operationally manageable while catching most risky docs.

        Args:
            probs: Calibrated probabilities [N], values in [0, 1].
            labels: Binary ground-truth labels [N] (0 = safe, 1 = risky).
            target_false_safe_rate: Maximum tolerated fraction of risky pages
                                    predicted as safe (default 0.05 = 5 %).

        Returns:
            Dict {'T_low': float, 'T_high': float}.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        risky_mask = labels == 1
        n_risky = risky_mask.sum()

        if n_risky == 0:
            # Edge case: no risky examples — set conservative defaults.
            t_low = 0.0
            t_high = t_low + (1.0 - t_low) / 2.0
            self.t_low = t_low
            self.t_high = t_high
            return {"T_low": t_low, "T_high": t_high}

        # Scan candidate thresholds from high to low; pick the highest
        # T_low where false-safe rate ≤ target.
        candidate_thresholds = np.sort(np.unique(probs))[::-1]

        t_low = 0.0  # conservative default: never call anything safe
        for thresh in candidate_thresholds:
            predicted_safe = probs < thresh
            false_safe_rate = predicted_safe[risky_mask].mean()
            if false_safe_rate <= target_false_safe_rate:
                t_low = float(thresh)
                break

        # T_high sits halfway between T_low and 1.0.
        t_high = t_low + (1.0 - t_low) / 2.0

        self.t_low = t_low
        self.t_high = t_high
        return {"T_low": t_low, "T_high": t_high}

    def save(self, path: str) -> None:
        """Serialises temperature and threshold state to path using pickle."""
        state = {
            "temperature": self.temperature,
            "t_low": self.t_low,
            "t_high": self.t_high,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Restores temperature and threshold state from path."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.temperature = state["temperature"]
        self.t_low = state.get("t_low")
        self.t_high = state.get("t_high")
