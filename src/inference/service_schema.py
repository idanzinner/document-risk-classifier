"""
service_schema.py — Pydantic request/response schemas for the inference service.

Defines the three-class RiskCategory enum and the structured
PredictionRequest / PredictionResponse models used by predict.py
and any future REST/gRPC service layer.
"""

from enum import Enum

from pydantic import BaseModel


class RiskCategory(str, Enum):
    """Three output categories for the hallucination-risk classifier."""

    SAFE = "safe_for_extraction"
    REVIEW = "review"
    HIGH_RISK = "high_hallucination_risk"


class PredictionRequest(BaseModel):
    """
    Input specification for a single-page prediction.

    Attributes:
        file_path: Absolute or relative path to the PDF file.
        page_num:  1-indexed page number within the PDF.
    """

    file_path: str
    page_num: int = 1


class PredictionResponse(BaseModel):
    """
    Structured prediction result for one PDF page.

    Attributes:
        file_path:      Source PDF path (echoed from request).
        page_num:       1-indexed page number (echoed from request).
        risk_category:  One of RiskCategory.{SAFE, REVIEW, HIGH_RISK}.
        confidence:     Calibrated probability in [0, 1].
        raw_logit:      Raw model output before calibration.
    """

    file_path: str
    page_num: int
    risk_category: RiskCategory
    confidence: float
    raw_logit: float

    @property
    def is_safe(self) -> bool:
        """True when the page is predicted safe for extraction."""
        return self.risk_category == RiskCategory.SAFE

    @property
    def needs_review(self) -> bool:
        """True when the page falls in the uncertain review band."""
        return self.risk_category == RiskCategory.REVIEW

    @property
    def is_high_risk(self) -> bool:
        """True when the page is predicted high hallucination risk."""
        return self.risk_category == RiskCategory.HIGH_RISK


class BatchPredictionRequest(BaseModel):
    """Batch wrapper for multiple single-page prediction requests."""

    requests: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction results with per-category summary counts.

    Attributes:
        responses: Ordered list of individual prediction results.
        summary:   Dict with counts per RiskCategory value.
    """

    responses: list[PredictionResponse]
    summary: dict

    @classmethod
    def from_responses(cls, responses: list[PredictionResponse]) -> "BatchPredictionResponse":
        """Builds a BatchPredictionResponse and computes category counts."""
        summary: dict[str, int] = {cat.value: 0 for cat in RiskCategory}
        for resp in responses:
            summary[resp.risk_category.value] += 1
        return cls(responses=responses, summary=summary)
