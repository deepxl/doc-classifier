from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ClassificationResult",
]


@dataclass
class ClassificationResult:
    """Generic document classification result shared by all classifiers"""

    document_type: str
    confidence: float
    processing_time_ms: float
    model_used: str
    inference_id: str
    # Optional fields used by some classifier variants
    model_tier: Optional[str] = None  # e.g. "primary" or "fallback"
    fallback_used: Optional[bool] = None 