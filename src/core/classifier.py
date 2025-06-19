"""
Production-ready document classifier for pipeline integration
"""

import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from ..models import ModelFallbackStrategy
from ..config import PROMPTS
from ..config.settings import settings
from ..core.exceptions import DocumentProcessingError


@dataclass
class ClassificationResult:
    """Document classification result with metadata"""

    document_type: str
    confidence: float
    processing_time_ms: float
    model_used: str
    model_tier: str  # "primary" or "fallback"
    fallback_used: bool
    inference_id: str


class DocumentClassifier:
    """
    Production document classifier for pipeline integration

    Features:
    - Works with preprocessed content (preprocessing handled externally)
    - Model fallback strategy (2.0 Flash → 2.0 Flash-Lite)
    - Comprehensive error handling and retry logic
    - Performance metrics and monitoring
    """

    def __init__(
        self,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        prompt_type: str = "detailed",
    ):
        # Initialize components
        self.fallback_strategy = ModelFallbackStrategy(primary_model, fallback_model)
        self.prompt = PROMPTS.get(prompt_type, PROMPTS["detailed"])

        # Validate settings
        settings.validate()

    def classify(self, content: Union[str, bytes, Dict[str, Any]]) -> ClassificationResult:
        """
        Classify preprocessed document content

        Args:
            content: Preprocessed document content (base64 image, text, or content dict)

        Returns:
            ClassificationResult with classification and metadata

        Raises:
            DocumentProcessingError: If classification fails
        """
        start_time = time.time()
        inference_id = f"classify_{int(time.time() * 1000)}"

        try:
            # Classify with fallback
            result = self.fallback_strategy.classify_with_fallback(
                self.prompt, content
            )

            total_time = (time.time() - start_time) * 1000

            return ClassificationResult(
                document_type=result["document_type"],
                confidence=result["confidence"],
                processing_time_ms=total_time,
                model_used=result["model_used"],
                model_tier=result.get("model_tier", "primary"),
                fallback_used=result.get("fallback_used", False),
                inference_id=inference_id,
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            raise DocumentProcessingError(
                f"Classification failed after {total_time:.0f}ms: {str(e)}"
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from fallback strategy"""
        return self.fallback_strategy.get_statistics()

    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.fallback_strategy.reset_statistics()

    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        return {
            "primary_model": self.fallback_strategy.primary_model,
            "fallback_model": self.fallback_strategy.fallback_model,
            "prompt_type": "detailed",  # Currently using detailed prompt
            "settings": {
                "max_workers": settings.MAX_WORKERS,
                "retry_attempts": settings.RETRY_ATTEMPTS,
                "request_timeout": settings.REQUEST_TIMEOUT,
            },
        }
