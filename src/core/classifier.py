"""
Production-ready document classifier for pipeline integration
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..preprocessing import ClassificationPreprocessor
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
    preprocessing_stats: Dict[str, Any]
    inference_id: str


class DocumentClassifier:
    """
    Production document classifier for pipeline integration

    Features:
    - Automatic preprocessing optimization for classification
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
        self.preprocessor = ClassificationPreprocessor()
        self.fallback_strategy = ModelFallbackStrategy(primary_model, fallback_model)
        self.prompt = PROMPTS.get(prompt_type, PROMPTS["detailed"])

        # Validate settings
        settings.validate()

    def classify(self, file_path: str) -> ClassificationResult:
        """
        Classify a single document

        Args:
            file_path: Path to document file

        Returns:
            ClassificationResult with classification and metadata

        Raises:
            DocumentProcessingError: If classification fails
        """
        start_time = time.time()
        inference_id = f"classify_{int(time.time() * 1000)}"

        try:
            # Step 1: Preprocess document
            preprocessing_start = time.time()
            preprocessed = self.preprocessor.preprocess(file_path)
            preprocessing_time = (time.time() - preprocessing_start) * 1000

            # Step 2: Classify with fallback
            classification_start = time.time()
            result = self.fallback_strategy.classify_with_fallback(
                self.prompt, preprocessed["content"]
            )
            classification_time = (time.time() - classification_start) * 1000

            total_time = (time.time() - start_time) * 1000

            return ClassificationResult(
                document_type=result["document_type"],
                confidence=result["confidence"],
                processing_time_ms=total_time,
                model_used=result["model_used"],
                model_tier=result.get("model_tier", "primary"),
                fallback_used=result.get("fallback_used", False),
                preprocessing_stats={
                    "strategy": preprocessed["optimization_strategy"],
                    "optimized": preprocessed["content"].get("optimized", False),
                    "original_size_kb": preprocessed["content"].get(
                        "original_size_kb", 0
                    ),
                    "final_size_kb": preprocessed["content"].get("final_size_kb", 0),
                    "preprocessing_time_ms": preprocessing_time,
                    "classification_time_ms": classification_time,
                },
                inference_id=inference_id,
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            raise DocumentProcessingError(
                f"Classification failed for {file_path} after {total_time:.0f}ms: {str(e)}"
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
            "preprocessing_strategy": "classification",
            "settings": {
                "max_workers": settings.MAX_WORKERS,
                "retry_attempts": settings.RETRY_ATTEMPTS,
                "request_timeout": settings.REQUEST_TIMEOUT,
                "image_optimization": {
                    "resize_threshold_kb": settings.IMAGE_RESIZE_THRESHOLD_KB,
                    "resize_dimension": settings.IMAGE_RESIZE_DIMENSION,
                    "image_quality": settings.IMAGE_QUALITY,
                    "pdf_dpi": settings.PDF_DPI,
                },
            },
        }
