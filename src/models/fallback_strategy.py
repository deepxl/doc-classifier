"""
Model fallback strategy for handling primary model failures
"""

from typing import Dict, Any, List, Optional
import logging

from .gemini_client import GeminiClient
from ..config.settings import settings
from ..core.exceptions import (
    ModelUnavailableError,
    APIQuotaExceededError,
    ClassificationError,
)

logger = logging.getLogger(__name__)


class ModelFallbackStrategy:
    """Handles fallback between primary and backup models"""

    def __init__(
        self, primary_model: Optional[str] = None, fallback_model: Optional[str] = None
    ):
        self.primary_model = (
            primary_model if primary_model is not None else settings.PRIMARY_MODEL
        )
        self.fallback_model = (
            fallback_model if fallback_model is not None else settings.FALLBACK_MODEL
        )

        # Initialize clients
        self.primary_client: Optional[GeminiClient] = None
        self.fallback_client: Optional[GeminiClient] = None

        # Statistics
        self.stats = {
            "primary_attempts": 0,
            "primary_successes": 0,
            "fallback_attempts": 0,
            "fallback_successes": 0,
            "total_failures": 0,
        }

    def _get_primary_client(self) -> GeminiClient:
        """Get or create primary model client"""
        if self.primary_client is None:
            self.primary_client = GeminiClient(self.primary_model)
        return self.primary_client

    def _get_fallback_client(self) -> GeminiClient:
        """Get or create fallback model client"""
        if self.fallback_client is None:
            self.fallback_client = GeminiClient(self.fallback_model)
        return self.fallback_client

    def classify_with_fallback(
        self, prompt: str, image_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempt classification with primary model, fallback to secondary on failure
        """
        # Try primary model first
        try:
            logger.debug(
                f"Attempting classification with primary model: {self.primary_model}"
            )
            self.stats["primary_attempts"] += 1

            client = self._get_primary_client()
            result = client.classify_document(prompt, image_content)

            self.stats["primary_successes"] += 1
            result["fallback_used"] = False
            result["model_tier"] = "primary"

            logger.debug(f"Primary model {self.primary_model} succeeded")
            return result

        except (ModelUnavailableError, APIQuotaExceededError) as e:
            logger.warning(
                f"Primary model {self.primary_model} failed: {str(e)}. Trying fallback..."
            )

            # Try fallback model
            try:
                self.stats["fallback_attempts"] += 1

                client = self._get_fallback_client()
                result = client.classify_document(prompt, image_content)

                self.stats["fallback_successes"] += 1
                result["fallback_used"] = True
                result["model_tier"] = "fallback"
                result["primary_failure_reason"] = str(e)

                logger.info(
                    f"Fallback model {self.fallback_model} succeeded after primary failure"
                )
                return result

            except Exception as fallback_error:
                logger.error(
                    f"Both primary and fallback models failed. Primary: {str(e)}, Fallback: {str(fallback_error)}"
                )
                self.stats["total_failures"] += 1
                raise ModelUnavailableError(
                    f"Both models failed. Primary ({self.primary_model}): {str(e)}. "
                    f"Fallback ({self.fallback_model}): {str(fallback_error)}"
                )

        except ClassificationError as e:
            # For classification errors, don't use fallback - likely a data issue
            logger.error(f"Classification error with primary model: {str(e)}")
            self.stats["total_failures"] += 1
            raise e

        except Exception as e:
            logger.error(f"Unexpected error with primary model: {str(e)}")
            self.stats["total_failures"] += 1
            raise ClassificationError(f"Unexpected error: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback strategy statistics"""
        total_attempts = (
            self.stats["primary_attempts"] + self.stats["fallback_attempts"]
        )
        total_successes = (
            self.stats["primary_successes"] + self.stats["fallback_successes"]
        )

        return {
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "success_rate": (
                total_successes / total_attempts if total_attempts > 0 else 0
            ),
            "primary_success_rate": (
                self.stats["primary_successes"] / self.stats["primary_attempts"]
                if self.stats["primary_attempts"] > 0
                else 0
            ),
            "fallback_usage_rate": (
                self.stats["fallback_attempts"] / total_attempts
                if total_attempts > 0
                else 0
            ),
            "detailed_stats": self.stats.copy(),
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self.stats = {
            "primary_attempts": 0,
            "primary_successes": 0,
            "fallback_attempts": 0,
            "fallback_successes": 0,
            "total_failures": 0,
        }
