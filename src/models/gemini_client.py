"""
Gemini API client with error handling and retry logic
"""

import json
import time
from typing import Dict, Any, Optional
import google.generativeai as genai

from ..config.settings import settings
from ..config import SUPPORTED_MODELS
from ..config.structured_output import (
    DOCUMENT_CLASSIFICATION_SCHEMA,
    validate_schema_response,
)
from ..core.exceptions import (
    ClassificationError,
    APIQuotaExceededError,
    ModelUnavailableError,
)


class GeminiClient:
    """Production Gemini API client with robust error handling"""

    def __init__(self, model_name: str):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Validate model
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Available: {list(SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_config = SUPPORTED_MODELS[model_name]
        self.model_instance = genai.GenerativeModel(self.model_config["name"])

        # Create generation config
        params = self.model_config.get("parameters", {})
        self.generation_config = genai.GenerationConfig(
            **params,
            response_mime_type="application/json",
            response_schema=DOCUMENT_CLASSIFICATION_SCHEMA,
        )

    def classify_document(
        self, prompt: str, image_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify document with retry logic and error handling
        """
        for attempt in range(settings.RETRY_ATTEMPTS):
            try:
                start_time = time.time()

                response = self.model_instance.generate_content(
                    [prompt, image_content],
                    generation_config=self.generation_config,
                )

                if not response.parts:
                    raise ClassificationError("Empty response from model")

                result_json = json.loads(response.text)

                # Validate response schema
                if not validate_schema_response(result_json):
                    raise ClassificationError("Response does not match required schema")

                processing_time_ms = (time.time() - start_time) * 1000

                return {
                    "document_type": result_json.get("document_type", "other"),
                    "confidence": result_json.get("confidence", 0.0),
                    "processing_time_ms": processing_time_ms,
                    "model_used": self.model_name,
                    "attempt": attempt + 1,
                }

            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON response from {self.model_name}: {str(e)}"
                if attempt == settings.RETRY_ATTEMPTS - 1:
                    raise ClassificationError(error_msg)
                time.sleep(settings.RETRY_DELAY)

            except Exception as e:
                error_msg = str(e).lower()

                # Check for quota/rate limiting
                if any(term in error_msg for term in ["quota", "rate", "limit", "429"]):
                    if attempt == settings.RETRY_ATTEMPTS - 1:
                        raise APIQuotaExceededError(
                            f"API quota exceeded for {self.model_name}"
                        )
                    time.sleep(
                        settings.RETRY_DELAY * (attempt + 1)
                    )  # Exponential backoff
                    continue

                # Check for model availability
                if any(term in error_msg for term in ["unavailable", "503", "502"]):
                    if attempt == settings.RETRY_ATTEMPTS - 1:
                        raise ModelUnavailableError(
                            f"Model {self.model_name} unavailable"
                        )
                    time.sleep(settings.RETRY_DELAY)
                    continue

                # Other errors - retry once then fail
                if attempt == settings.RETRY_ATTEMPTS - 1:
                    raise ClassificationError(
                        f"Classification failed with {self.model_name}: {str(e)}"
                    )
                time.sleep(settings.RETRY_DELAY)

        raise ClassificationError(
            f"All {settings.RETRY_ATTEMPTS} attempts failed for {self.model_name}"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info"""
        return {
            "name": self.model_name,
            "config": self.model_config,
            "parameters": self.model_config.get("parameters", {}),
            "description": self.model_config.get("description", ""),
            "recommended_use": self.model_config.get("recommended_use", ""),
        }
