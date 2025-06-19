"""
Document parser for extracting structured data (Step 3 of pipeline)
"""

import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from ..models import ModelFallbackStrategy
from ..config import PROMPTS
from ..config.settings import settings
from ..core.exceptions import ParsingError


@dataclass
class ParsingResult:
    """Document parsing result with extracted data"""

    extracted_data: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    model_used: str
    model_tier: str
    fallback_used: bool
    inference_id: str


class DocumentParser:
    """
    Production document parser for data extraction (Step 3 of pipeline)

    Features:
    - Works with preprocessed content (preprocessing handled externally)
    - Model fallback strategy (2.0 Flash → 2.0 Flash-Lite)
    - Structured data extraction with validation
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

    def parse(self, content: Union[str, bytes, Dict[str, Any]], document_type: str) -> ParsingResult:
        """
        Parse preprocessed document content to extract structured data

        Args:
            content: Preprocessed document content (base64 image, text, or content dict)
            document_type: Document type from classification step

        Returns:
            ParsingResult with extracted data and metadata

        Raises:
            ParsingError: If parsing fails
        """
        start_time = time.time()
        inference_id = f"parse_{int(time.time() * 1000)}"

        try:
            # Create parsing prompt based on document type
            parsing_prompt = self._create_parsing_prompt(document_type)

            # Parse with fallback (placeholder for now)
            # TODO: Implement actual parsing logic with Gemini
            # result = self.fallback_strategy.parse_with_fallback(parsing_prompt, content)

            total_time = (time.time() - start_time) * 1000

            # Placeholder result
            return ParsingResult(
                extracted_data={
                    "status": "placeholder",
                    "document_type": document_type,
                },
                confidence=0.95,
                processing_time_ms=total_time,
                model_used="gemini-2.0-flash",
                model_tier="primary",
                fallback_used=False,
                inference_id=inference_id,
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            raise ParsingError(
                f"Parsing failed after {total_time:.0f}ms: {str(e)}"
            )

    def _create_parsing_prompt(self, document_type: str) -> str:
        """Create document-type specific parsing prompt"""
        # TODO: Implement document-type specific prompts
        return f"Extract structured data from this {document_type} document."

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from fallback strategy"""
        return self.fallback_strategy.get_statistics()

    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.fallback_strategy.reset_statistics()
