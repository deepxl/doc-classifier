"""
Document parser for extracting structured data (Step 3 of pipeline)
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..preprocessing import ParsingPreprocessor
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
    preprocessing_stats: Dict[str, Any]
    inference_id: str


class DocumentParser:
    """
    Production document parser for data extraction (Step 3 of pipeline)

    Features:
    - High-quality preprocessing optimized for parsing accuracy
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
        self.preprocessor = ParsingPreprocessor()
        self.fallback_strategy = ModelFallbackStrategy(primary_model, fallback_model)
        self.prompt = PROMPTS.get(prompt_type, PROMPTS["detailed"])

        # Validate settings
        settings.validate()

    def parse(self, file_path: str, document_type: str) -> ParsingResult:
        """
        Parse document to extract structured data

        Args:
            file_path: Path to document file
            document_type: Document type from classification step

        Returns:
            ParsingResult with extracted data and metadata

        Raises:
            ParsingError: If parsing fails
        """
        start_time = time.time()
        inference_id = f"parse_{int(time.time() * 1000)}"

        try:
            # Step 1: Preprocess document for parsing (higher quality)
            preprocessing_start = time.time()
            preprocessed = self.preprocessor.preprocess(file_path)
            preprocessing_time = (time.time() - preprocessing_start) * 1000

            # Step 2: Create parsing prompt based on document type
            parsing_prompt = self._create_parsing_prompt(document_type)

            # Step 3: Parse with fallback (placeholder for now)
            parsing_start = time.time()
            # TODO: Implement actual parsing logic with Gemini
            # result = self.fallback_strategy.parse_with_fallback(parsing_prompt, preprocessed["content"])
            parsing_time = (time.time() - parsing_start) * 1000

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
                preprocessing_stats={
                    "strategy": preprocessed["optimization_strategy"],
                    "optimized": preprocessed["content"].get("optimized", False),
                    "original_size_kb": preprocessed["content"].get(
                        "original_size_kb", 0
                    ),
                    "final_size_kb": preprocessed["content"].get("final_size_kb", 0),
                    "preprocessing_time_ms": preprocessing_time,
                    "parsing_time_ms": parsing_time,
                },
                inference_id=inference_id,
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            raise ParsingError(
                f"Parsing failed for {file_path} after {total_time:.0f}ms: {str(e)}"
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
