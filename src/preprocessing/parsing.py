"""
Parsing-specific preprocessing optimized for accuracy and detail preservation
"""

from typing import Dict, Any
from .base import BasePreprocessor


class ParsingPreprocessor(BasePreprocessor):
    """Preprocessor optimized for document parsing and data extraction"""

    def __init__(self):
        super().__init__()
        # Parsing-specific optimizations
        self.resize_dim = 1536  # Higher resolution for text clarity
        self.image_quality = 90  # Higher quality to preserve text details
        self.pdf_dpi = 200  # Higher DPI for better text recognition
        self.max_size_kb = 2000  # Allow larger files for parsing accuracy

    def preprocess(self, file_path: str) -> Dict[str, Any]:
        """
        Preprocess document for parsing and data extraction

        Optimizations for parsing:
        - Higher resolution to preserve text clarity
        - Better image quality for accurate OCR/text extraction
        - Higher PDF DPI for crisp text rendering
        - Less aggressive compression to maintain detail
        """
        # Validate document first
        self.validate_document(file_path)

        # Prepare high-quality content for parsing
        image_content = self._prepare_image_content(file_path)

        return {
            "content": image_content,
            "optimization_strategy": "parsing",
            "optimizations_applied": {
                "resize_dimension": self.resize_dim,
                "image_quality": self.image_quality,
                "pdf_dpi": self.pdf_dpi,
                "pdf_pages": "first_page_only",
                "max_size_threshold_kb": self.max_size_kb,
            },
        }
