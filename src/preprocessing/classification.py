"""
Classification-specific preprocessing optimized for speed and document type recognition
"""

from typing import Dict, Any
from .base import BasePreprocessor


class ClassificationPreprocessor(BasePreprocessor):
    """Preprocessor optimized for document classification"""

    def __init__(self):
        super().__init__()
        # Classification-specific optimizations
        self.resize_dim = 1024  # Smaller size for faster classification
        self.image_quality = 80  # Good balance of quality vs speed
        self.pdf_dpi = 150  # Sufficient for document type recognition

    def preprocess(self, file_path: str) -> Dict[str, Any]:
        """
        Preprocess document for classification

        Optimizations for classification:
        - Aggressive compression for speed
        - First page only for PDFs (document type usually visible on page 1)
        - Lower DPI but sufficient for text/layout recognition
        """
        # Validate document first
        self.validate_document(file_path)

        # Prepare optimized content
        image_content = self._prepare_image_content(file_path)

        return {
            "content": image_content,
            "optimization_strategy": "classification",
            "optimizations_applied": {
                "resize_dimension": self.resize_dim,
                "image_quality": self.image_quality,
                "pdf_dpi": self.pdf_dpi,
                "pdf_pages": "first_page_only",
            },
        }
