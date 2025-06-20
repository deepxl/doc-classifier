"""
Core classification functionality
"""

from .document_classifier import UltraFastDocumentClassifier
from .vertex_ai_document_classifier import VertexAIDocumentClassifier
from .types import ClassificationResult

__all__ = [
    "UltraFastDocumentClassifier",
    "VertexAIDocumentClassifier", 
    "ClassificationResult",
]
