"""
Configuration package for the document classifier
"""

from .categories import DOCUMENT_CATEGORIES
from .prompts import PROMPTS
from .models import SUPPORTED_MODELS

__all__ = ["DOCUMENT_CATEGORIES", "PROMPTS", "SUPPORTED_MODELS"]
