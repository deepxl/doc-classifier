"""
Custom exceptions for document processing pipeline
"""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors"""

    pass


class PreprocessingError(DocumentProcessingError):
    """Raised when document preprocessing fails"""

    pass


class ClassificationError(DocumentProcessingError):
    """Raised when document classification fails"""

    pass


class ParsingError(DocumentProcessingError):
    """Raised when document parsing fails"""

    pass


class ModelUnavailableError(DocumentProcessingError):
    """Raised when all model fallbacks are exhausted"""

    pass


class InvalidDocumentError(DocumentProcessingError):
    """Raised when document format is invalid or unsupported"""

    pass


class APIQuotaExceededError(DocumentProcessingError):
    """Raised when API quota is exceeded"""

    pass
