"""
Production settings for document processing pipeline
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env.local")

# Fix gRPC fork handler issues in parallel processing
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"


class Settings:
    """Production settings configuration"""

    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model Configuration
    PRIMARY_MODEL: str = "gemini-2.0-flash"
    FALLBACK_MODEL: str = "gemini-2.0-flash-lite"

    # Performance Settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))

    # File Processing
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    SUPPORTED_FORMATS: List[str] = ["jpg", "jpeg", "png", "pdf"]

    # Image Optimization
    IMAGE_RESIZE_THRESHOLD_KB: int = int(os.getenv("IMAGE_RESIZE_THRESHOLD_KB", "1000"))
    IMAGE_RESIZE_DIMENSION: int = int(os.getenv("IMAGE_RESIZE_DIMENSION", "1024"))
    IMAGE_QUALITY: int = int(os.getenv("IMAGE_QUALITY", "80"))
    PDF_DPI: int = int(os.getenv("PDF_DPI", "150"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    @classmethod
    def validate(cls) -> None:
        """Validate required settings"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")

        if cls.MAX_WORKERS < 1:
            raise ValueError("MAX_WORKERS must be >= 1")

        if cls.REQUEST_TIMEOUT < 1:
            raise ValueError("REQUEST_TIMEOUT must be >= 1")


# Production vs Development configs
class ProductionSettings(Settings):
    """Production-specific settings"""

    LOG_LEVEL: str = "WARNING"
    DEBUG: bool = False
    MAX_WORKERS: int = 20


class DevelopmentSettings(Settings):
    """Development-specific settings"""

    LOG_LEVEL: str = "DEBUG"
    DEBUG: bool = True
    MAX_WORKERS: int = 5


def get_settings() -> Settings:
    """Get settings based on environment"""
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return ProductionSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()
