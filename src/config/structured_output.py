"""
Google Gemini Structured Output Configuration
Optimized JSON schema for document classification
"""

import google.generativeai as genai
from .models import get_model_parameters

# 🎯 Document Classification Schema (the only one we use)
DOCUMENT_CLASSIFICATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "document_type": {
            "type": "STRING",
            "description": "The type of document identified",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence score between 0.0 and 1.0",
        },
    },
    "required": ["document_type", "confidence"],
}


def get_optimized_config_for_model(
    model_name: str = "gemini-2.0-flash",
) -> genai.types.GenerationConfig:
    """
    Get optimized generation config for specific model using model-specific parameters

    Args:
        model_name: Name of the Gemini model

    Returns:
        Optimized GenerationConfig object with model-specific parameters
    """
    # Get model-specific parameters
    params = get_model_parameters(model_name)

    # Create config with model-specific parameters + structured output
    config_dict = {
        "response_mime_type": "application/json",
        "response_schema": DOCUMENT_CLASSIFICATION_SCHEMA,
        **params,  # Unpack model-specific parameters
    }

    return genai.types.GenerationConfig(**config_dict)


def validate_schema_response(response: dict) -> bool:
    """
    Validate that response matches our schema

    Args:
        response: JSON response from model

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["document_type", "confidence"]
    return all(field in response for field in required_fields)


def create_speed_optimized_config() -> dict:
    """
    Create speed-optimized structured output config

    Returns:
        Speed-optimized schema configuration
    """
    return {
        "response_mime_type": "application/json",
        "response_schema": DOCUMENT_CLASSIFICATION_SCHEMA,
    }


# 🔧 Schema Optimization Notes
"""
Best Practices Applied:
1. ✅ Keep schemas simple and focused  
2. ✅ Make only essential fields required
3. ✅ Use descriptive field names but keep them short
4. ✅ Prefer flat structures over deep nesting

Performance Optimizations:
- Minimal required fields for speed
- Removed unsupported constraints for API compatibility
- Focused on essential data only (document_type + confidence)

Note: Advanced constraints (min/max, maxItems, propertyOrdering) removed for compatibility
"""
