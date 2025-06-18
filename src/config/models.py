"""
Supported models configuration for document classification
"""

SUPPORTED_MODELS = {
    "gemini-2.0-flash": {
        "name": "gemini-2.0-flash",
        "provider": "vertex_ai",
        "version": "2.0",
        "family": "flash",
        "status": "active",
        "avg_response_time_ms": 878,  # Vertex AI performance
        "success_rate": 100,
        "avg_confidence": 0.98,
        "recommended_use": "production",
        "description": "Gemini 2.0 Flash on Vertex AI - excellent performance",
        # Optimal parameters for Vertex AI with structured output
        "parameters": {
            "temperature": 0.0,
            "top_p": 0.01,
            "top_k": 1,
            "max_output_tokens": 50,  # Back to 50 for structured JSON
        },
    },
    "gemini-2.0-flash-lite": {
        "name": "gemini-2.0-flash-lite",
        "provider": "vertex_ai",
        "version": "2.0",
        "family": "flash",
        "status": "active",
        "avg_response_time_ms": 1225,  # Vertex AI performance
        "success_rate": 100,
        "avg_confidence": 0.94,
        "recommended_use": "cost-sensitive",
        "description": "Lightweight Gemini 2.0 Flash on Vertex AI for cost optimization",
        # Optimal parameters for Vertex AI with structured output
        "parameters": {
            "temperature": 0.0,
            "top_p": 0.01,
            "top_k": 1,
            "max_output_tokens": 50,
        },
    },
}

# Model recommendations by use case
MODEL_RECOMMENDATIONS = {
    "production": "gemini-2.0-flash",  # Best overall performance
    "baseline": "gemini-2.0-flash-lite",  # Default fallback model
}

# Active models only (excluding deprecated and blocked)
ACTIVE_MODELS = {
    k: v
    for k, v in SUPPORTED_MODELS.items()
    if v["status"] in ["active", "stable", "experimental"]
}

# Production-ready models only
PRODUCTION_MODELS = {
    k: v for k, v in SUPPORTED_MODELS.items() if v["recommended_use"] == "production"
}

# Model families for easy filtering
FLASH_MODELS = {k: v for k, v in SUPPORTED_MODELS.items() if v["family"] == "flash"}

PRO_MODELS = {k: v for k, v in SUPPORTED_MODELS.items() if v["family"] == "pro"}

# Performance tiers (only working models)
SPEED_TIER = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-exp"]
BALANCED_TIER = ["gemini-1.5-flash-8b", "gemini-2.0-flash-lite"]
ACCURACY_TIER = ["gemini-1.5-pro"]

# Default model selection (using our optimized choice)
DEFAULT_MODEL = MODEL_RECOMMENDATIONS["production"]
FALLBACK_MODEL = MODEL_RECOMMENDATIONS["baseline"]

# Threading parameters for batch processing (moved from parameters.py)
THREADING_PARAMS = {
    "max_workers": 30,  # Optimized for M4 Max
    "timeout_seconds": 30,  # Per-request timeout
    "retry_attempts": 3,  # Number of retries on failure
    "retry_delay": 1.0,  # Delay between retries (seconds)
}


def get_model_info(model_name: str) -> dict | None:
    """Get detailed information about a specific model"""
    return SUPPORTED_MODELS.get(model_name)


def get_model_parameters(model_name: str) -> dict:
    """Get optimal parameters for a specific model"""
    model_info = get_model_info(model_name)
    if model_info and "parameters" in model_info:
        return model_info["parameters"]

    # Fallback to default parameters if model not found
    return {
        "temperature": 0.0,
        "top_p": 0.01,
        "top_k": 1,
        "max_output_tokens": 50,
    }


def get_recommended_model(use_case: str) -> str:
    """Get recommended model for a specific use case"""
    return MODEL_RECOMMENDATIONS.get(use_case, DEFAULT_MODEL)


def is_model_available(model_name: str) -> bool:
    """Check if a model is available and not deprecated"""
    model_info = get_model_info(model_name)
    return model_info is not None and model_info["status"] != "deprecated"
