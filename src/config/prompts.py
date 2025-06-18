"""
Prompt configurations for document classification
"""

from .categories import DOCUMENT_CATEGORIES

# Generate category list for prompts
CATEGORY_LIST_COMMA = ", ".join(DOCUMENT_CATEGORIES)

# Optimized prompt (modern prompt engineering approach)
# Direct and focused - no unnecessary instructions, structured output handles JSON format
DETAILED_PROMPT = f"Classify this document image: {CATEGORY_LIST_COMMA}."

# Vertex AI optimized prompt that requests explicit format
VERTEX_AI_PROMPT = f"Classify this document image as one of: {CATEGORY_LIST_COMMA}. Respond with just the document type name."

# Prompt configuration (simplified to single optimal prompt)
PROMPTS = {
    "detailed": DETAILED_PROMPT,  # Proven fastest and most accurate (regular API)
    "vertex_ai": VERTEX_AI_PROMPT,  # Optimized for Vertex AI text responses
}

# Default prompt (using the optimal detailed prompt)
DEFAULT_PROMPT = DETAILED_PROMPT
