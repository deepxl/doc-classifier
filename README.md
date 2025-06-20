# 🚀 Gemini Document Classifier

A high-performance document classification **component library** built on Google's Gemini models, designed for integration into your FastAPI applications.

## 🎯 Project Goals

- **Speed**: Achieve ultra-fast classification, aiming for < 400ms per document.
- **Accuracy**: Maintain >99% accuracy on a wide range of document types.
- **Efficiency**: Use optimized prompts and model parameters to minimize cost and latency.

## 🏗️ Architecture

The system provides **standalone document classification** capabilities:

### Core Features

- **Direct Google Gemini API**: No intermediate layers, ensuring minimal latency.
- **Vertex AI Support**: Production-ready Vertex AI integration for enterprise workloads.
- **Optimized Configuration**: A modular system for managing models, prompts, and parameters.
- **Structured Output**: Gemini's structured output feature guarantees reliable JSON responses.
- **Parallel Processing**: Built for high-throughput with parallel request handling.
- **Flexible Input**: Supports both file paths and preprocessed content (base64, bytes, PIL Images).

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create your environment file
touch .env.local

# Add your Google API Key to .env.local
echo "GEMINI_API_KEY=your_gemini_api_key" > .env.local

# Optional: Add Vertex AI credentials for production
echo "GCP_PROJECT_ID=your_project_id" >> .env.local
echo "GCP_REGION=us-central1" >> .env.local
```

### 2. Basic Usage

```python
# Initialize the classifier
from src.core.document_classifier import UltraFastDocumentClassifier

classifier = UltraFastDocumentClassifier(
    model="gemini-2.0-flash",
    prompt_type="detailed"
)

# Option 1: Classify from file path
result = classifier.classify_single("path/to/document.jpg")

# Option 2: Classify from preprocessed content
preprocessed_content = {
    'content_type': 'image',
    'format': 'base64', 
    'data': your_base64_data,
    'metadata': {'original_size': '1024x768'}
}
result = classifier.classify_content(preprocessed_content)

print(f"Document type: {result.document_type}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Processing time: {result.processing_time_ms:.0f}ms")
```

### 3. Vertex AI (Production)

```python
# For production workloads with Vertex AI
from src.core.vertex_ai_document_classifier import VertexAIDocumentClassifier

classifier = VertexAIDocumentClassifier(
    model="gemini-2.0-flash",
    prompt_type="detailed"
)

result = classifier.classify_single("document.pdf")
```

## 📁 Project Structure

```
gemini-classifier/
├── src/                      # 🔧 Core library code
│   ├── config/               # Configuration system
│   │   ├── models.py         # Model definitions and parameters
│   │   ├── prompts.py        # Prompt templates
│   │   ├── categories.py     # Document type definitions
│   │   ├── structured_output.py # JSON output schemas
│   │   └── settings.py       # Runtime settings
│   ├── core/                 # Main classifiers
│   │   ├── document_classifier.py # Primary classification engine
│   │   ├── vertex_ai_document_classifier.py # Vertex AI classifier
│   │   ├── types.py          # Shared data types
│   │   └── exceptions.py     # Custom exceptions
│   └── __init__.py          # Package exports
├── tests/                   # Test suite
│   ├── test-with-ground-truth.py # Ground truth validation
│   └── ground-truth.json    # Test data definitions
├── setup.py                 # 📦 Package installation
├── requirements.txt         # Dependencies
├── test-component.py        # Component test script
└── README.md               # This documentation
```

## 🔧 Configuration

The classifier's behavior is controlled by configuration files in `src/config/`:

- **`models.py`**: Defines available Gemini models with optimized parameters
- **`prompts.py`**: Contains prompt templates (detailed prompt recommended)
- **`categories.py`**: Centralized list of supported document categories
- **`structured_output.py`**: JSON schemas for consistent responses
- **`settings.py`**: Runtime configuration and environment settings

## 🔧 Installation & Integration

This is a **component library** designed to be integrated into your existing FastAPI applications.

### 1. Install as Package

```bash
# Install from source
git clone https://github.com/deepxl/doc-classifier.git
cd doc-classifier
pip install -e .

# Or install with FastAPI extras
pip install -e ".[fastapi]"
```

### 2. Basic Integration

```python
from src.core import UltraFastDocumentClassifier, VertexAIDocumentClassifier

# Initialize the classifier
classifier = UltraFastDocumentClassifier(
    model="gemini-2.0-flash",
    prompt_type="detailed"
)

# Classify from file path or preprocessed content
result = classifier.classify_single("path/to/document.jpg")  # File path
# OR
result = classifier.classify_content(preprocessed_content)   # Preprocessed content

print(f"Type: {result.document_type}, Confidence: {result.confidence}")
```

### 3. FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, File
from src.core import UltraFastDocumentClassifier

app = FastAPI()
classifier = UltraFastDocumentClassifier()

@app.post("/classify/")
async def classify_document(file: UploadFile = File(...)):
    # Save temporarily and classify
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    result = classifier.classify_single(temp_path)
    
    return {
        "document_type": result.document_type,
        "confidence": result.confidence,
        "processing_time_ms": result.processing_time_ms
    }
```

## 📊 Supported Content Formats

The classifier accepts multiple input formats:

```python
# File paths
result = classifier.classify_single("document.jpg")

# Base64 strings
result = classifier.classify_content("iVBORw0KGgoAAAANSUhEUgAA...")

# Bytes data
with open("document.jpg", "rb") as f:
    result = classifier.classify_content(f.read())

# Structured dictionaries
content = {
    'content_type': 'image',
    'format': 'base64',
    'data': base64_encoded_data,
    'metadata': {'source': 'scanner'}
}
result = classifier.classify_content(content)

# PIL Images
from PIL import Image
img = Image.open("document.jpg")
result = classifier.classify_content(img)
```

## 💡 Key Optimizations & Insights

This project incorporates extensive testing and optimization:

- **`detailed` Prompt is Fastest**: Counter-intuitively, detailed prompts outperform minimal ones
- **Gemini 2.0 Flash is Optimal**: Best balance of speed, accuracy, and confidence
- **Structured Output is Essential**: Native structured output is faster than text parsing
- **Optimal Parameters**: `temperature=0.0`, `top_p=0.01` for consistent results
- **Vertex AI for Production**: Better availability and performance for enterprise workloads

## 📋 Supported Document Types

**11 Core Categories:**
- **Identity**: passport, id_card, driver_license, passport_card
- **Financial**: bank_statement, utility_bill, paystub, tax_document  
- **Business**: employment_card, green_card
- **Fallback**: other

## 🧪 Testing

```bash
# Run component tests
python test-component.py

# Run ground truth validation (requires test images)
python tests/test-with-ground-truth.py
```

## 📋 Project Status

**Version 1.2.0** - Simplified Classification Component ✅

**Key Features:**
- 🎯 **Classification-Only Focus** - Streamlined for single-purpose use
- ⚡ **Ultra-Fast Performance** - <400ms per document
- 🔧 **Dual API Support** - Regular Gemini API + Vertex AI
- 📦 **Clean Architecture** - Minimal dependencies, focused functionality
- 🚀 **Production Ready** - Comprehensive error handling and monitoring

**Installation:**
```bash
pip install -e .              # Core functionality
pip install -e ".[fastapi]"   # With FastAPI extras
```

---

_A lean, focused component for production document classification._
