# 🚀 Gemini Document Classifier

A high-performance document classification **component library** built on Google's Gemini models, designed for integration into your FastAPI applications.

## 🎯 Project Goals

- **Speed**: Achieve ultra-fast classification, aiming for < 400ms per document.
- **Accuracy**: Maintain >99% accuracy on a wide range of document types.
- **Efficiency**: Use optimized prompts and model parameters to minimize cost and latency.

## 🏗️ Architecture

The system provides both **standalone classification** and **integrated pipeline** capabilities:

### Standalone Classification

- **Direct Google Gemini API**: No intermediate layers, ensuring minimal latency.
- **Optimized Configuration**: A modular system for managing models, prompts, and parameters.
- **Structured Output**: Gemini's structured output feature is used to guarantee reliable JSON responses.
- **Parallel Processing**: Built for high-throughput with parallel request handling.

### Integrated Pipeline

- **Classification → Parsing Pipeline**: Seamlessly connect classification with your existing document parser.
- **Parser Integration**: Simple interface to integrate any existing parser project.
- **End-to-End Processing**: Complete document processing with comprehensive statistics and reporting.
- **Flexible Architecture**: Support for different parser interfaces and integration patterns.

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create your environment file from the example
# NOTE: .env.local is used automatically by the scripts
# a .env.local.example should be created
touch .env.local

# Add your Google API Key to .env.local
echo "GEMINI_API_KEY=your_gemini_api_key" > .env.local
```

### 2. Test the Component

```python
# Initialize the classifier
from src.core.document_classifier import UltraFastDocumentClassifier

classifier = UltraFastDocumentClassifier(
    model="gemini-2.0-flash",
    prompt_type="detailed"
)

# Option 1: Classify from file path
result = classifier.classify_single("path/to/document.jpg")

# Option 2: Classify from preprocessed content (NEW!)
preprocessed_content = {
    'content_type': 'image',
    'format': 'base64', 
    'data': your_base64_data,
    'metadata': {'original_size': '1024x768'}
}
result = classifier.classify_content(preprocessed_content)

print(f"Document type: {result.document_type}")
print(f"Confidence: {result.confidence:.2%}")
```

## 📁 Project Structure

```
gemini-classifier/
├── src/                      # 🔧 Core library code
│   ├── config/               # Configuration system
│   │   ├── models.py         # Model definitions and parameters
│   │   ├── prompts.py        # Prompt templates
│   │   ├── categories.py     # Document type definitions
│   │   └── structured_output.py # JSON output schemas
│   ├── core/                 # Main classifiers
│   │   ├── document_classifier.py # Primary classification engine
│   │   ├── vertex_ai_document_classifier.py # Vertex AI classifier
│   │   └── document_pipeline.py # Classification + parsing pipeline
│   └── models/               # Supporting models
├── setup.py                  # 📦 Package installation
├── requirements.txt          # Development dependencies
├── README.md                 # This documentation
└── .gitignore               # Version control
```

## 🔧 Configuration

The classifier's behavior is controlled by a set of configuration files in `src/config/`.

- **`models.py`**: Defines the Gemini models available for use, along with their specific recommended parameters (`temperature`, `top_p`, etc.) for different use cases (e.g., `optimal`, `standard`).
- **`prompts.py`**: Contains various prompt templates. The test script uses the `detailed` prompt, which has been found to be the most performant.
- **`categories.py`**: A centralized list of all document categories the classifier is trained to recognize.
- **`structured_output.py`**: Manages the JSON schemas sent to Gemini to ensure consistent, structured responses.

### Using the Classifier in Your Code

You can easily integrate the `DocumentClassifier` into your own applications.

```python
from src.core.document_classifier import UltraFastDocumentClassifier
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv()

# Initialize with optimal settings
classifier = UltraFastDocumentClassifier(
    model="gemini-2.0-flash",
    prompt_type="detailed"
)

# Option 1: Classify from file path
result = classifier.classify_single("path/to/document.jpg")

# Option 2: Classify preprocessed content (preprocessing handled in your main project)
result = classifier.classify_content(your_preprocessed_content)

if result:
    print(f"Type: {result.document_type} (Confidence: {result.confidence:.2%})")
    print(f"Time: {result.processing_time_ms:.0f}ms")
```

## 🔗 Parser Integration

To integrate your existing parser with the classification system, you can use the `DocumentProcessingPipeline`:

### 1. Pipeline Integration

```python
from src.core.document_pipeline import DocumentProcessingPipeline

# Initialize the pipeline
pipeline = DocumentProcessingPipeline(
    classifier_model="gemini-2.0-flash",
    classifier_prompt="detailed"
)

# Integrate your parser (replace with your actual parser)
your_parser = YourExistingParser()
pipeline.set_parser(your_parser)

# Process documents with preprocessed content
documents = [
    {
        'content': your_preprocessed_content_1,
        'document_id': 'doc_001',
        'document_name': 'passport.jpg'
    },
    {
        'content': your_preprocessed_content_2,
        'document_id': 'doc_002', 
        'document_name': 'license.jpg'
    }
]

results = pipeline.process_batch(
    documents=documents,
    parse_documents=True
)

# Get comprehensive statistics
stats = pipeline.get_pipeline_stats(results)
print(f"Success rate: {stats['parsing_success_rate']:.1f}%")
```

### 2. Parser Interface

Your parser should work with preprocessed content:

**Context-aware parsing (recommended)**

```python
class YourParser:
    def parse_document(self, content, document_type: str, confidence: float):
        # Use document_type to apply type-specific parsing logic
        # content is your preprocessed document content
        return {"field1": "value1", "field2": "value2"}
```

**Simple parsing**

```python
class YourParser:
    def parse(self, content):
        # Simple parsing without classification context
        # content is your preprocessed document content
        return {"field1": "value1", "field2": "value2"}
```

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
from src.core.document_classifier import UltraFastDocumentClassifier

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

For FastAPI integration, use the classifier in your routes:

```python
from fastapi import FastAPI, UploadFile, File
from src.core.document_classifier import UltraFastDocumentClassifier

app = FastAPI()
classifier = UltraFastDocumentClassifier()

@app.post("/classify/")
async def classify_document(file: UploadFile = File(...)):
    # Option 1: Save temporarily and classify from path
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    result = classifier.classify_single(temp_path)
    
    # Option 2: Classify directly from uploaded content
    # content = await file.read()
    # result = classifier.classify_content(content)
    
    return {
        "document_type": result.document_type,
        "confidence": result.confidence,
        "processing_time_ms": result.processing_time_ms
    }
```

## 💡 Key Optimizations & Insights

This project is the result of extensive testing and optimization. The key findings are:

- **`detailed` Prompt is Fastest**: Counter-intuitively, a more detailed prompt yielded better performance than ultra-minimal prompts.
- **Gemini 2.0 Flash is the Champion**: Consistently provides the best balance of speed, accuracy, and confidence.
- **Structured Output is Essential**: Using Gemini's native structured output is significantly faster and more reliable than parsing raw text.
- **Optimal Parameters**: A `temperature` of `0.0` and `top_p` of `0.01` (`optimal` set in `models.py`) provides the most consistent and accurate results.

## 📋 Project Status

**Version 1.1.0** - Enhanced Component Library ✅

This project is a **comprehensive, production-ready component library** for document classification in FastAPI applications.

**Key Features:**

- 📦 **Ultra-Clean Structure** - Minimal files, no preprocessing or development artifacts
- 🚀 **Dual Input Support** - Works with both file paths AND preprocessed content
- ⚡ **Optimized Performance** - Ultra-fast classification (<400ms per document)
- 🔧 **Easy Integration** - Simple import and usage in existing projects
- 🎯 **Focused Categories** - Supports 11 core document types including identity, financial, and business documents
- 🔄 **Flexible Content Formats** - Accepts base64, bytes, PIL Images, and structured dictionaries

**Installation:**

```bash
# Core library
pip install -e .

# With FastAPI extras
pip install -e ".[fastapi]"
```

**Supported Document Types:** 11 core categories including passport, driver_license, bank_statement, utility_bill, employment_card, and more.

---

_A clean, minimal component for production document classification._
