# 🚀 Gemini Document Classifier

An enterprise-grade document classification system built on Google's Gemini models, optimized for speed and accuracy.

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

### 2. Run Model Comparison Test

The primary script in this project, `scripts/test-models.py`, runs a comprehensive performance comparison of key Gemini models using a pre-defined optimal configuration.

```bash
# Run the test from the root directory
python3.11 scripts/test-models.py
```

The script will:

- Load the ground truth data from `test-images/ground-truth.json`.
- Test a suite of Gemini 2.0 Flash models.
- Calculate and display detailed metrics: accuracy, F1-score, speed, and confidence.
- Provide a "champion" model recommendation based on the results.
- Save a detailed report to `results/optimized-model-comparison.json`.

## 📁 Project Structure

```
gemini-classifier/
├── src/
│   ├── config/               # Modular configuration system
│   │   ├── models.py         # Model definitions and parameters
│   │   ├── prompts.py        # Prompt templates
│   │   ├── categories.py     # Document type definitions
│   │   └── structured_output.py # JSON output schemas
│   └── core/
│       ├── document_classifier.py # Main classification engine
│       └── document_pipeline.py   # Integrated pipeline (classification + parsing)
├── scripts/
│   ├── test-models.py        # Comprehensive model performance test script
│   └── integrate-parser-example.py # Parser integration example
├── test-images/
│   ├── ground-truth.json     # Ground truth labels for test images
│   └── ...                   # Test images (.jpg, .pdf)
├── results/
│   └── archive/              # Archive of past test results
├── docs/                     # Project documentation
├── requirements.txt          # Python dependencies
├── .env.local                # Local environment variables (API keys)
└── README.md                 # This file
```

## 🔧 Configuration

The classifier's behavior is controlled by a set of configuration files in `src/config/`.

- **`models.py`**: Defines the Gemini models available for use, along with their specific recommended parameters (`temperature`, `top_p`, etc.) for different use cases (e.g., `optimal`, `standard`).
- **`prompts.py`**: Contains various prompt templates. The test script uses the `detailed` prompt, which has been found to be the most performant.
- **`categories.py`**: A centralized list of all document categories the classifier is trained to recognize.
- **`structured_output.py`**: Manages the JSON schemas sent to Gemini to ensure consistent, structured responses.

### Using the Classifier in Your Code

You can easily integrate the `UltraFastDocumentClassifier` into your own applications.

```python
from src.core.document_classifier import UltraFastDocumentClassifier
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv()

# Initialize with default (optimal) settings
classifier = UltraFastDocumentClassifier()

# Classify a single document
# Ensure the image path is correct
image_path = "test-images/pass1.jpg"
result = classifier.classify_single(image_path)

if result:
    print(f"Document: {image_path}")
    print(f"Type: {result.document_type} (Confidence: {result.confidence:.2%})")
    print(f"Time: {result.processing_time_ms:.0f}ms")

```

## 🔗 Parser Integration

To integrate your existing parser with the classification system, you can use the `DocumentProcessingPipeline`:

### 1. Basic Integration

```python
from src.core.document_pipeline import DocumentProcessingPipeline

# Initialize the pipeline
pipeline = DocumentProcessingPipeline()

# Integrate your parser (replace with your actual parser)
your_parser = YourExistingParser()
pipeline.set_parser(your_parser)

# Process documents through the complete pipeline
results = pipeline.process_batch(
    file_paths=["document1.pdf", "document2.jpg"],
    parse_documents=True
)

# Get comprehensive statistics
stats = pipeline.get_pipeline_stats(results)
print(f"Success rate: {stats['parsing_success_rate']:.1f}%")
```

### 2. Parser Interface Requirements

Your parser should implement one of these interfaces:

**Option A: Context-aware parsing**

```python
class YourParser:
    def parse_document(self, file_path: str, document_type: str, confidence: float):
        # Use document_type to apply type-specific parsing logic
        return {"field1": "value1", "field2": "value2"}
```

**Option B: Simple parsing**

```python
class YourParser:
    def parse(self, file_path: str):
        # Simple parsing without classification context
        return {"field1": "value1", "field2": "value2"}
```

### 3. Test Integration

```bash
# Run the integration example
python3.11 scripts/integrate-parser-example.py
```

This will show you exactly how to connect your parser and test the complete pipeline.

## 🌐 FastAPI Deployment

For production web API deployment, use the optimized FastAPI processor with multiple preprocessing strategies:

### 1. Quick Start

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn python-multipart

# Run the example server
python3.11 scripts/fastapi-example.py
```

### 2. FastAPI Integration

```python
from src.core.fastapi_document_processor import create_fastapi_processor

# Create processor with adaptive strategy (recommended)
processor = create_fastapi_processor(
    preprocessing_strategy="adaptive",
    confidence_threshold=0.8
)

# Add your parser
processor.set_parser(your_parser_instance)

# Process documents asynchronously
result = await processor.process_document_async(
    file_content=file_bytes,
    filename="document.pdf",
    parse_document=True
)
```

### 3. Preprocessing Strategies

Choose the optimal strategy for your use case:

- **Adaptive** (Recommended): Smart preprocessing based on classification confidence - 20-30% faster than dual
- **Parallel**: Concurrent preprocessing for maximum throughput on multi-core systems
- **Dual**: Your current approach - separate preprocessing for classification and parsing
- **Single High**: Single high-quality preprocessing when parsing accuracy is critical

See [docs/preprocessing-strategies.md](docs/preprocessing-strategies.md) for detailed comparison and performance benchmarks.

### 4. API Endpoints

The FastAPI server provides:

- `POST /process-document/` - Process single document with strategy selection
- `POST /batch-process/` - Process multiple documents in batch
- `GET /strategies/` - View available preprocessing strategies and current config
- `GET /docs` - Interactive API documentation

## 💡 Key Optimizations & Insights

This project is the result of extensive testing and optimization. The key findings are:

- **`detailed` Prompt is Fastest**: Counter-intuitively, a more detailed prompt yielded better performance than ultra-minimal prompts.
- **Gemini 2.0 Flash is the Champion**: Consistently provides the best balance of speed, accuracy, and confidence.
- **Structured Output is Essential**: Using Gemini's native structured output is significantly faster and more reliable than parsing raw text.
- **Optimal Parameters**: A `temperature` of `0.0` and `top_p` of `0.01` (`optimal` set in `models.py`) provides the most consistent and accurate results.

---

_This project is focused on providing a clean, powerful, and highly-optimized baseline for document classification tasks._
