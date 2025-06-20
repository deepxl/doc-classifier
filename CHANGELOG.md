# Changelog

## [1.2.0] - 2024-12-19 - Classification-Only Cleanup

### 🧹 Major Cleanup & Simplification
- **REMOVED**: Legacy pipeline code (`src/core/document_pipeline.py`)
- **REMOVED**: Legacy classifier/parser stubs (`src/core/classifier.py`, `src/core/parser.py`)
- **REMOVED**: Unused fallback strategy infrastructure (`src/models/` package)
- **REMOVED**: `DocumentProcessingResult` dataclass (pipeline-only)
- **SIMPLIFIED**: Test suite focused on classification only
- **UNIFIED**: Single source of truth for `ClassificationResult` in `src/core/types.py`

### ✨ What Remains (Core Features)
- ✅ `UltraFastDocumentClassifier` - Primary classification engine
- ✅ `VertexAIDocumentClassifier` - Production Vertex AI classifier  
- ✅ Shared `ClassificationResult` type
- ✅ Complete configuration system (`src/config/`)
- ✅ Comprehensive test suite
- ✅ Ground truth validation

### 🎯 Benefits
- **Focused Purpose**: Pure classification component, no parsing complexity
- **Reduced Dependencies**: Removed unused model abstraction layers
- **Cleaner Imports**: Simplified module structure
- **Easier Maintenance**: Less code to maintain, clearer responsibilities
- **Better Performance**: No overhead from unused pipeline infrastructure

### 📦 Installation
```bash
pip install -e .              # Core functionality
pip install -e ".[fastapi]"   # With FastAPI extras
```

### 🚀 Usage
```python
from src.core import UltraFastDocumentClassifier

classifier = UltraFastDocumentClassifier()
result = classifier.classify_single("document.jpg")
print(f"{result.document_type} ({result.confidence:.2%})")
```

---

## [1.1.0] - 2024-12-18 - Enhanced Component Library

### Added
- Enhanced pipeline integration
- Preprocessed content support
- Comprehensive testing framework

### Changed
- Improved documentation
- Better error handling
- Performance optimizations

---

## [1.0.0] - 2024-12-17 - Initial Release

### Added
- Basic document classification
- Gemini API integration
- Ground truth validation
- Core configuration system 