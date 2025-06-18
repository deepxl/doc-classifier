#!/usr/bin/env python3.11
"""
FastAPI Example: Document Processing with Different Preprocessing Strategies
"""

import sys
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional, List
from enum import Enum

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.fastapi_document_processor import create_fastapi_processor, ProcessingConfig


class PreprocessingStrategy(str, Enum):
    DUAL = "dual"
    SINGLE_HIGH = "single_high"
    ADAPTIVE = "adaptive"
    PARALLEL = "parallel"


# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="Ultra-fast document classification and parsing with multiple preprocessing strategies",
    version="1.0.0",
)

# Global processor instance (initialize with adaptive strategy)
processor = create_fastapi_processor(
    preprocessing_strategy="adaptive", confidence_threshold=0.8
)


# Example parser integration (replace with your actual parser)
class ExampleParser:
    def parse_document(self, file_path: str, document_type: str, confidence: float):
        # This is where you'd integrate your actual parser
        return {
            "document_type": document_type,
            "classification_confidence": confidence,
            "extracted_fields": {
                "field1": "example_value1",
                "field2": "example_value2",
            },
            "parser_version": "1.0.0",
        }


# Set up parser
processor.set_parser(ExampleParser())


@app.post("/process-document/")
async def process_document(
    file: UploadFile = File(...),
    parse_document: bool = Query(True, description="Whether to run document parsing"),
    strategy: PreprocessingStrategy = Query(
        PreprocessingStrategy.ADAPTIVE, description="Preprocessing strategy to use"
    ),
):
    """
    Process a document with classification and optional parsing

    Strategies:
    - **dual**: Separate preprocessing for classification (fast) and parsing (accurate)
    - **single_high**: Single high-quality preprocessing for both steps
    - **adaptive**: Smart strategy that adapts based on classification confidence
    - **parallel**: Parallel preprocessing for maximum speed
    """

    try:
        # Validate file type
        if not file.content_type or not (
            file.content_type.startswith("image/")
            or file.content_type == "application/pdf"
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Only images and PDFs are supported.",
            )

        # Read file content
        file_content = await file.read()

        # Update processor strategy if different
        if strategy != processor.config.preprocessing_strategy:
            processor.config.preprocessing_strategy = strategy

        # Process document
        result = await processor.process_document_async(
            file_content=file_content,
            filename=file.filename or "unknown",
            parse_document=parse_document,
        )

        # Convert to JSON-serializable format
        response_data = {
            "success": result.success,
            "processing_time_ms": result.processing_time_ms,
            "strategy_used": result.strategy_used,
            "classification": {
                "document_type": result.document_type,
                "confidence": result.confidence,
                "processing_time_ms": result.classification_time_ms,
            },
            "preprocessing": {
                "time_ms": result.preprocessing_time_ms,
                "operations": result.preprocessing_operations,
            },
        }

        # Add parsing results if available
        if result.parsing_success:
            response_data["parsing"] = {
                "success": True,
                "processing_time_ms": result.parsing_time_ms,
                "data": result.parsed_data,
            }
        elif result.parsing_error:
            response_data["parsing"] = {
                "success": False,
                "error": result.parsing_error,
                "processing_time_ms": result.parsing_time_ms or 0,
            }

        # Add error if any
        if result.error_message:
            response_data["error"] = result.error_message

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/batch-process/")
async def batch_process_documents(
    files: List[UploadFile] = File(...),
    parse_documents: bool = Query(True, description="Whether to run document parsing"),
    strategy: PreprocessingStrategy = Query(
        PreprocessingStrategy.ADAPTIVE, description="Preprocessing strategy to use"
    ),
):
    """
    Process multiple documents in batch
    """

    if len(files) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 files allowed per batch"
        )

    # Update strategy
    if strategy != processor.config.preprocessing_strategy:
        processor.config.preprocessing_strategy = strategy

    results = []
    total_start_time = time.time()

    for file in files:
        try:
            file_content = await file.read()

            result = await processor.process_document_async(
                file_content=file_content,
                filename=file.filename or "unknown",
                parse_document=parse_documents,
            )

            results.append(
                {
                    "filename": file.filename,
                    "success": result.success,
                    "document_type": result.document_type,
                    "confidence": result.confidence,
                    "processing_time_ms": result.processing_time_ms,
                    "parsing_success": result.parsing_success,
                    "error": result.error_message,
                }
            )

        except Exception as e:
            results.append(
                {"filename": file.filename, "success": False, "error": str(e)}
            )

    total_time = (time.time() - total_start_time) * 1000

    return JSONResponse(
        content={
            "batch_results": results,
            "summary": {
                "total_files": len(files),
                "successful": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"]),
                "total_processing_time_ms": total_time,
                "average_time_per_document": total_time / len(files),
                "strategy_used": strategy,
            },
        }
    )


@app.get("/strategies/")
async def get_strategies():
    """
    Get information about available preprocessing strategies
    """

    return JSONResponse(
        content={
            "available_strategies": [
                {
                    "name": "dual",
                    "description": "Separate preprocessing for classification (speed) and parsing (accuracy)",
                    "use_case": "When you need both speed and accuracy optimization",
                    "preprocessing_operations": 2,
                },
                {
                    "name": "single_high",
                    "description": "Single high-quality preprocessing for both classification and parsing",
                    "use_case": "When parsing accuracy is more important than classification speed",
                    "preprocessing_operations": 1,
                },
                {
                    "name": "adaptive",
                    "description": "Smart strategy that adapts based on classification confidence",
                    "use_case": "Best overall performance - fast for easy docs, accurate for complex ones",
                    "preprocessing_operations": "1-2 (adaptive)",
                },
                {
                    "name": "parallel",
                    "description": "Parallel preprocessing for maximum throughput",
                    "use_case": "When you have multiple CPU cores and want maximum speed",
                    "preprocessing_operations": 2,
                },
            ],
            "recommended": "adaptive",
            "current_config": {
                "strategy": processor.config.preprocessing_strategy,
                "confidence_threshold": processor.config.confidence_threshold,
                "classification_settings": {
                    "max_size_kb": processor.config.classification_max_size_kb,
                    "resize_dim": processor.config.classification_resize_dim,
                    "quality": processor.config.classification_quality,
                },
                "parsing_settings": {
                    "max_size_kb": processor.config.parsing_max_size_kb,
                    "resize_dim": processor.config.parsing_resize_dim,
                    "quality": processor.config.parsing_quality,
                },
            },
        }
    )


@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import time

    print("🚀 Starting FastAPI Document Processing Server")
    print("📄 Available endpoints:")
    print("   • POST /process-document/ - Process single document")
    print("   • POST /batch-process/ - Process multiple documents")
    print("   • GET /strategies/ - View preprocessing strategies")
    print("   • GET /health/ - Health check")
    print("   • GET /docs - Interactive API documentation")
    print()
    print("🔧 Test with curl:")
    print(
        "   curl -X POST 'http://localhost:8000/process-document/?strategy=adaptive' \\"
    )
    print("        -F 'file=@your-document.pdf'")
    print()

    uvicorn.run(
        "fastapi-example:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
