#!/usr/bin/env python3.11
"""
Document Processing Pipeline
Integrates classification with parsing for end-to-end document processing
"""

import os
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv

from .document_classifier import UltraFastDocumentClassifier, ClassificationResult

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env.local")


@dataclass
class DocumentProcessingResult:
    """Complete document processing result"""

    # File information
    file_path: str
    file_name: str

    # Classification results
    classification: ClassificationResult

    # Parsing results (to be populated by your parser)
    parsed_data: Optional[Dict[str, Any]] = None
    parsing_time_ms: Optional[float] = None
    parsing_success: bool = False
    parsing_error: Optional[str] = None

    # Pipeline metadata
    total_processing_time_ms: float = 0.0
    pipeline_id: str = ""


class DocumentProcessingPipeline:
    """
    End-to-end document processing pipeline
    Classification → Parsing → Results
    """

    def __init__(
        self,
        classifier_model: str = "detailed",
        classifier_prompt: str = "detailed",
        classifier_params: str = "optimal",
    ):
        """
        Initialize the processing pipeline

        Args:
            classifier_model: Model configuration for classification
            classifier_prompt: Prompt type for classification
            classifier_params: Parameter set for classification
        """
        # Initialize classifier
        self.classifier = UltraFastDocumentClassifier(
            model=classifier_model,
            prompt_type=classifier_prompt,
            parameter_set=classifier_params,
        )

        # Your parser integration point
        self.parser = None  # Will be set via set_parser()

    def set_parser(self, parser_instance):
        """
        Set the parser instance from your other project

        Args:
            parser_instance: Your parser class instance
        """
        self.parser = parser_instance
        print("✅ Parser integrated successfully")

    def process_single_document(
        self, file_path: str, parse_document: bool = True
    ) -> DocumentProcessingResult:
        """
        Process a single document through the complete pipeline

        Args:
            file_path: Path to document file
            parse_document: Whether to run parsing step

        Returns:
            Complete processing result
        """
        start_time = time.time()
        pipeline_id = f"pipeline_{int(time.time() * 1000)}"
        file_path_obj = Path(file_path)

        # Step 1: Classification
        print(f"📋 Classifying: {file_path_obj.name}")
        classification_result = self.classifier.classify_single(file_path)

        if not classification_result:
            # Create a dummy classification result for failed cases
            dummy_classification = ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="none",
                inference_id="failed",
            )
            return DocumentProcessingResult(
                file_path=file_path,
                file_name=file_path_obj.name,
                classification=dummy_classification,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                pipeline_id=pipeline_id,
            )

        print(
            f"✅ Classified as: {classification_result.document_type} ({classification_result.confidence:.2f})"
        )

        # Initialize result
        result = DocumentProcessingResult(
            file_path=file_path,
            file_name=file_path_obj.name,
            classification=classification_result,
            pipeline_id=pipeline_id,
        )

        # Step 2: Parsing (if parser is available and requested)
        if parse_document and self.parser:
            print(f"🔧 Parsing document...")
            parsing_start = time.time()

            try:
                # Call your parser with classification context
                parsed_data = self._call_parser(
                    file_path=str(file_path),
                    document_type=classification_result.document_type,
                    confidence=classification_result.confidence,
                )

                result.parsed_data = parsed_data
                result.parsing_success = True
                result.parsing_time_ms = (time.time() - parsing_start) * 1000

                print(f"✅ Parsing completed in {result.parsing_time_ms:.0f}ms")

            except Exception as e:
                result.parsing_error = str(e)
                result.parsing_success = False
                result.parsing_time_ms = (time.time() - parsing_start) * 1000
                print(f"❌ Parsing failed: {e}")

        elif parse_document and not self.parser:
            print("⚠️  Parser not set - skipping parsing step")

        # Finalize timing
        result.total_processing_time_ms = (time.time() - start_time) * 1000

        return result

    def _call_parser(
        self, file_path: str, document_type: str, confidence: float
    ) -> Dict[str, Any]:
        """
        Call your parser with classification context

        Args:
            file_path: Path to document
            document_type: Classified document type
            confidence: Classification confidence

        Returns:
            Parsed data dictionary
        """
        # This is where you'll integrate your parser
        # Example integration patterns:

        if self.parser and hasattr(self.parser, "parse_document"):
            # Method 1: Direct parsing with context
            return self.parser.parse_document(
                file_path=file_path, document_type=document_type, confidence=confidence
            )

        elif self.parser and hasattr(self.parser, "parse"):
            # Method 2: Simple parsing call
            return self.parser.parse(file_path)

        else:
            # Method 3: Custom integration
            raise ValueError("Parser integration method not implemented")

    def process_batch(
        self,
        file_paths: List[str],
        parse_documents: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[DocumentProcessingResult]:
        """
        Process multiple documents through the pipeline

        Args:
            file_paths: List of document file paths
            parse_documents: Whether to run parsing step
            max_workers: Number of parallel workers

        Returns:
            List of processing results
        """
        print(f"🚀 Processing {len(file_paths)} documents...")

        results = []
        for file_path in file_paths:
            result = self.process_single_document(
                file_path=file_path, parse_document=parse_documents
            )
            results.append(result)

        return results

    def get_pipeline_stats(self, results: List[DocumentProcessingResult]) -> Dict:
        """Get comprehensive pipeline statistics"""

        if not results:
            return {}

        successful_classifications = [r for r in results if r.classification]
        successful_parsings = [r for r in results if r.parsing_success]

        classification_times = [
            r.classification.processing_time_ms for r in successful_classifications
        ]
        parsing_times = [
            r.parsing_time_ms for r in successful_parsings if r.parsing_time_ms
        ]
        total_times = [r.total_processing_time_ms for r in results]

        stats = {
            # Overall stats
            "total_documents": len(results),
            "total_processing_time_s": sum(total_times) / 1000,
            "avg_total_time_ms": sum(total_times) / len(total_times),
            "documents_per_second": len(results) / (sum(total_times) / 1000),
            # Classification stats
            "classification_success_rate": len(successful_classifications)
            / len(results)
            * 100,
            "avg_classification_time_ms": (
                sum(classification_times) / len(classification_times)
                if classification_times
                else 0
            ),
            "avg_classification_confidence": (
                sum(r.classification.confidence for r in successful_classifications)
                / len(successful_classifications)
                if successful_classifications
                else 0
            ),
            # Parsing stats
            "parsing_success_rate": (
                len(successful_parsings) / len(results) * 100
                if any(r.parsing_time_ms for r in results)
                else 0
            ),
            "avg_parsing_time_ms": (
                sum(parsing_times) / len(parsing_times) if parsing_times else 0
            ),
            # Document types found
            "document_types_found": list(
                set(r.classification.document_type for r in successful_classifications)
            ),
        }

        return stats

    def export_results(
        self,
        results: List[DocumentProcessingResult],
        output_path: str,
        format: str = "json",
    ):
        """
        Export processing results to file

        Args:
            results: Processing results to export
            output_path: Output file path
            format: Export format ("json" or "csv")
        """
        output_path_obj = Path(output_path)

        if format.lower() == "json":
            # Convert dataclasses to dictionaries
            export_data = []
            for result in results:
                result_dict = asdict(result)
                # Convert ClassificationResult to dict if present
                if result_dict["classification"]:
                    result_dict["classification"] = asdict(result.classification)
                export_data.append(result_dict)

            with open(output_path_obj, "w") as f:
                json.dump(export_data, f, indent=2)

        else:
            raise ValueError(f"Export format '{format}' not supported")

        print(f"📁 Results exported to: {output_path_obj}")


def main():
    """Test the integrated pipeline"""
    print("🔗 Document Processing Pipeline Test")
    print("=" * 50)

    # Initialize pipeline
    pipeline = DocumentProcessingPipeline()

    # Find test documents
    import glob

    test_files = (
        glob.glob("test-images/*.jpg")
        + glob.glob("test-images/*.jpeg")
        + glob.glob("test-images/*.png")
        + glob.glob("test-images/*.pdf")
    )

    if not test_files:
        print("❌ No test files found")
        return

    test_files = test_files[:3]  # Test with first 3 files
    print(f"🔍 Testing with {len(test_files)} files")

    # Process without parsing (parser not integrated yet)
    results = pipeline.process_batch(
        file_paths=test_files,
        parse_documents=False,  # Set to True when parser is integrated
    )

    # Show results
    stats = pipeline.get_pipeline_stats(results)
    print(f"\n📊 Pipeline Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Export results
    pipeline.export_results(results, "results/pipeline-test-results.json")


if __name__ == "__main__":
    main()
