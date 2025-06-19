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

    # Document information
    document_id: str
    document_name: str

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
    
    Designed to work with preprocessed content (preprocessing handled externally)
    """

    def __init__(
        self,
        classifier_model: str = "gemini-2.0-flash",
        classifier_prompt: str = "detailed",
        classifier_params: str = "optimal",
    ):
        """
        Initialize the processing pipeline

        Args:
            classifier_model: Model name for classification
            classifier_prompt: Prompt type for classification
            classifier_params: Parameter set for classification
        """
        # Initialize classifier
        self.classifier = UltraFastDocumentClassifier(
            model=classifier_model,
            prompt_type=classifier_prompt,
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
        self, 
        content: Union[str, bytes, Dict[str, Any]], 
        document_id: str = None,
        document_name: str = None,
        parse_document: bool = True
    ) -> DocumentProcessingResult:
        """
        Process a single document through the complete pipeline

        Args:
            content: Preprocessed document content (base64 image, text, or content dict)
            document_id: Unique identifier for the document
            document_name: Human-readable name for the document
            parse_document: Whether to run parsing step

        Returns:
            Complete processing result
        """
        start_time = time.time()
        pipeline_id = f"pipeline_{int(time.time() * 1000)}"
        
        # Set defaults if not provided
        if document_id is None:
            document_id = f"doc_{int(time.time() * 1000)}"
        if document_name is None:
            document_name = document_id

        # Step 1: Classification
        print(f"📋 Classifying: {document_name}")
        
        try:
            classification_result = self.classifier.classify_content(content)
        except Exception as e:
            # Create a dummy classification result for failed cases
            dummy_classification = ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="none",
                inference_id="failed",
            )
            return DocumentProcessingResult(
                document_id=document_id,
                document_name=document_name,
                classification=dummy_classification,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                pipeline_id=pipeline_id,
            )

        print(
            f"✅ Classified as: {classification_result.document_type} ({classification_result.confidence:.2f})"
        )

        # Initialize result
        result = DocumentProcessingResult(
            document_id=document_id,
            document_name=document_name,
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
                    content=content,
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
        self, 
        content: Union[str, bytes, Dict[str, Any]], 
        document_type: str, 
        confidence: float
    ) -> Dict[str, Any]:
        """
        Call your parser with classification context

        Args:
            content: Preprocessed document content
            document_type: Classified document type
            confidence: Classification confidence

        Returns:
            Parsed data dictionary
        """
        # This is where you'll integrate your parser
        # Example integration patterns:

        if self.parser and hasattr(self.parser, "parse_document"):
            # Method 1: Direct parsing with context (parser takes preprocessed content)
            return self.parser.parse_document(
                content=content, document_type=document_type, confidence=confidence
            )

        elif self.parser and hasattr(self.parser, "parse"):
            # Method 2: Simple parsing call (parser takes preprocessed content)
            return self.parser.parse(content)

        else:
            # Method 3: Custom integration
            raise ValueError("Parser integration method not implemented")

    def process_batch(
        self,
        documents: List[Dict[str, Any]],
        parse_documents: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[DocumentProcessingResult]:
        """
        Process multiple documents through the pipeline

        Args:
            documents: List of document dictionaries with keys:
                      - 'content': Preprocessed document content
                      - 'document_id': (optional) Unique identifier
                      - 'document_name': (optional) Human-readable name
            parse_documents: Whether to run parsing step
            max_workers: Number of parallel workers (future enhancement)

        Returns:
            List of processing results
        """
        print(f"🚀 Processing {len(documents)} documents...")

        results = []
        for doc in documents:
            result = self.process_single_document(
                content=doc['content'],
                document_id=doc.get('document_id'),
                document_name=doc.get('document_name'),
                parse_document=parse_documents
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

    # Example usage with preprocessed content
    example_documents = [
        {
            'content': 'base64_encoded_image_data_here',  # Replace with actual preprocessed content
            'document_id': 'doc_001',
            'document_name': 'sample_passport.jpg'
        },
        {
            'content': 'another_base64_encoded_image_data',  # Replace with actual preprocessed content
            'document_id': 'doc_002', 
            'document_name': 'sample_license.jpg'
        }
    ]

    print(f"🔍 Testing with {len(example_documents)} example documents")
    print("📝 Note: Replace example_documents with your actual preprocessed content")

    # Process without parsing (parser not integrated yet)
    # Uncomment when you have actual preprocessed content:
    # results = pipeline.process_batch(
    #     documents=example_documents,
    #     parse_documents=False,  # Set to True when parser is integrated
    # )

    # Show stats example
    print(f"\n📊 Pipeline ready for your preprocessed content!")
    print(f"💡 Usage: pipeline.process_batch(documents=your_preprocessed_docs)")


if __name__ == "__main__":
    main()
