#!/usr/bin/env python3.11
"""
Example: How to integrate your existing parser with the classification pipeline
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.document_pipeline import DocumentProcessingPipeline


# Example 1: Simple parser integration
class YourExistingParser:
    """
    This represents your existing parser from the other project
    Replace this with your actual parser class
    """

    def parse_document(self, file_path: str, document_type: str, confidence: float):
        """
        Your parser method that receives classification context

        Args:
            file_path: Path to the document
            document_type: Classified document type (passport, driver_license, etc.)
            confidence: Classification confidence (0.0 to 1.0)

        Returns:
            Dictionary of parsed fields
        """
        # This is where your actual parsing logic would go
        # You can use the document_type to apply type-specific parsing

        print(f"    🔧 Parsing {document_type} with {confidence:.2f} confidence")

        # Example return structure - replace with your actual parsing
        if document_type == "passport":
            return {
                "document_number": "A12345678",
                "full_name": "John Doe",
                "date_of_birth": "1990-01-01",
                "nationality": "US",
                "expiry_date": "2030-01-01",
                "issuing_country": "USA",
            }
        elif document_type == "driver_license":
            return {
                "license_number": "DL123456789",
                "full_name": "Jane Smith",
                "date_of_birth": "1985-05-15",
                "address": "123 Main St, City, State",
                "expiry_date": "2028-05-15",
                "license_class": "C",
            }
        else:
            return {
                "document_type": document_type,
                "status": "parsed",
                "confidence": confidence,
            }


def main():
    """Example integration workflow"""
    print("🔗 Parser Integration Example")
    print("=" * 50)

    # Step 1: Initialize the pipeline
    pipeline = DocumentProcessingPipeline(
        classifier_model="detailed",  # Use your preferred model config
        classifier_prompt="detailed",  # Use your preferred prompt
        classifier_params="optimal",  # Use optimal parameters
    )

    # Step 2: Create your parser instance
    your_parser = YourExistingParser()

    # Step 3: Integrate parser with pipeline
    pipeline.set_parser(your_parser)

    # Step 4: Test with sample documents
    import glob

    test_files = (
        glob.glob("test-images/*.jpg")
        + glob.glob("test-images/*.jpeg")
        + glob.glob("test-images/*.png")
        + glob.glob("test-images/*.pdf")
    )

    if not test_files:
        print("❌ No test files found in test-images/")
        print("   Add some document images to test the integration")
        return

    # Limit to first 2 files for demo
    test_files = test_files[:2]
    print(f"🔍 Testing integration with {len(test_files)} files")

    # Step 5: Process documents through the complete pipeline
    results = pipeline.process_batch(
        file_paths=test_files, parse_documents=True, max_workers=None  # Enable parsing
    )

    # Step 6: Review results
    print(f"\n📊 Integration Results:")
    print(f"=" * 30)

    for result in results:
        print(f"\n📄 File: {result.file_name}")
        print(
            f"   📋 Classification: {result.classification.document_type} ({result.classification.confidence:.2f})"
        )
        print(
            f"   ⏱️  Classification time: {result.classification.processing_time_ms:.0f}ms"
        )

        if result.parsing_success:
            print(f"   ✅ Parsing: SUCCESS ({result.parsing_time_ms:.0f}ms)")
            print(f"   🔧 Parsed fields: {len(result.parsed_data)} fields")
            # Show first few fields
            for key, value in list(result.parsed_data.items())[:3]:
                print(f"      • {key}: {value}")
        else:
            print(f"   ❌ Parsing: FAILED - {result.parsing_error}")

        print(f"   🕒 Total time: {result.total_processing_time_ms:.0f}ms")

    # Step 7: Get comprehensive stats
    stats = pipeline.get_pipeline_stats(results)
    print(f"\n📈 Pipeline Performance:")
    print(f"   • Total documents: {stats['total_documents']}")
    print(f"   • Classification success: {stats['classification_success_rate']:.1f}%")
    print(f"   • Parsing success: {stats['parsing_success_rate']:.1f}%")
    print(f"   • Avg total time: {stats['avg_total_time_ms']:.0f}ms")
    print(f"   • Processing speed: {stats['documents_per_second']:.1f} docs/sec")

    # Step 8: Export results
    pipeline.export_results(results, "results/integration-test-results.json")

    print(f"\n✅ Integration test completed successfully!")
    print(f"🔧 Next steps:")
    print(f"   1. Replace YourExistingParser with your actual parser class")
    print(f"   2. Update the parse_document method to match your parser's interface")
    print(f"   3. Test with your actual documents")
    print(f"   4. Adjust parameters as needed")


if __name__ == "__main__":
    main()
