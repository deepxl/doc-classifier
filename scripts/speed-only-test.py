#!/usr/bin/env python3.11
"""
🚀 Speed-Only Vertex AI Test
Tests API response times without needing ground truth data
"""

import os
import sys
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vertex_ai_document_classifier import VertexAIDocumentClassifier


def create_test_image():
    """Create a simple test image for speed testing"""
    from PIL import Image, ImageDraw, ImageFont
    import tempfile

    # Create a simple test document image
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # Add some text to make it look like a document
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None

    text_lines = [
        "INVOICE",
        "Date: 2024-01-15",
        "Amount: $1,234.56",
        "From: ABC Company",
        "To: XYZ Corp",
    ]

    y = 50
    for line in text_lines:
        draw.text((50, y), line, fill="black", font=font)
        y += 50

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        return tmp.name


def speed_test_model(model_name, num_tests=10):
    """Test model speed without ground truth"""

    print(f"\n🚀 Speed Testing: {model_name}")
    print(f"   Running {num_tests} speed tests...")

    try:
        # Use Vertex AI classifier with optimal settings
        classifier = VertexAIDocumentClassifier(
            model=model_name,
            prompt_type="detailed",
            parameter_set="optimal",
        )

        # Create test image
        test_image_path = create_test_image()

        # Run speed tests
        times = []
        successful_tests = 0

        for i in range(num_tests):
            try:
                start_time = time.time()

                # Use the classify_single method with our test image
                result = classifier.classify_single(test_image_path)

                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms

                if result:
                    times.append(response_time)
                    successful_tests += 1
                    print(
                        f"   Test {i+1}: {response_time:.0f}ms - {result.document_type}"
                    )
                else:
                    print(f"   Test {i+1}: FAILED (no result)")

            except Exception as e:
                print(f"   Test {i+1}: ERROR - {str(e)}")

        # Calculate statistics
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            median_time = statistics.median(times)

            return {
                "model": model_name,
                "successful_tests": successful_tests,
                "total_tests": num_tests,
                "avg_time_ms": avg_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "median_time_ms": median_time,
                "success_rate": successful_tests / num_tests,
                "all_times": times,
                "success": True,
            }
        else:
            return {
                "model": model_name,
                "error": "No successful tests",
                "success": False,
            }

    except Exception as e:
        return {"model": model_name, "error": str(e), "success": False}


def main():
    """Run speed-only tests"""

    print("⚡ VERTEX AI SPEED-ONLY TEST")
    print("Testing API response times from us-central1")
    print("=" * 50)

    # Models to test
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]

    # Run speed tests
    results = []
    for model in models:
        result = speed_test_model(model, num_tests=5)
        results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("🏁 SPEED TEST RESULTS")
    print("=" * 60)

    successful_results = [r for r in results if r["success"]]

    if successful_results:
        print(f"\n{'Model':<25} {'Avg Time':<12} {'Min Time':<12} {'Success Rate':<12}")
        print("-" * 65)

        # Sort by average time (fastest first)
        sorted_results = sorted(successful_results, key=lambda x: x["avg_time_ms"])

        for result in sorted_results:
            model_name = result["model"][:24]
            avg_time = f"{result['avg_time_ms']:.0f}ms"
            min_time = f"{result['min_time_ms']:.0f}ms"
            success_rate = f"{result['success_rate']:.1%}"

            print(f"{model_name:<25} {avg_time:<12} {min_time:<12} {success_rate:<12}")

        # Show fastest model
        fastest = sorted_results[0]
        print(f"\n🚀 FASTEST MODEL: {fastest['model']}")
        print(f"   Average: {fastest['avg_time_ms']:.0f}ms")
        print(
            f"   Range: {fastest['min_time_ms']:.0f}ms - {fastest['max_time_ms']:.0f}ms"
        )
        print(f"   Success Rate: {fastest['success_rate']:.1%}")

    else:
        print("❌ No successful speed tests")

    # Show failed models
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        print(f"\n❌ FAILED MODELS:")
        for result in failed_results:
            print(f"   {result['model']}: {result['error']}")


if __name__ == "__main__":
    main()
