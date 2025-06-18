#!/usr/bin/env python3.11
"""
🌍 LOCAL vs GCP REGION SPEED COMPARISON

Quick test to compare:
1. Current speeds from your MacBook to Vertex AI
2. Expected improvement when running from us-central1

This gives you a preview before running the full GCP test.
"""

import json
import statistics
import sys
import time
from pathlib import Path

# Get project root and add to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vertex_ai_document_classifier import VertexAIDocumentClassifier


def load_ground_truth():
    """Load ground truth labels from JSON file"""
    ground_truth_file = project_root / "test-images" / "ground-truth.json"
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)
    return ground_truth


def quick_speed_test(model_name, test_images, num_samples=5):
    """Quick speed test with small sample"""

    print(f"\n🤖 Quick test: {model_name}")
    print(f"   📱 Testing from your MacBook to Vertex AI")
    print(f"   🔢 Using {num_samples} sample images for speed")

    try:
        classifier = VertexAIDocumentClassifier(
            model=model_name,
            prompt_type="detailed",
            parameter_set="optimal",
        )

        # Use subset of images for quick test
        sample_images = test_images[:num_samples]

        processing_times = []
        successful = 0

        for i, image_file in enumerate(sample_images, 1):
            image_path = project_root / "test-images" / image_file
            if not image_path.exists():
                continue

            try:
                print(f"   [{i}/{len(sample_images)}] Testing {image_file}...", end="")
                result = classifier.classify_single(str(image_path))

                if result is not None:
                    processing_times.append(result.processing_time_ms)
                    successful += 1
                    print(f" {result.processing_time_ms:.0f}ms ✅")
                else:
                    print(" Failed ❌")

            except Exception as e:
                print(f" Error: {str(e)[:50]}... ❌")

        if processing_times:
            avg_time = statistics.mean(processing_times)
            return {
                "model": model_name,
                "avg_time_ms": avg_time,
                "successful": successful,
                "total": len(sample_images),
                "times": processing_times,
            }
        else:
            return None

    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        return None


def main():
    """Run quick local vs GCP comparison"""

    print("🌍 LOCAL vs GCP SPEED PREVIEW")
    print("Quick test to estimate regional speed improvement")
    print("=" * 55)

    # Load test images
    ground_truth = load_ground_truth()
    test_images = list(ground_truth.keys())

    print(f"\n📋 Available test images: {len(test_images)}")
    print("🚀 Running quick 5-image speed test...")

    # Test both Vertex AI models quickly
    models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]
    results = []

    for model in models:
        result = quick_speed_test(model, test_images, num_samples=5)
        if result:
            results.append(result)

    if not results:
        print("❌ No successful tests completed")
        return

    print("\n" + "=" * 60)
    print("📊 SPEED RESULTS FROM YOUR MACBOOK")
    print("=" * 60)

    for result in results:
        print(f"\n🤖 {result['model']}:")
        print(f"   ⚡ Average: {result['avg_time_ms']:.0f}ms")
        print(f"   ✅ Success: {result['successful']}/{result['total']}")
        print(f"   📈 Range: {min(result['times']):.0f}-{max(result['times']):.0f}ms")

    # Calculate expected improvement
    print("\n" + "=" * 60)
    print("🌍 EXPECTED IMPROVEMENT IN us-central1")
    print("=" * 60)

    avg_local_time = statistics.mean([r["avg_time_ms"] for r in results])

    # Estimated latency reductions (conservative estimates)
    # MacBook to GCP: ~50-150ms round trip
    # us-central1 to Vertex AI: ~5-15ms round trip
    latency_improvement = 100  # Conservative 100ms improvement

    estimated_gcp_time = max(avg_local_time - latency_improvement, avg_local_time * 0.7)
    improvement_percent = ((avg_local_time - estimated_gcp_time) / avg_local_time) * 100

    print(f"\n📱 Current (MacBook): {avg_local_time:.0f}ms average")
    print(f"🌍 Expected (us-central1): {estimated_gcp_time:.0f}ms average")
    print(f"🚀 Estimated improvement: {improvement_percent:.1f}% faster")
    print(f"💡 Network latency reduction: ~{latency_improvement}ms")

    # Throughput comparison
    current_throughput = 3600000 / avg_local_time  # docs per hour
    expected_throughput = 3600000 / estimated_gcp_time

    print(f"\n📈 THROUGHPUT COMPARISON:")
    print(f"   📱 Current: {current_throughput:.0f} docs/hour")
    print(f"   🌍 Expected: {expected_throughput:.0f} docs/hour")
    print(f"   🚀 Increase: +{expected_throughput - current_throughput:.0f} docs/hour")

    print(f"\n🎯 RECOMMENDATION:")
    if improvement_percent > 20:
        print(f"   ✅ Significant improvement expected!")
        print(f"   🔧 Run the full GCP test: ./scripts/gcp-speed-test.sh")
    elif improvement_percent > 10:
        print(f"   ✅ Moderate improvement expected")
        print(f"   🔧 Consider GCP test for production validation")
    else:
        print(f"   ⚠️  Minimal improvement expected")
        print(f"   💭 Your current setup may already be well-optimized")

    # Save quick results
    results_file = project_root / "results" / "local-speed-preview.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "local_speed_preview",
                "location": "MacBook to Vertex AI",
                "avg_time_ms": avg_local_time,
                "estimated_gcp_time_ms": estimated_gcp_time,
                "estimated_improvement_percent": improvement_percent,
                "models_tested": [r["model"] for r in results],
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
