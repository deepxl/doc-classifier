#!/usr/bin/env python3.11
"""
🌍 SIMPLE VERTEX AI TEST

Quick test to show current Vertex AI speeds and estimate us-central1 improvement.
No VM setup needed - runs locally but gives you the key insights.
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


def quick_vertex_test():
    """Quick Vertex AI test with key insights"""

    print("🌍 SIMPLE VERTEX AI SPEED TEST")
    print("Testing from your MacBook + showing us-central1 estimates")
    print("=" * 60)

    # Load a few test images
    test_images = ["pass1.jpg", "dl1.jpg", "id1.jpg", "pc1.jpg", "ub1.pdf"]

    print(f"\n📋 Quick test with {len(test_images)} sample images")

    models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]
    results = {}

    for model in models:
        print(f"\n🤖 Testing {model}...")

        try:
            classifier = VertexAIDocumentClassifier(
                model=model,
                prompt_type="detailed",
                parameter_set="optimal",
            )

            times = []
            successful = 0

            for i, img in enumerate(test_images, 1):
                img_path = project_root / "test-images" / img
                if not img_path.exists():
                    continue

                try:
                    print(f"   [{i}/{len(test_images)}] {img}...", end="")
                    result = classifier.classify_single(str(img_path))

                    if result:
                        times.append(result.processing_time_ms)
                        successful += 1
                        print(f" {result.processing_time_ms:.0f}ms ✅")
                    else:
                        print(" Failed ❌")

                except Exception as e:
                    print(f" Error ❌")

            if times:
                avg_time = statistics.mean(times)
                results[model] = {
                    "avg_time": avg_time,
                    "times": times,
                    "successful": successful,
                }
                print(
                    f"   ⚡ Average: {avg_time:.0f}ms ({successful}/{len(test_images)} successful)"
                )

        except Exception as e:
            print(f"   ❌ Failed to initialize: {str(e)}")

    if not results:
        print("\n❌ No successful tests")
        return

    print("\n" + "=" * 60)
    print("📊 RESULTS & us-central1 ESTIMATES")
    print("=" * 60)

    for model, data in results.items():
        current_time = data["avg_time"]

        # Conservative estimates for regional improvement
        # Network latency reduction: ~80-120ms
        # More stable connections: ~10-20% improvement
        latency_reduction = 100  # ms
        stability_improvement = 0.15  # 15% faster

        # Combined improvement (conservative)
        estimated_gcp_time = max(
            current_time - latency_reduction,  # Latency reduction
            current_time * (1 - stability_improvement),  # Stability improvement
        )

        improvement = ((current_time - estimated_gcp_time) / current_time) * 100

        print(f"\n🤖 {model}:")
        print(f"   📱 Current (MacBook): {current_time:.0f}ms")
        print(f"   🌍 Expected (us-central1): {estimated_gcp_time:.0f}ms")
        print(f"   🚀 Improvement: {improvement:.1f}% faster")
        print(f"   📊 Range: {min(data['times']):.0f}-{max(data['times']):.0f}ms")

    # Overall recommendation
    best_current = min(results.values(), key=lambda x: x["avg_time"])
    best_model = [k for k, v in results.items() if v == best_current][0]

    current_avg = statistics.mean([r["avg_time"] for r in results.values()])
    estimated_avg = current_avg * 0.75  # Conservative 25% improvement

    print(f"\n🎯 SUMMARY:")
    print(f"   🏆 Fastest model: {best_model} ({best_current['avg_time']:.0f}ms)")
    print(f"   📱 Current average: {current_avg:.0f}ms")
    print(f"   🌍 Expected us-central1: {estimated_avg:.0f}ms")
    print(
        f"   📈 Estimated throughput increase: {((current_avg - estimated_avg) / current_avg) * 100:.1f}%"
    )

    # Production recommendation
    throughput_current = 3600000 / current_avg
    throughput_gcp = 3600000 / estimated_avg

    print(f"\n💡 PRODUCTION IMPACT:")
    print(f"   📱 Current: ~{throughput_current:.0f} docs/hour")
    print(f"   🌍 Expected: ~{throughput_gcp:.0f} docs/hour")
    print(
        f"   🚀 Additional capacity: +{throughput_gcp - throughput_current:.0f} docs/hour"
    )

    if improvement > 15:
        print(f"\n✅ RECOMMENDATION: Deploy to us-central1!")
        print(f"   Significant speed improvement expected")
    else:
        print(f"\n✅ RECOMMENDATION: Your current setup is already quite optimized")
        print(f"   us-central1 would provide stability benefits more than speed")


if __name__ == "__main__":
    quick_vertex_test()
