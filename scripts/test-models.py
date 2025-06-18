#!/usr/bin/env python3.11
"""
🏁 HEAD-TO-HEAD PLATFORM COMPARISON

Comprehensive testing comparing Regular API vs Vertex AI:
- Same models tested on both platforms
- Same optimal settings and parameters
- Direct performance comparison
- Accuracy, speed, and reliability metrics

Tests best models from each platform to determine the winner.
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, Counter
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.document_classifier import UltraFastDocumentClassifier
from src.core.vertex_ai_document_classifier import VertexAIDocumentClassifier


def load_ground_truth():
    """Load ground truth data from test-images/ground-truth.json"""
    ground_truth_file = project_root / "test-images" / "ground-truth.json"

    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    print(f"📋 Loaded {len(ground_truth)} ground truth labels")

    # Show distribution
    categories = Counter(ground_truth.values())
    for category, count in categories.most_common():
        print(f"   {category}: {count} images")

    return ground_truth


def calculate_metrics(predictions, ground_truth):
    """Calculate comprehensive accuracy metrics"""

    # Overall accuracy
    correct = sum(
        1
        for filename, pred_type in predictions.items()
        if ground_truth.get(filename) == pred_type
    )
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    # Per-category metrics
    categories = set(ground_truth.values()) | set(predictions.values())
    category_metrics = {}

    for category in categories:
        # True positives, false positives, false negatives
        tp = sum(
            1
            for filename, pred_type in predictions.items()
            if pred_type == category and ground_truth.get(filename) == category
        )
        fp = sum(
            1
            for filename, pred_type in predictions.items()
            if pred_type == category and ground_truth.get(filename) != category
        )
        fn = sum(
            1
            for filename, true_type in ground_truth.items()
            if true_type == category and predictions.get(filename) != category
        )

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        category_metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": sum(1 for v in ground_truth.values() if v == category),
        }

    # Macro averages
    macro_precision = statistics.mean(
        [m["precision"] for m in category_metrics.values()]
    )
    macro_recall = statistics.mean([m["recall"] for m in category_metrics.values()])
    macro_f1 = statistics.mean([m["f1"] for m in category_metrics.values()])

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "category_metrics": category_metrics,
    }


def test_model(model_name, platform, ground_truth, test_images):
    """Test a single model against ground truth on specified platform"""

    print(f"\n🤖 Testing Model: {model_name} ({platform.upper()})")
    print(f"   Using OPTIMIZED settings with structured output")

    try:
        # Use appropriate classifier based on platform
        if platform == "regular_api":
            classifier = UltraFastDocumentClassifier(
                model=model_name,
                prompt_type="detailed",  # 🎯 OPTIMAL: Fastest prompt
                parameter_set="optimal",  # 🚀 OPTIMAL: Best speed parameters
            )
        elif platform == "vertex_ai":
            classifier = VertexAIDocumentClassifier(
                model=model_name,
                prompt_type="detailed",  # 🎯 OPTIMAL: Structured output prompt
                parameter_set="optimal",  # 🚀 OPTIMAL: Best speed parameters
            )
        else:
            raise ValueError(f"Unknown platform: {platform}")

        # Build list of valid image paths
        valid_images = []
        for image_file in test_images:
            image_path = project_root / "test-images" / image_file
            if image_path.exists():
                valid_images.append((image_file, str(image_path)))
            else:
                print(f"   ⚠️  Skipping missing file: {image_file}")

        print(f"   🚀 Processing {len(valid_images)} images in PARALLEL...")

        # Process images in parallel with correct filename mapping
        predictions = {}
        processing_times = []
        confidences = []
        per_file_performance = {}

        start_time = time.time()

        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks with filename mapping
            future_to_file = {
                executor.submit(classifier.classify_single, image_path): image_file
                for image_file, image_path in valid_images
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                image_file = future_to_file[future]
                completed += 1

                image_path = project_root / "test-images" / image_file
                file_size_kb = (
                    image_path.stat().st_size / 1024 if image_path.exists() else 0
                )

                try:
                    result = future.result(timeout=30)
                    if result is not None:
                        predictions[image_file] = result.document_type
                        processing_times.append(result.processing_time_ms)
                        confidences.append(result.confidence)
                        per_file_performance[image_file] = {
                            "document_type": result.document_type,
                            "confidence": result.confidence,
                            "processing_time_ms": result.processing_time_ms,
                            "file_size_kb": round(file_size_kb, 2),
                        }

                        # Show progress with real-time feedback
                        expected = ground_truth.get(image_file, "unknown")
                        correct = "✅" if result.document_type == expected else "❌"
                        print(
                            f"   [{completed:2d}/{len(valid_images)}] {image_file} ({file_size_kb:.1f} KB): {result.document_type} (conf: {result.confidence:.2f}) - {result.processing_time_ms:.0f}ms {correct}"
                        )
                    else:
                        # Handle None result
                        predictions[image_file] = "error"
                        per_file_performance[image_file] = {
                            "error": "Classification returned None",
                            "file_size_kb": round(file_size_kb, 2),
                        }
                        print(
                            f"   ❌ [{completed:2d}/{len(valid_images)}] {image_file}: Classification failed (returned None)"
                        )
                except Exception as e:
                    error_message = f"Error processing {image_file}: {str(e)}"
                    print(f"   ❌ {error_message}")
                    predictions[image_file] = "error"
                    per_file_performance[image_file] = {
                        "error": error_message,
                        "file_size_kb": round(file_size_kb, 2),
                    }

        total_batch_time = time.time() - start_time
        print(
            f"   ⚡ Completed {len(valid_images)} images in {total_batch_time:.1f}s (PARALLEL)"
        )

        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth)

        # Performance stats
        avg_time = statistics.mean(processing_times) if processing_times else 0
        avg_confidence = statistics.mean(confidences) if confidences else 0

        return {
            "model": model_name,
            "platform": platform,
            "metrics": metrics,
            "predictions": predictions,
            "per_file_performance": per_file_performance,
            "performance": {
                "avg_processing_time_ms": avg_time,
                "avg_confidence": avg_confidence,
                "total_time_s": sum(processing_times) / 1000,
                "images_per_second": (
                    len(processing_times) / (sum(processing_times) / 1000)
                    if processing_times
                    else 0
                ),
            },
            "success": True,
        }

    except Exception as e:
        print(f"   ❌ Model failed: {str(e)}")
        return {
            "model": model_name,
            "platform": platform,
            "error": str(e),
            "success": False,
        }


def print_model_comparison(results):
    """Print comprehensive model comparison"""

    print("\n" + "=" * 80)
    print("🏁 HEAD-TO-HEAD PLATFORM COMPARISON")
    print("Regular API vs Vertex AI - Identical settings")
    print("=" * 80)

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("❌ No successful models to compare")
        return

    # Model comparison table
    print(f"\n🏆 PLATFORM BATTLE RESULTS")
    print(
        f"{'Model + Platform':<45} {'Accuracy':<10} {'Macro F1':<10} {'Avg Time':<12} {'Confidence':<12}"
    )
    print("-" * 99)

    # Sort by accuracy first, then by F1, then by speed (lower time = better)
    sorted_results = sorted(
        successful_results,
        key=lambda x: (
            x["metrics"]["accuracy"],
            x["metrics"]["macro_f1"],
            -x["performance"]["avg_processing_time_ms"],
        ),
        reverse=True,
    )

    for result in sorted_results:
        platform_name = (
            "Regular API" if result["platform"] == "regular_api" else "Vertex AI"
        )
        model_platform = f"{result['model']} ({platform_name})"[
            :44
        ]  # Increased from 29 to 44
        accuracy = f"{result['metrics']['accuracy']:.1%}"
        macro_f1 = f"{result['metrics']['macro_f1']:.3f}"
        avg_time = f"{result['performance']['avg_processing_time_ms']:.0f}ms"
        confidence = f"{result['performance']['avg_confidence']:.3f}"

        print(
            f"{model_platform:<45} {accuracy:<10} {macro_f1:<10} {avg_time:<12} {confidence:<12}"
        )

    # Champions in each category
    best_accuracy = max(successful_results, key=lambda x: x["metrics"]["accuracy"])
    best_f1 = max(successful_results, key=lambda x: x["metrics"]["macro_f1"])
    fastest = min(
        successful_results, key=lambda x: x["performance"]["avg_processing_time_ms"]
    )
    most_confident = max(
        successful_results, key=lambda x: x["performance"]["avg_confidence"]
    )

    print(f"\n🏆 PLATFORM CHAMPIONS:")
    best_accuracy_platform = (
        "Regular API" if best_accuracy["platform"] == "regular_api" else "Vertex AI"
    )
    best_f1_platform = (
        "Regular API" if best_f1["platform"] == "regular_api" else "Vertex AI"
    )
    fastest_platform = (
        "Regular API" if fastest["platform"] == "regular_api" else "Vertex AI"
    )
    most_confident_platform = (
        "Regular API" if most_confident["platform"] == "regular_api" else "Vertex AI"
    )

    print(
        f"   🎯 Most Accurate: {best_accuracy['model']} on {best_accuracy_platform} ({best_accuracy['metrics']['accuracy']:.1%})"
    )
    print(
        f"   📊 Best F1 Score: {best_f1['model']} on {best_f1_platform} ({best_f1['metrics']['macro_f1']:.3f})"
    )
    print(
        f"   🚀 Fastest: {fastest['model']} on {fastest_platform} ({fastest['performance']['avg_processing_time_ms']:.0f}ms)"
    )
    print(
        f"   💪 Most Confident: {most_confident['model']} on {most_confident_platform} ({most_confident['performance']['avg_confidence']:.3f})"
    )

    # Detailed breakdown for best overall model
    best_overall = sorted_results[0]
    best_overall_platform = (
        "Regular API" if best_overall["platform"] == "regular_api" else "Vertex AI"
    )
    print(f"\n🥇 OVERALL WINNER: {best_overall['model']} on {best_overall_platform}")
    print(
        f"   Overall Accuracy: {best_overall['metrics']['accuracy']:.1%} ({best_overall['metrics']['correct']}/{best_overall['metrics']['total']})"
    )
    print(f"   Macro F1: {best_overall['metrics']['macro_f1']:.3f}")
    print(
        f"   Avg Processing Time: {best_overall['performance']['avg_processing_time_ms']:.0f}ms"
    )
    print(f"   Avg Confidence: {best_overall['performance']['avg_confidence']:.3f}")

    # Speed comparison
    print(f"\n⚡ SPEED BATTLE (Platform Comparison):")
    speed_sorted = sorted(
        successful_results, key=lambda x: x["performance"]["avg_processing_time_ms"]
    )

    if speed_sorted:
        fastest_time = speed_sorted[0]["performance"]["avg_processing_time_ms"]

        # Handle case where fastest time is 0 (failed model)
        if fastest_time == 0:
            print(
                "   ⚠️  Speed comparison skipped (fastest model had 0ms due to failures)"
            )
        else:
            for result in speed_sorted:
                platform_name = (
                    "Regular API"
                    if result["platform"] == "regular_api"
                    else "Vertex AI"
                )
                model_platform = f"{result['model']} ({platform_name})"[
                    :40
                ]  # Increased from 25 to 40
                time_ms = result["performance"]["avg_processing_time_ms"]
                relative_speed = time_ms / fastest_time if fastest_time > 0 else 1
                speed_diff = (
                    f"{relative_speed:.1f}x" if relative_speed > 1 else "FASTEST"
                )

                print(f"   {model_platform:<40} {time_ms:>6.0f}ms  ({speed_diff})")
    else:
        print("   ⚠️  No successful models to compare")


def main():
    """Run head-to-head platform comparison"""

    print("🏁 HEAD-TO-HEAD PLATFORM COMPARISON")
    print("Regular API vs Vertex AI - STABLE Gemini 2.0 models only")
    print("=" * 60)

    # Load ground truth
    try:
        ground_truth = load_ground_truth()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Get available test images
    test_images = list(ground_truth.keys())
    print(f"\n🔍 Testing with {len(test_images)} labeled images")

    # Test configurations: [model, platform]
    # Test configurations: [model, platform] - STABLE MODELS ONLY
    test_configs = [
        # Regular API models - 100% accuracy proven
        ("gemini-2.0-flash", "regular_api"),
        ("gemini-2.0-flash-lite", "regular_api"),
        # Vertex AI models - 100% accuracy proven
        ("gemini-2.0-flash", "vertex_ai"),
        ("gemini-2.0-flash-lite", "vertex_ai"),
    ]

    print(f"\n🤖 Testing {len(test_configs)} STABLE model/platform combinations:")
    for model, platform in test_configs:
        platform_name = "Regular API" if platform == "regular_api" else "Vertex AI"
        version = "2.0 (PROVEN STABLE)"
        print(f"   • {model} on {platform_name} (v{version})")

    print(f"\n🎯 CONFIGURATION (Both Platforms):")
    print(f"   Prompt: detailed (structured output)")
    print(f"   Parameters: optimal (proven best)")
    print(f"   Tokens: 50 (JSON optimized)")
    print(f"   Output: structured JSON (guaranteed format)")

    # Run tests
    results = []
    for model_name, platform in test_configs:
        result = test_model(model_name, platform, ground_truth, test_images)
        results.append(result)

    # Show model comparison
    print_model_comparison(results)

    # Save results
    results_file = project_root / "results" / "optimized-model-comparison.json"
    results_file.parent.mkdir(exist_ok=True)

    # Prepare data for JSON (remove non-serializable objects)
    json_results = []
    for result in results:
        if result["success"]:
            json_result = {
                "model": result["model"],
                "configuration": "detailed_prompt_ultimate_speed",
                "accuracy": result["metrics"]["accuracy"],
                "macro_precision": result["metrics"]["macro_precision"],
                "macro_recall": result["metrics"]["macro_recall"],
                "macro_f1": result["metrics"]["macro_f1"],
                "correct_predictions": result["metrics"]["correct"],
                "total_predictions": result["metrics"]["total"],
                "avg_processing_time_ms": result["performance"][
                    "avg_processing_time_ms"
                ],
                "avg_confidence": result["performance"]["avg_confidence"],
                "images_per_second": result["performance"]["images_per_second"],
                "predictions": result["predictions"],
                "per_file_performance": result.get("per_file_performance", {}),
                "category_metrics": result["metrics"]["category_metrics"],
            }
        else:
            json_result = {
                "model": result["model"],
                "error": result["error"],
                "success": False,
            }
        json_results.append(json_result)

    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_description": "Optimized model comparison using detailed prompt + ultimate_speed parameters",
                "configuration": {
                    "prompt_type": "detailed",
                    "parameter_set": "ultimate_speed",
                    "max_tokens": 50,
                    "structured_output": True,
                },
                "ground_truth_file": "test-images/ground-truth.json",
                "total_test_images": len(test_images),
                "models_tested": [
                    f"{model}_{platform}" for model, platform in test_configs
                ],
                "results": json_results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Results saved to: {results_file}")

    # Give final recommendation with platform analysis
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        best_model = max(
            successful_results,
            key=lambda x: (x["metrics"]["accuracy"], x["metrics"]["macro_f1"]),
        )
        best_platform = (
            "Regular API" if best_model["platform"] == "regular_api" else "Vertex AI"
        )

        print(f"\n🎯 FINAL RECOMMENDATION:")
        print(f"   🏆 Winner: {best_model['model']} on {best_platform}")
        print(f"       Accuracy: {best_model['metrics']['accuracy']:.1%}")
        print(f"       F1 Score: {best_model['metrics']['macro_f1']:.3f}")
        print(
            f"       Speed: {best_model['performance']['avg_processing_time_ms']:.0f}ms avg"
        )
        print(f"       Confidence: {best_model['performance']['avg_confidence']:.3f}")

        # Platform analysis
        regular_api_results = [
            r for r in successful_results if r["platform"] == "regular_api"
        ]
        vertex_ai_results = [
            r for r in successful_results if r["platform"] == "vertex_ai"
        ]

        print(f"\n📊 PLATFORM ANALYSIS:")
        regular_avg_time = 0
        regular_avg_accuracy = 0
        vertex_avg_time = 0
        vertex_avg_accuracy = 0

        if regular_api_results:
            regular_avg_time = statistics.mean(
                [
                    r["performance"]["avg_processing_time_ms"]
                    for r in regular_api_results
                ]
            )
            regular_avg_accuracy = statistics.mean(
                [r["metrics"]["accuracy"] for r in regular_api_results]
            )
            print(
                f"   📱 Regular API: {regular_avg_accuracy:.1%} avg accuracy, {regular_avg_time:.0f}ms avg speed"
            )

        if vertex_ai_results:
            vertex_avg_time = statistics.mean(
                [r["performance"]["avg_processing_time_ms"] for r in vertex_ai_results]
            )
            vertex_avg_accuracy = statistics.mean(
                [r["metrics"]["accuracy"] for r in vertex_ai_results]
            )
            print(
                f"   🏢 Vertex AI: {vertex_avg_accuracy:.1%} avg accuracy, {vertex_avg_time:.0f}ms avg speed"
            )

        # Determine platform winner
        if regular_api_results and vertex_ai_results:
            if (
                vertex_avg_accuracy >= regular_avg_accuracy
                and vertex_avg_time <= regular_avg_time * 1.1
            ):  # Within 10% speed
                platform_winner = "Vertex AI"
                platform_reason = "better/equal accuracy with similar speed"
            elif regular_avg_accuracy > vertex_avg_accuracy:
                platform_winner = "Regular API"
                platform_reason = "higher accuracy"
            elif regular_avg_time < vertex_avg_time * 0.9:  # More than 10% faster
                platform_winner = "Regular API"
                platform_reason = "significantly faster"
            else:
                platform_winner = "Vertex AI"
                platform_reason = "enterprise features with comparable performance"

            print(f"   🏆 Platform Winner: {platform_winner} ({platform_reason})")


if __name__ == "__main__":
    main()
