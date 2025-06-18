#!/usr/bin/env python3.11
"""
🏁 VERTEX AI REGIONAL SPEED TEST

Test Vertex AI performance from within us-central1 to get realistic production speeds.
Only tests Vertex AI models to focus on regional performance.
"""

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Get project root and add to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vertex_ai_document_classifier import VertexAIDocumentClassifier


def load_ground_truth():
    """Load ground truth labels from JSON file"""
    ground_truth_file = project_root / "test-images" / "ground-truth.json"
    if not ground_truth_file.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_file}\n"
            f"Please run: python scripts/generate_ground_truth.py"
        )

    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    # Show data distribution
    categories = {}
    for category in ground_truth.values():
        categories[category] = categories.get(category, 0) + 1

    print(f"📋 Loaded {len(ground_truth)} ground truth labels")
    for category, count in sorted(categories.items()):
        print(f"   {category}: {count} images")

    return ground_truth


def calculate_metrics(predictions, ground_truth):
    """Calculate classification metrics"""
    # Filter ground truth to only include files we have predictions for
    valid_ground_truth = {k: v for k, v in ground_truth.items() if k in predictions}

    if not valid_ground_truth:
        return {
            "accuracy": 0,
            "correct": 0,
            "total": 0,
            "macro_precision": 0,
            "macro_recall": 0,
            "macro_f1": 0,
            "category_metrics": {},
        }

    # Calculate accuracy
    correct = sum(
        1
        for file, pred in predictions.items()
        if pred == valid_ground_truth.get(file) and pred not in ["error", "unknown"]
    )
    total = len(valid_ground_truth)
    accuracy = correct / total if total > 0 else 0

    # Get unique categories
    all_categories = set(valid_ground_truth.values()) | set(predictions.values())
    all_categories.discard("error")
    all_categories.discard("unknown")

    # Calculate per-category metrics
    category_metrics = {}
    for category in all_categories:
        tp = sum(
            1
            for file in valid_ground_truth
            if valid_ground_truth[file] == category
            and predictions.get(file) == category
        )
        fp = sum(
            1
            for file in valid_ground_truth
            if valid_ground_truth[file] != category
            and predictions.get(file) == category
        )
        fn = sum(
            1
            for file in valid_ground_truth
            if valid_ground_truth[file] == category
            and predictions.get(file) != category
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
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
            "support": sum(1 for v in valid_ground_truth.values() if v == category),
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


def test_vertex_model(model_name, ground_truth, test_images):
    """Test a single Vertex AI model against ground truth"""

    print(f"\n🤖 Testing Vertex AI Model: {model_name}")
    print(f"   🌍 Testing from us-central1 region")
    print(f"   🚀 Using OPTIMIZED settings with structured output")

    try:
        classifier = VertexAIDocumentClassifier(
            model=model_name,
            prompt_type="detailed",  # 🎯 OPTIMAL: Structured output prompt
            parameter_set="optimal",  # 🚀 OPTIMAL: Best speed parameters
        )

        # Build list of valid image paths
        valid_images = []
        for image_file in test_images:
            image_path = project_root / "test-images" / image_file
            if image_path.exists():
                valid_images.append((image_file, str(image_path)))
            else:
                print(f"   ⚠️  Skipping missing file: {image_file}")

        print(f"   🚀 Processing {len(valid_images)} images in PARALLEL...")

        # Process images in parallel
        predictions = {}
        processing_times = []
        confidences = []
        per_file_performance = {}

        start_time = time.time()

        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
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

                        # Show progress
                        expected = ground_truth.get(image_file, "unknown")
                        correct = "✅" if result.document_type == expected else "❌"
                        print(
                            f"   [{completed:2d}/{len(valid_images)}] {image_file} ({file_size_kb:.1f} KB): {result.document_type} (conf: {result.confidence:.2f}) - {result.processing_time_ms:.0f}ms {correct}"
                        )
                    else:
                        predictions[image_file] = "error"
                        per_file_performance[image_file] = {
                            "error": "Classification returned None",
                            "file_size_kb": round(file_size_kb, 2),
                        }
                        print(
                            f"   ❌ [{completed:2d}/{len(valid_images)}] {image_file}: Classification failed"
                        )
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    print(
                        f"   ❌ [{completed:2d}/{len(valid_images)}] {image_file}: {error_message}"
                    )
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
            "region": "us-central1",
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
            "region": "us-central1",
            "error": str(e),
            "success": False,
        }


def main():
    """Run Vertex AI regional speed test"""

    print("🌍 VERTEX AI REGIONAL SPEED TEST")
    print("Testing from us-central1 for realistic production performance")
    print("=" * 70)

    # Load ground truth
    try:
        ground_truth = load_ground_truth()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Get available test images
    test_images = list(ground_truth.keys())
    print(f"\n🔍 Testing with {len(test_images)} labeled images")

    # Vertex AI models only - focus on regional performance
    vertex_models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    print(f"\n🤖 Testing {len(vertex_models)} Vertex AI models from us-central1:")
    for model in vertex_models:
        print(f"   • {model} (v2.0 STABLE)")

    print(f"\n🎯 REGIONAL TEST CONFIGURATION:")
    print(f"   Region: us-central1 (production environment)")
    print(f"   Platform: Vertex AI only")
    print(f"   Prompt: detailed (structured output)")
    print(f"   Parameters: optimal (proven best)")
    print(f"   Tokens: 50 (JSON optimized)")

    # Run tests
    results = []
    for model_name in vertex_models:
        result = test_vertex_model(model_name, ground_truth, test_images)
        results.append(result)

    # Show results
    print("\n" + "=" * 80)
    print("🏆 VERTEX AI REGIONAL SPEED RESULTS")
    print("Performance from us-central1 region")
    print("=" * 80)

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("❌ No successful models to compare")
        return

    # Model comparison table
    print(f"\n🚀 VERTEX AI MODEL SPEEDS (us-central1):")
    print(f"{'Model':<25} {'Accuracy':<10} {'Avg Time':<12} {'Confidence':<12}")
    print("-" * 69)

    # Sort by speed (faster = better)
    sorted_results = sorted(
        successful_results, key=lambda x: x["performance"]["avg_processing_time_ms"]
    )

    for result in sorted_results:
        model_name = result["model"][:24]
        accuracy = f"{result['metrics']['accuracy']:.1%}"
        avg_time = f"{result['performance']['avg_processing_time_ms']:.0f}ms"
        confidence = f"{result['performance']['avg_confidence']:.3f}"

        print(f"{model_name:<25} {accuracy:<10} {avg_time:<12} {confidence:<12}")

    # Speed comparison
    if sorted_results:
        fastest = sorted_results[0]
        print(f"\n🏆 FASTEST VERTEX AI MODEL (us-central1):")
        print(f"   🥇 Winner: {fastest['model']}")
        print(f"       Accuracy: {fastest['metrics']['accuracy']:.1%}")
        print(
            f"       Speed: {fastest['performance']['avg_processing_time_ms']:.0f}ms avg"
        )
        print(f"       Confidence: {fastest['performance']['avg_confidence']:.3f}")

        if len(sorted_results) > 1:
            print(f"\n⚡ SPEED COMPARISON:")
            fastest_time = fastest["performance"]["avg_processing_time_ms"]
            for result in sorted_results:
                model_name = result["model"][:20]
                time_ms = result["performance"]["avg_processing_time_ms"]
                relative_speed = time_ms / fastest_time if fastest_time > 0 else 1
                speed_diff = (
                    f"{relative_speed:.1f}x" if relative_speed > 1 else "FASTEST"
                )
                print(f"   {model_name:<20} {time_ms:>6.0f}ms  ({speed_diff})")

    # Save results
    results_file = project_root / "results" / "vertex-ai-us-central1-speed.json"
    results_file.parent.mkdir(exist_ok=True)

    # Prepare data for JSON
    json_results = []
    for result in results:
        if result["success"]:
            json_result = {
                "model": result["model"],
                "region": "us-central1",
                "platform": "vertex_ai",
                "accuracy": result["metrics"]["accuracy"],
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
            }
        else:
            json_result = {
                "model": result["model"],
                "region": "us-central1",
                "platform": "vertex_ai",
                "error": result["error"],
                "success": False,
            }
        json_results.append(json_result)

    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_description": "Vertex AI speed test from us-central1 region",
                "region": "us-central1",
                "platform": "vertex_ai",
                "total_test_images": len(test_images),
                "models_tested": vertex_models,
                "results": json_results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Results saved to: {results_file}")

    # Performance summary
    if successful_results:
        avg_speed = statistics.mean(
            [r["performance"]["avg_processing_time_ms"] for r in successful_results]
        )
        avg_accuracy = statistics.mean(
            [r["metrics"]["accuracy"] for r in successful_results]
        )

        print(f"\n📊 VERTEX AI REGIONAL PERFORMANCE SUMMARY:")
        print(f"   🌍 Region: us-central1")
        print(f"   🎯 Average Accuracy: {avg_accuracy:.1%}")
        print(f"   ⚡ Average Speed: {avg_speed:.0f}ms")
        print(f"   🚀 Expected throughput: {3600000/avg_speed:.0f} docs/hour")


if __name__ == "__main__":
    main()
