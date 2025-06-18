#!/usr/bin/env python3.11

import os
import time
import json
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from dotenv import load_dotenv
import io
from pathlib import Path
from PIL import Image
import mimetypes
from pdf2image import convert_from_path

# Import configurations
from src.config import PROMPTS, SUPPORTED_MODELS
from src.config.structured_output import (
    DOCUMENT_CLASSIFICATION_SCHEMA,
    validate_schema_response,
)

# Load environment variables from .env.local in project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env.local")

# Fix gRPC fork handler issues in parallel processing
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"


@dataclass
class ClassificationResult:
    """Document classification result with metadata"""

    document_type: str
    confidence: float
    processing_time_ms: float
    model_used: str
    inference_id: str


class UltraFastDocumentClassifier:
    """
    Ultra-fast document classifier optimized for sub-500ms performance
    Based on proven speed optimization techniques
    """

    def __init__(
        self,
        model: str = "detailed",
        prompt_type: str = "detailed",
        parameter_set: str = "optimal",
    ):
        """
        Initialize classifier with optimized settings

        Args:
            model: Model name to use
            prompt_type: Type of prompt from PROMPTS config
            parameter_set: Parameter set from MODEL_PARAMETERS config
        """
        self.model = model
        self.prompt_type = prompt_type
        self.parameter_set = parameter_set

        # API and model setup
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)

        # Get model configuration
        model_info = SUPPORTED_MODELS.get(self.model)
        if not model_info:
            raise ValueError(f"Model {self.model} not found in supported models.")
        self.model_instance = genai.GenerativeModel(model_info["name"])

        self.prompt = PROMPTS.get(self.prompt_type, PROMPTS["detailed"])

        # Get parameters from the model config
        params = model_info.get("parameters", {})

        # Get and set structured output schema
        self.schema = DOCUMENT_CLASSIFICATION_SCHEMA
        self.generation_config = genai.GenerationConfig(
            **params,
            response_mime_type="application/json",
            response_schema=self.schema,
        )

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _parse_response_text(self, response_text: str) -> dict:
        """
        Parse response text handling both regular JSON and thinking responses.

        Gemini 2.5 models with "thinking on by default" may return:
        - Regular JSON (for older models)
        - Thinking text followed by JSON (for 2.5 models)
        """
        # First try to parse as regular JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # If that fails, try to extract JSON from thinking response
        # Look for JSON objects in the response text
        import re

        # Try to find JSON objects with common patterns
        json_patterns = [
            r'\{[^{}]*"document_type"[^{}]*\}',  # Look for our specific schema
            r'\{.*?"document_type".*?\}',  # More flexible pattern
            r'\{[^{}]*"confidence"[^{}]*\}',  # Alternative pattern
            r"\{.*\}",  # Last resort - any JSON object
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    # Check if it looks like our expected schema
                    if isinstance(parsed, dict) and (
                        "document_type" in parsed or "confidence" in parsed
                    ):
                        return parsed
                except json.JSONDecodeError:
                    continue

        # If no JSON found, try to extract from the end of the response
        # Sometimes thinking models put JSON at the end
        lines = response_text.strip().split("\n")
        for i in range(len(lines)):
            try:
                # Try parsing from each line to the end
                candidate = "\n".join(lines[i:])
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        # If all else fails, raise the original error
        raise json.JSONDecodeError(
            f"Could not parse JSON from response: {response_text[:200]}...",
            response_text,
            0,
        )

    def _prepare_image(
        self, image_path: str, max_size_kb: int = 1000, resize_dim: int = 1024
    ):
        """Prepare image for API, resizing if necessary."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        # If file is large, optimize it
        mime_type, _ = mimetypes.guess_type(path)
        file_size_kb = path.stat().st_size / 1024

        if mime_type and path.stat().st_size > max_size_kb * 1024:
            if "image" in mime_type:
                # Resize large images
                print(f"    Resizing {path.name} ({file_size_kb:.1f} KB)...")
                with Image.open(path) as img:
                    img.thumbnail((resize_dim, resize_dim))
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format="JPEG", quality=80)
                    img_byte_arr.seek(0)

                    return {"mime_type": "image/jpeg", "data": img_byte_arr.read()}

            elif "pdf" in mime_type:
                # Convert PDF first page to optimized image for classification
                print(f"    Converting {path.name} ({file_size_kb:.1f} KB) to image...")
                try:
                    # Convert first page to PIL Image
                    images = convert_from_path(
                        str(path),
                        first_page=1,  # Only first page
                        last_page=1,
                        dpi=150,  # Good quality for classification
                        fmt="RGB",
                    )

                    if not images:
                        print(f"      PDF conversion failed, using original")
                        # Fall through to use original file
                    else:
                        first_page_img = images[0]
                        print(
                            f"      Converted to image: {first_page_img.size[0]}x{first_page_img.size[1]} pixels"
                        )

                        # Apply image optimization (thumbnail + JPEG compression)
                        first_page_img.thumbnail((resize_dim, resize_dim))
                        if first_page_img.mode != "RGB":
                            first_page_img = first_page_img.convert("RGB")

                        img_byte_arr = io.BytesIO()
                        first_page_img.save(img_byte_arr, format="JPEG", quality=80)
                        img_byte_arr.seek(0)

                        optimized_data = img_byte_arr.read()
                        optimized_size_kb = len(optimized_data) / 1024
                        reduction_ratio = optimized_size_kb / file_size_kb

                        print(
                            f"      Optimized to: {optimized_size_kb:.1f} KB ({reduction_ratio:.1%} of original)"
                        )

                        return {"mime_type": "image/jpeg", "data": optimized_data}

                except Exception as e:
                    print(f"      PDF to image conversion failed: {e}, using original")
                    # Fall through to use original file

        # Use original file (small files or compression failed)
        return {
            "mime_type": mime_type or "application/octet-stream",
            "data": path.read_bytes(),
        }

    def classify_single(self, image_path: str) -> Optional[ClassificationResult]:
        """Classify a single document image."""
        start_time = time.time()
        inference_id = f"gemini_{int(time.time() * 1000)}"

        try:
            image_content = self._prepare_image(image_path)

            response = self.model_instance.generate_content(
                [self.prompt, image_content],
                generation_config=self.generation_config,
            )

            if not response.parts:
                return None

            # Handle both regular JSON responses and thinking responses
            result_json = self._parse_response_text(response.text)

            # Validate response
            if not validate_schema_response(result_json):
                raise ValueError("Response does not match the required schema.")

            processing_time = (time.time() - start_time) * 1000

            return ClassificationResult(
                document_type=result_json.get("document_type", "other"),
                confidence=result_json.get("confidence", 0.0),
                processing_time_ms=processing_time,
                model_used=self.model,
                inference_id=inference_id,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"Classification error for {image_path}: {e}")
            return None

    def classify_batch(
        self, image_paths: List[str], max_workers: Optional[int] = None
    ) -> List[ClassificationResult]:
        """
        Classify multiple documents in parallel
        """
        if max_workers is None:
            max_workers = 5  # Default to 5 workers

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.classify_single, path): path
                for path in image_paths
            }

            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    timeout_val = 30  # Default timeout
                    result = future.result(timeout=timeout_val)
                    if result:
                        results.append(result)
                        print(
                            f"✓ {path}: {result.document_type} ({result.confidence:.2f}) - {result.processing_time_ms:.0f}ms"
                        )
                except Exception as e:
                    print(f"✗ {path}: Error - {e}")

        return results

    def get_performance_stats(self, results: List[ClassificationResult]) -> Dict:
        """Get performance statistics"""
        if not results:
            return {}

        processing_times = [r.processing_time_ms for r in results]
        successful_results = [r for r in results if r.confidence > 0]

        return {
            "total_documents": len(results),
            "successful_classifications": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "min_processing_time_ms": min(processing_times),
            "max_processing_time_ms": max(processing_times),
            "total_processing_time_s": sum(processing_times) / 1000,
            "documents_per_second": len(results) / (sum(processing_times) / 1000),
            "avg_confidence": (
                sum(r.confidence for r in successful_results) / len(successful_results)
                if successful_results
                else 0
            ),
            "categories_found": list(set(r.document_type for r in successful_results)),
            "model_used": self.model,
            "prompt_type": self.prompt_type,
            "parameter_set": self.parameter_set,
        }

    def get_supported_categories(self) -> List[str]:
        """Get list of supported document categories"""
        return list(SUPPORTED_MODELS.keys())

    def change_model(self, new_model: str) -> bool:
        """
        Change the model being used

        Args:
            new_model: Name of the new model to use

        Returns:
            True if successful, False if model not available
        """
        if new_model not in SUPPORTED_MODELS:
            print(f"❌ Model {new_model} is not available")
            return False

        # Reinitialize with new model
        old_model = self.model
        try:
            self.__init__(new_model, self.prompt_type, self.parameter_set)
            print(f"✅ Changed from {old_model} to {new_model}")
            return True
        except Exception as e:
            print(f"❌ Failed to change to {new_model}: {e}")
            # Restore old model
            self.__init__(old_model, self.prompt_type, self.parameter_set)
            return False

    def change_prompt(self, new_prompt_type: str):
        """Change the prompt type."""
        self.prompt_type = new_prompt_type
        self.prompt = PROMPTS.get(new_prompt_type, PROMPTS["detailed"])
        print(f"✅ Changed to prompt: {new_prompt_type}")


def main():
    """Test the ultra-fast classifier"""
    print("🚀 Ultra-Fast Document Classifier")
    print("=" * 50)

    # Initialize classifier with config
    classifier = UltraFastDocumentClassifier()

    # Display supported categories
    print(
        f"📋 Supported categories: {', '.join(classifier.get_supported_categories())}"
    )

    # Auto-detect test images
    import glob

    test_images = (
        glob.glob("test-images/*.jpg")
        + glob.glob("test-images/*.jpeg")
        + glob.glob("test-images/*.png")
        + glob.glob("test-images/*.pdf")
    )
    test_images.sort()

    if not test_images:
        print("❌ No test images found in test-images/ directory")
        print("   Add .jpg, .jpeg, .png, or .pdf files to test-images/")
        return

    print(f"🔍 Found {len(test_images)} test images")
    print(f"📄 Testing with first 5 images...")
    test_images = test_images[:5]  # Limit to first 5 for quick testing

    # Run classification
    start_time = time.time()
    results = classifier.classify_batch(test_images)
    total_time = time.time() - start_time

    # Display results
    print(f"\n📊 Performance Results:")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📈 Documents/second: {len(results)/total_time:.1f}")

    stats = classifier.get_performance_stats(results)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"📋 {key}: {value:.2f}")
        else:
            print(f"📋 {key}: {value}")


if __name__ == "__main__":
    main()
