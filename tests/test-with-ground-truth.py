#!/usr/bin/env python3.11
"""
Test script using ground truth validation
Tests the classifier against known document types
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.document_classifier import UltraFastDocumentClassifier, ClassificationResult


def load_ground_truth(file_path: str = "tests/ground-truth.json") -> Dict:
    """Load ground truth data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Ground truth file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in ground truth file: {e}")
        return {}


def validate_classification(result: ClassificationResult, expected_type: str, confidence_threshold: float) -> Tuple[bool, str]:
    """Validate a classification result against ground truth"""
    if not result:
        return False, "Classification failed - no result"
    
    type_correct = result.document_type.lower() == expected_type.lower()
    confidence_ok = result.confidence >= confidence_threshold
    
    if type_correct and confidence_ok:
        return True, "✅ PASS"
    elif type_correct and not confidence_ok:
        return False, f"❌ FAIL - Low confidence ({result.confidence:.2%} < {confidence_threshold:.2%})"
    else:
        return False, f"❌ FAIL - Wrong type (got {result.document_type}, expected {expected_type})"


def run_ground_truth_test():
    """Run classification test using ground truth validation"""
    print("🧪 Ground Truth Validation Test")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    # Load ground truth
    ground_truth = load_ground_truth()
    if not ground_truth or 'ground_truth' not in ground_truth:
        print("❌ No valid ground truth data found")
        return False
    
    truth_data = ground_truth['ground_truth']
    metadata = ground_truth.get('metadata', {})
    
    print(f"📊 Ground Truth Summary:")
    print(f"   • Total documents: {metadata.get('total_documents', len(truth_data))}")
    print(f"   • Document types: {metadata.get('document_types', {})}")
    print(f"   • Version: {metadata.get('version', 'unknown')}")
    
    # Initialize classifier
    try:
        classifier = UltraFastDocumentClassifier(
            model="gemini-2.0-flash",
            prompt_type="detailed"
        )
        print(f"✅ Classifier initialized with {classifier.model}")
    except Exception as e:
        print(f"❌ Classifier initialization failed: {e}")
        return False
    
    # Run tests
    print(f"\n🔍 Running Classification Tests...")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for file_path, expected in truth_data.items():
        expected_type = expected['expected_type']
        confidence_threshold = expected.get('confidence_threshold', 0.8)
        description = expected.get('description', '')
        
        print(f"\n📄 Testing: {file_path}")
        print(f"   Expected: {expected_type} (≥{confidence_threshold:.1%} confidence)")
        print(f"   Description: {description}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"   ❌ SKIP - File not found: {file_path}")
            results.append({
                'file': file_path,
                'expected': expected_type,
                'actual': None,
                'passed': False,
                'reason': 'File not found'
            })
            continue
        
        # Classify document
        try:
            result = classifier.classify_single(file_path)
            
            if result:
                # Validate result
                passed, status = validate_classification(result, expected_type, confidence_threshold)
                
                print(f"   Result: {result.document_type} ({result.confidence:.2%}) - {result.processing_time_ms:.0f}ms")
                print(f"   Status: {status}")
                
                results.append({
                    'file': file_path,
                    'expected': expected_type,
                    'actual': result.document_type,
                    'confidence': result.confidence,
                    'processing_time_ms': result.processing_time_ms,
                    'passed': passed,
                    'reason': status
                })
            else:
                print(f"   ❌ FAIL - Classification returned no result")
                results.append({
                    'file': file_path,
                    'expected': expected_type,
                    'actual': None,
                    'passed': False,
                    'reason': 'Classification failed'
                })
                
        except Exception as e:
            print(f"   ❌ ERROR - {e}")
            results.append({
                'file': file_path,
                'expected': expected_type,
                'actual': None,
                'passed': False,
                'reason': f'Exception: {e}'
            })
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    passed_tests = sum(1 for r in results if r['passed'])
    total_tests = len(results)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    successful_results = [r for r in results if r['passed'] and r.get('processing_time_ms')]
    if successful_results:
        avg_time = sum(r['processing_time_ms'] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
    else:
        avg_time = 0
        avg_confidence = 0
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"📊 Overall Performance:")
    print(f"   • Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"   • Total time: {total_time:.1f}s")
    print(f"   • Avg processing time: {avg_time:.0f}ms")
    print(f"   • Avg confidence: {avg_confidence:.2%}")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        filename = Path(result['file']).name
        expected = result['expected']
        actual = result.get('actual', 'None')
        confidence = result.get('confidence', 0)
        
        print(f"   {status} {filename}: {expected} → {actual} ({confidence:.1%})")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    if success_rate == 100:
        print("   🏆 Excellent! All tests passed.")
    elif success_rate >= 80:
        print("   👍 Good performance, most tests passed.")
    elif success_rate >= 60:
        print("   ⚠️  Moderate performance, some issues detected.")
    else:
        print("   🚨 Poor performance, significant issues detected.")
    
    if avg_time > 0:
        if avg_time < 400:
            print(f"   ⚡ Speed target achieved! ({avg_time:.0f}ms < 400ms)")
        else:
            print(f"   🐌 Speed target missed ({avg_time:.0f}ms > 400ms)")
    
    return success_rate == 100


if __name__ == "__main__":
    success = run_ground_truth_test()
    sys.exit(0 if success else 1) 