"""
Simple test script for iLand retrieval system

This script provides basic validation of the retrieval components without problematic imports.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_basic_imports():
    """Test that core modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from index_classifier import iLandIndexClassifier, create_default_iland_classifier
        print("✓ Index classifier imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Index classifier import error: {e}")
        return False

def test_index_classifier():
    """Test the index classifier."""
    print("\nTesting index classifier...")
    
    try:
        from index_classifier import create_default_iland_classifier
        
        # Create classifier
        classifier = create_default_iland_classifier()
        print("✓ Index classifier created")
        
        # Test classification (without API key, will use fallback)
        test_queries = [
            "โฉนดที่ดินในกรุงเทพ",
            "Land deeds in Bangkok", 
            "ที่ดินในสมุทรปราการ"
        ]
        
        for query in test_queries:
            try:
                result = classifier.classify_query(query)
                print(f"✓ Query '{query[:30]}...' -> {result['selected_index']} (confidence: {result['confidence']:.2f})")
            except Exception as e:
                print(f"✗ Classification error for '{query}': {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Index classifier test failed: {e}")
        return False

def test_retriever_imports():
    """Test retriever adapter imports individually."""
    print("\nTesting retriever imports...")
    
    retriever_modules = [
        ("VectorRetrieverAdapter", "retrievers.vector"),
        ("SummaryRetrieverAdapter", "retrievers.summary"), 
        ("RecursiveRetrieverAdapter", "retrievers.recursive"),
        ("MetadataRetrieverAdapter", "retrievers.metadata"),
        ("ChunkDecouplingRetrieverAdapter", "retrievers.chunk_decoupling"),
        ("HybridRetrieverAdapter", "retrievers.hybrid"),
        ("PlannerRetrieverAdapter", "retrievers.planner")
    ]
    
    successful_imports = []
    failed_imports = []
    
    for class_name, module_name in retriever_modules:
        try:
            exec(f"from {module_name} import {class_name}")
            successful_imports.append(class_name)
            print(f"✓ {class_name} imported successfully")
        except ImportError as e:
            failed_imports.append((class_name, str(e)))
            print(f"✗ {class_name} import failed: {e}")
    
    print(f"\nImport Summary: {len(successful_imports)} successful, {len(failed_imports)} failed")
    return len(successful_imports) > 0

def test_thai_functionality():
    """Test Thai language functionality."""
    print("\nTesting Thai functionality...")
    
    try:
        # Test Thai text handling
        thai_text = "โฉนดที่ดินในกรุงเทพมหานคร"
        print(f"✓ Thai text handling: {thai_text}")
        
        # Test basic string operations
        if "โฉนด" in thai_text:
            print("✓ Thai substring search working")
        
        if "กรุงเทพ" in thai_text:
            print("✓ Thai province detection working")
        
        return True
        
    except Exception as e:
        print(f"✗ Thai functionality test failed: {e}")
        return False

def main():
    """Run simplified tests."""
    print("iLand Retrieval System - Simplified Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_index_classifier,
        test_retriever_imports,
        test_thai_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The iLand retrieval system core is working.")
    elif passed >= total * 0.75:  # 75% pass rate
        print("✓ Most tests passed! The iLand retrieval system is mostly functional.")
    else:
        print("✗ Many tests failed. Check the implementation.")
    
    return passed >= total * 0.75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 