#!/usr/bin/env python3
"""
Simple test script to verify the agentic routing logic
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_index_classifier():
    """Test the index classifier to see what indices it recognizes."""
    try:
        from agentic_retriever.index_classifier import create_default_classifier, DEFAULT_INDICES
        
        print("🔍 Testing Index Classifier")
        print("=" * 50)
        
        # Show available indices
        print("📚 Available Indices:")
        for index_name, description in DEFAULT_INDICES.items():
            print(f"  • {index_name}: {description}")
        
        # Create classifier
        classifier = create_default_classifier()
        
        # Test queries
        test_queries = [
            "What age groups are represented in the candidate profiles?",
            "What are the most common educational backgrounds?",
            "What is the salary range for different positions?",
            "Which provinces are candidates located in?"
        ]
        
        print(f"\n🧪 Testing Query Classification:")
        for query in test_queries:
            result = classifier.classify_query(query)
            print(f"\nQuery: {query}")
            print(f"  → Index: {result['selected_index']}")
            print(f"  → Confidence: {result.get('confidence', 0):.3f}")
            print(f"  → Method: {result.get('method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing index classifier: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_router_creation():
    """Test creating the agentic router."""
    try:
        from agentic_retriever.cli import create_agentic_router
        
        print(f"\n🔧 Testing Router Creation")
        print("=" * 50)
        
        router = create_agentic_router()
        
        if router:
            print("✅ Router created successfully!")
            
            # Get available retrievers
            available = router.get_available_retrievers()
            print(f"\n📊 Available Retrievers:")
            for index_name, strategies in available.items():
                print(f"  • {index_name}: {strategies}")
            
            return True
        else:
            print("❌ Failed to create router")
            return False
            
    except Exception as e:
        print(f"❌ Error creating router: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_query():
    """Test a simple query through the system."""
    try:
        from agentic_retriever.cli import query_agentic_retriever
        
        print(f"\n🚀 Testing Simple Query")
        print("=" * 50)
        
        test_query = "What age groups are represented in the candidate profiles?"
        print(f"Query: {test_query}")
        
        result = query_agentic_retriever(test_query, top_k=3)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return False
        
        print(f"✅ Query successful!")
        
        # Show routing information
        metadata = result.get("metadata", {})
        print(f"\n📊 Routing Information:")
        print(f"  • Index: {metadata.get('index', 'unknown')}")
        print(f"  • Strategy: {metadata.get('strategy', 'unknown')}")
        print(f"  • Index Confidence: {metadata.get('index_confidence', 'N/A')}")
        print(f"  • Strategy Confidence: {metadata.get('strategy_confidence', 'N/A')}")
        print(f"  • Sources: {metadata.get('num_sources', 0)}")
        print(f"  • Latency: {metadata.get('total_time_ms', 0)} ms")
        
        # Show response preview
        response = result.get("response", "")
        if response:
            preview = response[:200] + "..." if len(response) > 200 else response
            print(f"\n📝 Response Preview:")
            print(f"  {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing query: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 AGENTIC ROUTING LOGIC TEST")
    print("=" * 60)
    
    # Test 1: Index Classifier
    classifier_success = test_index_classifier()
    
    # Test 2: Router Creation
    router_success = test_router_creation()
    
    # Test 3: Simple Query
    query_success = test_simple_query()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Index Classifier: {'SUCCESS' if classifier_success else 'FAILED'}")
    print(f"✅ Router Creation: {'SUCCESS' if router_success else 'FAILED'}")
    print(f"✅ Simple Query: {'SUCCESS' if query_success else 'FAILED'}")
    
    if all([classifier_success, router_success, query_success]):
        print(f"\n🎉 ALL TESTS PASSED! The agentic routing logic is working correctly.")
        print(f"\nNow you should see:")
        print(f"  • Multiple indices (not just 'general_docs')")
        print(f"  • Proper index classification based on query content")
        print(f"  • Strategy selection (currently 'vector')")
    else:
        print(f"\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 