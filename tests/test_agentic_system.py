#!/usr/bin/env python3
"""
Test script for the agentic retrieval system
Tests the end-to-end functionality with our updated index classifier
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_index_classifier():
    """Test the index classifier with our updated configuration."""
    print("🔧 Testing Index Classifier")
    print("=" * 50)
    
    try:
        from agentic_retriever.index_classifier import create_default_classifier
        
        # Create classifier
        classifier = create_default_classifier()
        
        # Test queries for our candidate profile data
        test_queries = [
            "What are the salary ranges mentioned in the profiles?",
            "Which candidates have programming skills?", 
            "What educational institutions are represented?",
            "Show me compensation details for HR professionals",
            "Find profiles with assessment scores above 80"
        ]
        
        print("Available indices:")
        for name, desc in classifier.get_available_indices().items():
            print(f"  - {name}: {desc[:60]}...")
        
        print(f"\nTesting {len(test_queries)} sample queries:")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            result = classifier.classify_query(query)
            
            print(f"\n{i}. Query: {query}")
            print(f"   → Selected: {result['selected_index']}")
            print(f"   → Confidence: {result['confidence']:.3f}")
            print(f"   → Method: {result['method']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing index classifier: {e}")
        return False


def test_embedding_availability():
    """Test if embeddings are available for retrieval."""
    print("\n🔧 Testing Embedding Availability")
    print("=" * 50)
    
    try:
        from load_embeddings import EmbeddingLoader
        
        # Check for embedding directories
        possible_paths = [
            Path("data/embedding"),
            Path("./data/embedding"),
            Path("../data/embedding")
        ]
        
        embedding_dir = None
        for path in possible_paths:
            if path.exists():
                embedding_dir = path
                break
        
        if not embedding_dir:
            print("❌ No embedding directory found")
            print("Please run the embedding pipeline first:")
            print("  python src/09_enhanced_batch_embeddings.py")
            return False
        
        print(f"✅ Found embedding directory: {embedding_dir}")
        
        # Load embedding loader
        loader = EmbeddingLoader(embedding_dir)
        latest_batch = loader.get_latest_batch()
        
        if latest_batch:
            print(f"✅ Latest batch found: {latest_batch}")
            
            # Try to load some embeddings
            try:
                full_data, _, _ = loader.load_embeddings_from_files(
                    latest_batch, "batch_1", "chunks"
                )
                
                if full_data:
                    print(f"✅ Successfully loaded {len(full_data)} embedded documents")
                    return True
                else:
                    print("❌ No embedding data found in batch")
                    return False
                    
            except Exception as e:
                print(f"❌ Error loading embeddings: {e}")
                return False
        else:
            print("❌ No embedding batches found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking embeddings: {e}")
        return False


def test_simple_query():
    """Test a simple end-to-end query."""
    print("\n🔧 Testing Simple End-to-End Query")
    print("=" * 50)
    
    try:
        # Import required modules
        from agentic_retriever.cli import query_agentic_retriever
        
        # Test query
        test_query = "What are the salary ranges mentioned in the profiles?"
        
        print(f"Testing query: {test_query}")
        print("-" * 30)
        
        # Execute query
        result = query_agentic_retriever(test_query, top_k=3)
        
        if result["error"]:
            print(f"❌ Query failed: {result['error']}")
            return False
        else:
            print("✅ Query executed successfully!")
            print(f"\nResponse preview: {str(result['response'])[:200]}...")
            
            metadata = result["metadata"]
            print(f"\nMetadata:")
            print(f"  - Index: {metadata.get('index', 'unknown')}")
            print(f"  - Strategy: {metadata.get('strategy', 'unknown')}")
            print(f"  - Latency: {metadata.get('total_time_ms', 0)} ms")
            print(f"  - Sources: {metadata.get('num_sources', 'unknown')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 AGENTIC RETRIEVAL SYSTEM - END-TO-END TEST")
    print("=" * 60)
    print("Testing our updated index classifier and routing system")
    
    # Run tests
    test_results = []
    
    # Test 1: Index Classifier
    test_results.append(test_index_classifier())
    
    # Test 2: Embedding Availability  
    test_results.append(test_embedding_availability())
    
    # Test 3: End-to-end query (only if embeddings available)
    if test_results[-1]:  # If embeddings are available
        test_results.append(test_simple_query())
    else:
        print("\n⚠️  Skipping end-to-end test (no embeddings available)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Agentic retriever is working correctly.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
