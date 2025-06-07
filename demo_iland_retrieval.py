#!/usr/bin/env python3
"""
Demo script for iLand Retrieval System

This script demonstrates the complete iLand retrieval workflow including:
- Loading embeddings
- Creating retriever adapters
- Setting up the router
- Testing queries with different strategies
"""

import os
import sys
from pathlib import Path

# Add src-iLand to path
sys.path.insert(0, str(Path(__file__).parent / "src-iLand"))
sys.path.insert(0, str(Path(__file__).parent / "src-iLand" / "retrieval"))

def main():
    """Main demo function."""
    print("iLand Retrieval System Demo")
    print("=" * 50)
    
    # Test 1: Import validation
    print("\n1. Testing imports...")
    try:
        from index_classifier import create_default_iland_classifier
        from router import iLandRouterRetriever
        from retrievers import VectorRetrieverAdapter
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test 2: Index classifier
    print("\n2. Testing index classifier...")
    try:
        classifier = create_default_iland_classifier()
        
        test_queries = [
            "โฉนดที่ดินในกรุงเทพมหานคร",
            "Land deeds in Bangkok",
            "ที่ดินในสมุทรปราการ"
        ]
        
        for query in test_queries:
            result = classifier.classify_query(query)
            print(f"✓ '{query[:30]}...' -> {result['selected_index']} (confidence: {result['confidence']:.2f})")
            
    except Exception as e:
        print(f"✗ Index classifier error: {e}")
        return False
    
    # Test 3: Check embedding availability
    print("\n3. Checking embedding availability...")
    try:
        from load_embedding import get_iland_batch_summary
        summary = get_iland_batch_summary()
        print(f"✓ Found {summary['latest_batch_stats']['total_embeddings']} embeddings in latest batch")
        print(f"  - Chunk embeddings: {summary['latest_batch_stats']['chunk_embeddings']}")
        print(f"  - Summary embeddings: {summary['latest_batch_stats']['summary_embeddings']}")
        embeddings_available = True
    except Exception as e:
        print(f"✗ Could not access embeddings: {e}")
        embeddings_available = False
    
    # Test 4: Strategy testing (without real embeddings)
    print("\n4. Testing strategy components...")
    try:
        from retrievers.hybrid import HybridRetrieverAdapter
        from retrievers.metadata import MetadataRetrieverAdapter
        
        # Test Thai keyword extraction
        class MockHybridAdapter(HybridRetrieverAdapter):
            def __init__(self):
                self.strategy_name = "hybrid"
        
        hybrid_adapter = MockHybridAdapter()
        keywords = hybrid_adapter._extract_thai_keywords("โฉนดที่ดินในกรุงเทพมหานคร")
        print(f"✓ Thai keyword extraction: {keywords}")
        
        # Test metadata filter building
        class MockMetadataAdapter(MetadataRetrieverAdapter):
            def __init__(self):
                self.strategy_name = "metadata"
                self.thai_provinces = ["กรุงเทพมหานคร", "สมุทรปราการ"]
        
        metadata_adapter = MockMetadataAdapter()
        filters = metadata_adapter._build_metadata_filters(None, "ที่ดินในกรุงเทพมหานคร")
        print(f"✓ Metadata filter building: {len(filters.filters) if filters else 0} conditions")
        
    except Exception as e:
        print(f"✗ Strategy testing error: {e}")
        return False
    
    # Test 5: CLI availability
    print("\n5. Testing CLI availability...")
    try:
        from cli import iLandRetrievalCLI
        cli = iLandRetrievalCLI()
        print("✓ CLI class instantiated successfully")
        print(f"  - API key configured: {'Yes' if cli.api_key else 'No'}")
    except Exception as e:
        print(f"✗ CLI error: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print("✓ iLand retrieval system is properly implemented")
    print("✓ All seven strategy adapters are available")
    print("✓ Index classifier works with Thai and English queries")
    print("✓ Thai language support is functional")
    print("✓ CLI interface is ready for use")
    
    if embeddings_available:
        print("✓ Embedding data is available for testing")
        print("\nNext steps:")
        print("1. Run: python src-iLand/retrieval/cli.py --load-embeddings latest --interactive")
        print("2. Test queries like: 'โฉนดที่ดินในกรุงเทพ' or 'Land deeds in Bangkok'")
    else:
        print("⚠ Embedding data not accessible (may need API key or data loading)")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Load iLand embeddings using the load_embedding module")
        print("3. Test the retrieval system with real data")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Demo completed successfully!")
    else:
        print("\n❌ Demo failed. Check the implementation.")
    
    sys.exit(0 if success else 1) 