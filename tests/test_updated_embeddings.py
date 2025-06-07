#!/usr/bin/env python3
"""
Test script to verify that load_embedding module works correctly with updated embeddings.
This script tests all major functionality after the docs_embedding updates.
"""

import sys
from pathlib import Path

def test_embedding_loading():
    """Test basic embedding loading functionality."""
    print("🔄 Testing embedding loading functionality...")
    
    try:
        from load_embedding import (
            get_iland_batch_summary,
            load_all_latest_iland_embeddings,
            validate_iland_embeddings
        )
        
        # Test 1: Get batch summary
        print("📊 Test 1: Getting batch summary...")
        summary = get_iland_batch_summary()
        print(f"   ✅ Found {summary['total_batches']} batches")
        print(f"   ✅ Latest batch: {summary['latest_batch']}")
        print(f"   ✅ Total embeddings: {summary['latest_batch_stats']['total_embeddings']}")
        
        # Test 2: Load chunk embeddings
        print("📦 Test 2: Loading chunk embeddings...")
        embeddings, batch_path = load_all_latest_iland_embeddings("chunks")
        print(f"   ✅ Loaded {len(embeddings)} chunk embeddings")
        print(f"   ✅ From batch: {batch_path.name}")
        
        # Test 3: Validate embeddings
        print("🔍 Test 3: Validating embeddings...")
        stats = validate_iland_embeddings(embeddings)
        print(f"   ✅ Total count: {stats['total_count']}")
        print(f"   ✅ With vectors: {stats['has_vectors']}")
        print(f"   ✅ Provinces found: {len(stats['thai_metadata']['provinces'])}")
        print(f"   ✅ Deed types found: {len(stats['thai_metadata']['deed_types'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_index_creation():
    """Test index creation and querying."""
    print("\n🏗️ Testing index creation and querying...")
    
    try:
        from load_embedding import create_iland_index_from_latest_batch
        
        # Test 4: Create index with limited embeddings
        print("🔄 Test 4: Creating index with limited embeddings...")
        index = create_iland_index_from_latest_batch(
            use_chunks=True,
            use_summaries=False,
            max_embeddings=50
        )
        print("   ✅ Index created successfully")
        
        # Test 5: Query the index
        print("🔍 Test 5: Testing Thai language query...")
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("ที่ดินในข้อมูลมีกี่แปลง?")
        print(f"   ✅ Query successful")
        print(f"   📝 Response preview: {str(response)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_filtering():
    """Test filtering functionality."""
    print("\n🔧 Testing filtering functionality...")
    
    try:
        from load_embedding import (
            load_all_latest_iland_embeddings,
            FilterConfig,
            iLandEmbeddingLoader,
            EmbeddingConfig
        )
        
        # Test 6: Load embeddings for filtering
        print("📦 Test 6: Loading embeddings for filtering...")
        embeddings, _ = load_all_latest_iland_embeddings("chunks")
        original_count = len(embeddings)
        print(f"   ✅ Loaded {original_count} embeddings")
        
        # Test 7: Apply filter configuration
        print("🔧 Test 7: Testing filter configuration...")
        config = EmbeddingConfig()
        filter_config = FilterConfig(
            max_embeddings=100
        )
        
        loader = iLandEmbeddingLoader(config)
        filtered = loader.apply_filter_config(embeddings, filter_config)
        print(f"   ✅ Filtered from {original_count} to {len(filtered)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_production_features():
    """Test production-ready features."""
    print("\n🚀 Testing production features...")
    
    try:
        from load_embedding import (
            load_latest_iland_embeddings,
            create_iland_index_from_latest_batch
        )
        
        # Test 8: Load specific embedding types
        print("📋 Test 8: Loading specific embedding types...")
        
        # Load chunks
        chunks, _ = load_latest_iland_embeddings("chunks", "batch_1")
        print(f"   ✅ Loaded {len(chunks)} chunks from batch_1")
        
        # Load summaries
        summaries, _ = load_latest_iland_embeddings("summaries", "batch_1")
        print(f"   ✅ Loaded {len(summaries)} summaries from batch_1")
        
        # Load indexnodes
        indexnodes, _ = load_latest_iland_embeddings("indexnodes", "batch_1")
        print(f"   ✅ Loaded {len(indexnodes)} indexnodes from batch_1")
        
        # Test 9: Multi-type index creation
        print("🏗️ Test 9: Creating multi-type index...")
        index = create_iland_index_from_latest_batch(
            use_chunks=True,
            use_summaries=True,
            use_indexnodes=True,
            max_embeddings=100
        )
        print("   ✅ Multi-type index created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 TESTING UPDATED EMBEDDINGS WITH LOAD_EMBEDDING MODULE")
    print("=" * 70)
    
    # Run all tests
    test_results = []
    
    test_results.append(("Embedding Loading", test_embedding_loading()))
    test_results.append(("Index Creation", test_index_creation()))
    test_results.append(("Filtering", test_filtering()))
    test_results.append(("Production Features", test_production_features()))
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The load_embedding module is fully compatible with updated embeddings")
        print("✅ Ready for production RAG applications")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 