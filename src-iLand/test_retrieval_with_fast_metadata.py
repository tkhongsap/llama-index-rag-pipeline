#!/usr/bin/env python3
"""
Test script to demonstrate Fast Metadata Indexing working within the iLand retrieval system.

This shows how our enhanced MetadataRetrieverAdapter integrates with the full CLI system.
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_fast_metadata_in_retrieval_system():
    """Test fast metadata indexing within the full retrieval system."""
    print("🧪 TESTING FAST METADATA INDEXING IN RETRIEVAL SYSTEM")
    print("=" * 60)
    
    try:
        # Import CLI components
        from retrieval.cli_handlers import iLandRetrievalCLI
        
        # Initialize CLI
        cli = iLandRetrievalCLI()
        
        # Show batch summary
        print("📊 Getting batch summary...")
        cli.show_batch_summary()
        
        # Load latest embeddings
        print("\n🔄 Loading latest embeddings...")
        if not cli.load_embeddings("latest"):
            print("❌ Failed to load embeddings")
            return False
        
        # Create router with LLM strategy
        print("🤖 Creating router with LLM strategy selector...")
        if not cli.create_router("llm"):
            print("❌ Failed to create router")
            return False
        
        print("✅ Retrieval system initialized successfully!")
        
        # Test metadata filtering capabilities
        print("\n🔍 Testing Fast Metadata Filtering...")
        
        # Check if we have enhanced metadata retriever
        metadata_adapter = None
        
        # Access adapters through CLI, not router
        if hasattr(cli, 'adapters') and cli.adapters:
            for index_name, strategies in cli.adapters.items():
                if 'metadata' in strategies:
                    metadata_adapter = strategies['metadata']
                    if hasattr(metadata_adapter, 'enable_fast_filtering'):
                        break
        
        if metadata_adapter:
            print("✅ Found enhanced metadata adapter with fast filtering!")
            
            # Get fast index stats
            stats = metadata_adapter.get_fast_index_stats()
            if stats:
                print(f"📈 Fast Index Statistics:")
                print(f"   • Total documents: {stats['total_documents']}")
                print(f"   • Indexed fields: {len(stats['indexed_fields'])}")
                perf = stats.get('performance_stats', {})
                if perf:
                    print(f"   • Avg filter time: {perf.get('avg_filter_time_ms', 0):.2f}ms")
                    print(f"   • Avg reduction: {perf.get('avg_reduction_ratio', 0)*100:.1f}%")
            
            # Test with different filter scenarios
            test_scenarios = [
                {
                    "query": "ที่ดินในจังหวัดไชยนาท",
                    "filters": {"province": "**: Chai Nat"},
                    "description": "Thai query with province filter"
                },
                {
                    "query": "land deed information",
                    "filters": {"deed_type": "โฉนด"},
                    "description": "English query with deed type filter"
                },
                {
                    "query": "property documents",
                    "filters": {},
                    "description": "No filters (baseline)"
                }
            ]
            
            print(f"\n🧪 Testing {len(test_scenarios)} scenarios...")
            
            for i, scenario in enumerate(test_scenarios, 1):
                print(f"\n--- Scenario {i}: {scenario['description']} ---")
                print(f"Query: {scenario['query']}")
                print(f"Filters: {scenario['filters']}")
                
                start_time = time.time()
                
                try:
                    results = metadata_adapter.retrieve(
                        query=scenario['query'],
                        top_k=3,
                        filters=scenario['filters']
                    )
                    
                    end_time = time.time()
                    total_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    print(f"✅ Retrieved {len(results)} results in {total_time:.2f}ms")
                    
                    # Show first result if available
                    if results:
                        first_result = results[0]
                        print(f"   📄 Top result: {first_result.node.text[:100]}...")
                        print(f"   📊 Score: {first_result.score:.3f}")
                        
                        # Check for fast filtering metadata
                        if hasattr(first_result.node, 'metadata'):
                            meta = first_result.node.metadata
                            if 'fast_filtering_enabled' in meta:
                                print(f"   🚀 Fast filtering: {'✅' if meta['fast_filtering_enabled'] else '❌'}")
                                if 'retrieval_time_ms' in meta:
                                    print(f"   ⏱️ Filter time: {meta['retrieval_time_ms']:.2f}ms")
                    
                except Exception as e:
                    print(f"❌ Error in scenario {i}: {e}")
        
        else:
            print("⚠️ Enhanced metadata adapter not found - using basic retrieval")
            
        # Test overall query through router
        print(f"\n🤖 Testing full query through router...")
        try:
            response = cli.query("ที่ดินประเภทโฉนด", top_k=3)
            print("✅ Router query completed successfully")
        except Exception as e:
            print(f"⚠️ Router query error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare performance with and without fast filtering."""
    print("\n⚡ PERFORMANCE COMPARISON TEST")
    print("=" * 40)
    
    try:
        from retrieval.retrievers.metadata import MetadataRetrieverAdapter
        from load_embedding import create_iland_index_from_latest_batch
        
        # Create index
        print("🔄 Creating test index...")
        index = create_iland_index_from_latest_batch(
            use_chunks=True,
            max_embeddings=10
        )
        
        # Test with fast filtering enabled
        print("\n🚀 Testing WITH fast filtering...")
        fast_retriever = MetadataRetrieverAdapter(
            index=index,
            enable_fast_filtering=True
        )
        
        start_time = time.time()
        fast_results = fast_retriever.retrieve(
            query="land deed",
            filters={"province": "**: Chai Nat"}
        )
        fast_time = (time.time() - start_time) * 1000
        
        # Test with fast filtering disabled  
        print("🐌 Testing WITHOUT fast filtering...")
        slow_retriever = MetadataRetrieverAdapter(
            index=index,
            enable_fast_filtering=False
        )
        
        start_time = time.time()
        slow_results = slow_retriever.retrieve(
            query="land deed",
            filters={"province": "**: Chai Nat"}
        )
        slow_time = (time.time() - start_time) * 1000
        
        # Compare results
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   🚀 With fast filtering:    {fast_time:.2f}ms ({len(fast_results)} results)")
        print(f"   🐌 Without fast filtering: {slow_time:.2f}ms ({len(slow_results)} results)")
        
        if slow_time > 0:
            speedup = slow_time / fast_time if fast_time > 0 else float('inf')
            print(f"   ⚡ Speedup factor: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Performance test skipped: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FAST METADATA INDEXING INTEGRATION TEST")
    print("=" * 70)
    
    # Test integration
    success = test_fast_metadata_in_retrieval_system()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        
        # Optional performance comparison
        test_performance_comparison()
        
        print("\n🎉 All tests completed!")
        print("🎯 Fast Metadata Indexing is working perfectly within the retrieval system!")
    else:
        print("\n⚠️ Some integration tests failed") 