#!/usr/bin/env python3
"""
Test script to verify we can load the user's actual embedding data
and run the agentic retrieval CLI with it.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from load_embeddings import EmbeddingLoader, IndexReconstructor

def test_embedding_loading():
    """Test loading embeddings from the data/embedding directory."""
    print("🔄 Testing Embedding Data Loading")
    print("=" * 50)
    
    try:
        # Initialize loader with correct path
        embedding_dir = Path("data/embedding")
        print(f"📁 Looking for embeddings in: {embedding_dir.absolute()}")
        
        if not embedding_dir.exists():
            print(f"❌ Embedding directory not found: {embedding_dir.absolute()}")
            return False
            
        loader = EmbeddingLoader(embedding_dir)
        
        # Show available batches
        batches = loader.get_available_batches()
        print(f"\n📊 Found {len(batches)} embedding batches:")
        for batch in batches:
            print(f"   • {batch.name}")
        
        if not batches:
            print("❌ No embedding batches found!")
            return False
            
        # Use latest batch
        latest_batch = loader.get_latest_batch()
        print(f"\n🎯 Using latest batch: {latest_batch.name}")
        
        # Load batch statistics
        stats = loader.load_batch_statistics(latest_batch)
        if stats:
            print(f"\n📈 Batch Statistics:")
            print(f"   • Total batches: {stats.get('total_batches', 'N/A')}")
            if 'grand_totals' in stats:
                totals = stats['grand_totals']
                print(f"   • Total embeddings: {totals.get('total_embeddings', 'N/A')}")
                print(f"   • Chunks: {totals.get('chunk_embeddings', 'N/A')}")
                print(f"   • Summaries: {totals.get('summary_embeddings', 'N/A')}")
                print(f"   • IndexNodes: {totals.get('indexnode_embeddings', 'N/A')}")
        
        # Try to load embeddings from different sub-batches and types
        print(f"\n🔍 Searching for available embeddings...")
        
        # Check what sub-batches exist
        sub_batches = sorted([
            d.name for d in latest_batch.iterdir() 
            if d.is_dir() and d.name.startswith("batch_")
        ])
        
        print(f"   • Found sub-batches: {sub_batches}")
        
        # Try to find any embeddings
        found_embeddings = []
        for sub_batch in sub_batches:
            for emb_type in ["chunks", "summaries", "indexnodes"]:
                try:
                    full_data, vectors, metadata = loader.load_embeddings_from_files(
                        latest_batch, sub_batch, emb_type
                    )
                    if full_data:
                        found_embeddings.append((sub_batch, emb_type, len(full_data)))
                        print(f"   ✅ {sub_batch}/{emb_type}: {len(full_data)} embeddings")
                except Exception as e:
                    print(f"   ⚠️ {sub_batch}/{emb_type}: {str(e)}")
        
        if not found_embeddings:
            print("❌ No embeddings found in any sub-batch!")
            return False
            
        # Use the first available embeddings for testing
        sub_batch, emb_type, count = found_embeddings[0]
        print(f"\n🧪 Testing with {sub_batch}/{emb_type} ({count} embeddings)")
        
        # Load the embeddings
        embeddings, _, _ = loader.load_embeddings_from_files(
            latest_batch, sub_batch, emb_type
        )
        
        print(f"✅ Successfully loaded {len(embeddings)} embeddings!")
        
        # Show sample embedding structure
        if embeddings:
            sample = embeddings[0]
            print(f"\n📋 Sample embedding structure:")
            print(f"   • Keys: {list(sample.keys())}")
            print(f"   • Type: {sample.get('type', 'unknown')}")
            print(f"   • Has text: {'text' in sample}")
            print(f"   • Has vector: {'embedding_vector' in sample}")
            if 'text' in sample:
                text_preview = sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text']
                print(f"   • Text preview: {text_preview}")
        
        return True, embeddings, sub_batch, emb_type
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_agentic_cli_with_data(embeddings, sub_batch, emb_type):
    """Test the agentic CLI with the loaded embeddings."""
    print(f"\n🚀 Testing Agentic CLI with {sub_batch}/{emb_type}")
    print("=" * 50)
    
    try:
        # Import agentic retriever components
        from agentic_retriever.retrievers.vector import VectorRetrieverAdapter
        from agentic_retriever.router import RouterRetriever
        
        print(f"🔧 Creating VectorRetrieverAdapter from {len(embeddings)} embeddings...")
        
        # Create vector retriever adapter
        vector_adapter = VectorRetrieverAdapter.from_embeddings(embeddings[:10])  # Use first 10 for testing
        
        print(f"🧠 Creating RouterRetriever...")
        
        # Create router with the adapter
        retrievers = {
            "general_docs": {
                "vector": vector_adapter
            }
        }
        
        router = RouterRetriever.from_retrievers(
            retrievers=retrievers,
            strategy_selector="llm"
        )
        
        print(f"✅ Router created successfully!")
        
        # Test a simple query
        print(f"\n🧪 Testing query retrieval...")
        test_query = "What are the main topics discussed in the documents?"
        
        nodes = router.retrieve(test_query)
        
        print(f"✅ Retrieved {len(nodes)} nodes!")
        
        if nodes:
            print(f"\n📄 Sample retrieved content:")
            sample_node = nodes[0]
            text_preview = sample_node.node.text[:200] + "..." if len(sample_node.node.text) > 200 else sample_node.node.text
            print(f"   • Score: {sample_node.score:.4f}")
            print(f"   • Text: {text_preview}")
            print(f"   • Metadata: {sample_node.node.metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing agentic CLI: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 AGENTIC RETRIEVAL - REAL DATA TEST")
    print("=" * 60)
    print("Testing the agentic retrieval system with your actual embedded data")
    
    # Test 1: Load embeddings
    result = test_embedding_loading()
    if not result:
        print("\n❌ Failed to load embeddings. Cannot proceed with CLI test.")
        return
        
    success, embeddings, sub_batch, emb_type = result
    if not success:
        print("\n❌ Failed to load embeddings. Cannot proceed with CLI test.")
        return
    
    # Test 2: Test agentic CLI
    cli_success = test_agentic_cli_with_data(embeddings, sub_batch, emb_type)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Embedding Loading: {'SUCCESS' if success else 'FAILED'}")
    print(f"✅ Agentic CLI Test: {'SUCCESS' if cli_success else 'FAILED'}")
    
    if success and cli_success:
        print(f"\n🎉 SUCCESS! Your agentic retrieval system is working with real data!")
        print(f"\n📚 Next steps:")
        print(f"   1. Run: cd src && python -m agentic_retriever.cli -q 'your question'")
        print(f"   2. Monitor: python -m agentic_retriever.stats")
        print(f"   3. Evaluate: cd .. && python tests/eval_agentic.py")
    else:
        print(f"\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 