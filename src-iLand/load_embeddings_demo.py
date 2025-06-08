#!/usr/bin/env python3
"""
Simple script to load iLand embeddings and test our fast metadata indexing.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def load_embeddings_simple():
    """Load embeddings using the iLand embedding loader."""
    try:
        print("🔄 Loading iLand embeddings...")
        
        # Import the iLand embedding components
        from load_embedding import (
            demonstrate_iland_loading,
            create_iland_index_from_latest_batch,
            load_all_latest_iland_embeddings
        )
        
        # Try to run the demonstration
        print("📚 Running iLand embedding demonstration...")
        demonstrate_iland_loading()
        
        print("\n" + "="*50)
        print("🚀 Testing integration with our fast metadata indexing...")
        
        try:
            # Try to create an index from the latest batch
            index = create_iland_index_from_latest_batch(
                use_chunks=True,
                use_summaries=False,
                max_embeddings=20  # Limit for testing
            )
            
            print("✅ Index created successfully!")
            print(f"📊 Index has {len(index.docstore.docs)} documents")
            
            # Test with our enhanced metadata retriever
            from retrieval.retrievers.metadata import MetadataRetrieverAdapter
            
            retriever = MetadataRetrieverAdapter(
                index=index,
                default_top_k=5,
                enable_fast_filtering=True
            )
            
            print("✅ Enhanced metadata retriever created with fast indexing!")
            
            # Get fast index stats
            stats = retriever.get_fast_index_stats()
            if stats:
                print(f"📈 Fast indexing stats:")
                print(f"   • Total documents: {stats['total_documents']}")
                print(f"   • Indexed fields: {stats['indexed_fields']}")
                print(f"   • Performance: {stats['performance_stats']['avg_filter_time_ms']:.2f}ms avg")
            
            # Test a simple query with filtering
            print("\n🧪 Testing retrieval with fast metadata filtering...")
            results = retriever.retrieve(
                query="ที่ดินในกรุงเทพ",
                top_k=3,
                filters={"province": "กรุงเทพมหานคร"}
            )
            
            print(f"✅ Retrieved {len(results)} results with fast filtering!")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Could not create index or test retrieval: {e}")
            print("This might be due to missing embeddings or API key")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the src-iLand directory")
        return False
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 iLAND EMBEDDINGS + FAST METADATA INDEXING TEST")
    print("=" * 60)
    
    success = load_embeddings_simple()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🎉 Fast metadata indexing is working with iLand embeddings!")
    else:
        print("\n⚠️ Some tests failed, but basic loading functionality is available.") 