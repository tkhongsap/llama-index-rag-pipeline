#!/usr/bin/env python3
"""
Debug script to check embedding types
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Import LlamaIndex utilities
from load_embeddings import EmbeddingLoader

def check_embedding_types():
    """Check what types of embeddings are available."""
    print("🔍 CHECKING EMBEDDING TYPES")
    print("=" * 50)
    
    try:
        # Load embeddings
        print("1. Loading embeddings from data/embedding...")
        loader = EmbeddingLoader(Path("data/embedding"))
        latest_batch = loader.get_latest_batch()
        
        if not latest_batch:
            print("❌ No embedding batches found!")
            return
            
        print(f"✅ Found latest batch: {latest_batch}")
        
        # Load all embeddings from the batch
        all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        print(f"\n2. Sub-batches in {latest_batch}:")
        for sub_batch, emb_types in all_embeddings.items():
            print(f"   📁 {sub_batch}:")
            for emb_type, embeddings in emb_types.items():
                print(f"      • {emb_type}: {len(embeddings)} embeddings")
                
                # Show first few embeddings with their 'type' field
                if embeddings:
                    print(f"        Sample embedding types:")
                    for i, emb in enumerate(embeddings[:3]):
                        emb_type_field = emb.get('type', 'NO_TYPE_FIELD')
                        print(f"          {i+1}. type='{emb_type_field}'")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_embedding_types() 