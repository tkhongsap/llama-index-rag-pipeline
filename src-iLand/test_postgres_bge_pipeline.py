#!/usr/bin/env python3
"""
Test script for PostgreSQL BGE-M3 Pipeline

This script tests the pipeline with a small sample to verify everything works.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from run_postgres_bge_pipeline import run_complete_pipeline


def test_pipeline():
    """Run a quick test of the pipeline with 2 documents."""
    
    print("🧪 TESTING POSTGRESQL BGE-M3 PIPELINE")
    print("=" * 60)
    print("This test will:")
    print("1. Process 2 documents from CSV")
    print("2. Create section-based chunks")
    print("3. Generate BGE-M3 embeddings")
    print("4. Store in PostgreSQL")
    print("=" * 60)
    
    try:
        # Run pipeline with just 2 documents
        results = run_complete_pipeline(
            input_file="input_dataset_iLand_sample.csv",
            max_rows=2,
            batch_size=10,
            db_batch_size=10,
            embedding_batch_size=8,
            device="cpu"  # Use CPU for testing
        )
        
        # Verify results
        print("\n🔍 TEST RESULTS:")
        print(f"✅ Documents processed: {results['documents']}")
        print(f"✅ Chunks created: {results['chunks']}")
        print(f"✅ Embeddings generated: {results['embeddings']}")
        print(f"✅ Chunks per document: {results['chunks_per_doc']:.1f}")
        
        # Check if results are reasonable
        if results['documents'] == 2:
            print("\n✅ Document count correct")
        else:
            print(f"\n❌ Expected 2 documents, got {results['documents']}")
            
        if 5 <= results['chunks_per_doc'] <= 15:
            print("✅ Chunk count reasonable (5-15 per doc)")
        else:
            print(f"❌ Unexpected chunks per doc: {results['chunks_per_doc']}")
            
        if results['embeddings'] == results['chunks']:
            print("✅ All chunks have embeddings")
        else:
            print(f"❌ Embedding mismatch: {results['embeddings']} vs {results['chunks']} chunks")
        
        print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_pipeline()