"""
Phase 3 Compatibility Test for Sales Promotion Embeddings

Validates that Phase 2 embeddings can be loaded and used for Phase 3 retrieval.
Tests:
- Loading PKL embeddings
- Verifying embedding dimensions (3072)
- Metadata preservation (29 fields)
- Thai UTF-8 encoding
- Vector operations readiness
"""

import pickle
import numpy as np
import json
from pathlib import Path
import sys


def test_load_pkl_embeddings():
    """Test loading PKL format embeddings."""
    print("\n" + "="*80)
    print("TEST 1: Load PKL Embeddings")
    print("="*80)
    
    embedding_dir = Path("data/embedding/embeddings_sales_20251026_172927")
    pkl_file = embedding_dir / "batch_1/indexnodes/batch_1_indexnodes_full.pkl"
    
    if not pkl_file.exists():
        print(f"❌ PKL file not found: {pkl_file}")
        return False
    
    try:
        with open(pkl_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"✅ Successfully loaded PKL file")
        print(f"✅ Number of embeddings: {len(embeddings)}")
        print(f"✅ Type: {type(embeddings)}")
        
        return embeddings
    
    except Exception as e:
        print(f"❌ Error loading PKL: {e}")
        return False


def test_embedding_dimensions(embeddings):
    """Test embedding vector dimensions."""
    print("\n" + "="*80)
    print("TEST 2: Verify Embedding Dimensions")
    print("="*80)
    
    if not embeddings:
        print("❌ No embeddings to test")
        return False
    
    sample_emb = embeddings[0]
    vector = sample_emb['embedding_vector']
    dim = len(vector)
    
    print(f"✅ Embedding dimension: {dim}")
    
    if dim == 3072:
        print(f"✅ Correct dimension for text-embedding-3-large")
        return True
    else:
        print(f"❌ Expected 3072, got {dim}")
        return False


def test_metadata_preservation(embeddings):
    """Test metadata field preservation."""
    print("\n" + "="*80)
    print("TEST 3: Verify Metadata Preservation")
    print("="*80)
    
    if not embeddings:
        print("❌ No embeddings to test")
        return False
    
    sample_emb = embeddings[0]
    metadata = sample_emb.get('metadata', {})
    
    print(f"✅ Metadata fields count: {len(metadata)}")
    print(f"✅ Metadata fields: {list(metadata.keys())[:10]}...")
    
    # Check for key fields
    required_fields = [
        'document_id', 'customer_type', 'product_category',
        'discount_percentage', 'is_active', 'batch_number'
    ]
    
    missing = [f for f in required_fields if f not in metadata]
    
    if missing:
        print(f"❌ Missing required fields: {missing}")
        return False
    
    print(f"✅ All required metadata fields present")
    return True


def test_thai_encoding(embeddings):
    """Test Thai UTF-8 encoding preservation."""
    print("\n" + "="*80)
    print("TEST 4: Verify Thai UTF-8 Encoding")
    print("="*80)
    
    if not embeddings:
        print("❌ No embeddings to test")
        return False
    
    sample_emb = embeddings[0]
    
    # Check Thai text in various fields
    text = sample_emb.get('text', '')
    customer_type = sample_emb['metadata'].get('customer_type', '')
    product_name = sample_emb['metadata'].get('product_name', '')
    
    thai_chars_found = any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text + customer_type + product_name)
    
    if thai_chars_found:
        print(f"✅ Thai characters found and properly encoded")
        print(f"   Sample customer_type: {customer_type}")
        print(f"   Sample product_name: {product_name[:50]}...")
        return True
    else:
        print(f"❌ No Thai characters found - encoding may be corrupted")
        return False


def test_numpy_compatibility():
    """Test NPY format compatibility."""
    print("\n" + "="*80)
    print("TEST 5: Verify NPY Format Compatibility")
    print("="*80)
    
    embedding_dir = Path("data/embedding/embeddings_sales_20251026_172927")
    npy_file = embedding_dir / "batch_1/indexnodes/batch_1_indexnodes_vectors.npy"
    
    if not npy_file.exists():
        print(f"❌ NPY file not found: {npy_file}")
        return False
    
    try:
        vectors = np.load(npy_file)
        
        print(f"✅ Successfully loaded NPY file")
        print(f"✅ Vector array shape: {vectors.shape}")
        print(f"✅ Data type: {vectors.dtype}")
        
        # Test vector operations
        norm = np.linalg.norm(vectors[0])
        print(f"✅ Vector norm (sample): {norm:.4f}")
        
        if vectors.shape[1] == 3072:
            print(f"✅ Correct vector dimensions for Phase 3 retrieval")
            return True
        else:
            print(f"❌ Expected dimension 3072, got {vectors.shape[1]}")
            return False
    
    except Exception as e:
        print(f"❌ Error loading NPY: {e}")
        return False


def test_json_format():
    """Test JSON format readability."""
    print("\n" + "="*80)
    print("TEST 6: Verify JSON Format Readability")
    print("="*80)
    
    embedding_dir = Path("data/embedding/embeddings_sales_20251026_172927")
    json_file = embedding_dir / "batch_1/indexnodes/batch_1_indexnodes_metadata.json"
    
    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        return False
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded JSON file")
        print(f"✅ Number of entries: {len(data)}")
        
        # Check sample entry
        sample = data[0]
        print(f"✅ Sample entry keys: {list(sample.keys())}")
        print(f"✅ Has embedding preview: {'embedding_preview' in sample}")
        print(f"✅ Embedding preview dims: {len(sample.get('embedding_preview', []))}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False


def test_combined_statistics():
    """Test combined statistics file."""
    print("\n" + "="*80)
    print("TEST 7: Verify Combined Statistics")
    print("="*80)
    
    embedding_dir = Path("data/embedding/embeddings_sales_20251026_172927")
    stats_file = embedding_dir / "combined_statistics.json"
    
    if not stats_file.exists():
        print(f"❌ Statistics file not found: {stats_file}")
        return False
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print(f"✅ Successfully loaded combined statistics")
        print(f"✅ Embedding model: {stats['embedding_model']}")
        print(f"✅ LLM model: {stats['llm_model']}")
        print(f"✅ Total batches: {stats['total_batches']}")
        print(f"✅ Total documents: {stats['grand_totals']['total_documents']}")
        print(f"✅ Total embeddings: {stats['grand_totals']['total_embeddings']}")
        print(f"✅ Metadata fields: {stats['metadata_analysis']['total_unique_metadata_fields']}")
        
        # Verify expected totals
        if stats['grand_totals']['total_documents'] == 71:
            print(f"✅ Correct number of documents (71)")
        else:
            print(f"❌ Expected 71 documents, got {stats['grand_totals']['total_documents']}")
            return False
        
        if stats['grand_totals']['total_embeddings'] == 142:
            print(f"✅ Correct number of embeddings (142)")
        else:
            print(f"❌ Expected 142 embeddings, got {stats['grand_totals']['total_embeddings']}")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading statistics: {e}")
        return False


def main():
    """Run all Phase 3 compatibility tests."""
    print("\n" + "="*80)
    print("PHASE 3 COMPATIBILITY TEST SUITE")
    print("Sales Promotion Embeddings - Generated with GPT-5 & text-embedding-3-large")
    print("="*80)
    
    results = {}
    
    # Test 1: Load PKL
    embeddings = test_load_pkl_embeddings()
    results['pkl_load'] = bool(embeddings)
    
    if embeddings:
        # Test 2: Dimensions
        results['dimensions'] = test_embedding_dimensions(embeddings)
        
        # Test 3: Metadata
        results['metadata'] = test_metadata_preservation(embeddings)
        
        # Test 4: Thai encoding
        results['thai_encoding'] = test_thai_encoding(embeddings)
    
    # Test 5: NPY compatibility
    results['npy_load'] = test_numpy_compatibility()
    
    # Test 6: JSON format
    results['json_load'] = test_json_format()
    
    # Test 7: Statistics
    results['statistics'] = test_combined_statistics()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed = sum(results.values())
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*80)
    print(f"OVERALL: {passed}/{total_tests} tests passed")
    
    if passed == total_tests:
        print("✅ ALL TESTS PASSED - Ready for Phase 3!")
    else:
        print("❌ SOME TESTS FAILED - Please review errors above")
    
    print("="*80)
    
    return passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

