#!/usr/bin/env python3
"""
Simple test script for hybrid search functionality
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    print("Testing imports...")
    from load_embeddings import create_index_from_latest_batch
    from load_embeddings import EmbeddingLoader
    print("✅ Imports successful")
    
    print("\nTesting index creation...")
    index = create_index_from_latest_batch(use_chunks=True)
    print("✅ Index created successfully")
    
    print("\nTesting basic query...")
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("education background")
    print(f"✅ Basic query successful: {str(response)[:100]}...")
    
    print("\nTesting hybrid search import...")
    # Import from the 16_hybrid_search module
    import importlib.util
    spec = importlib.util.spec_from_file_location("hybrid_search", "src/16_hybrid_search.py")
    hybrid_search = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_search)
    print("✅ Hybrid search import successful")
    
    print("\nTesting hybrid search engine creation...")
    hybrid_engine = hybrid_search.HybridSearchEngine(index, similarity_top_k=3)
    print("✅ Hybrid search engine created")
    
    print("\nTesting hybrid query...")
    result = hybrid_engine.query("education", top_k=3, show_details=False)
    print(f"✅ Hybrid query successful: {result['response'][:100]}...")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 