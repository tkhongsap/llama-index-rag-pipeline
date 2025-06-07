#!/usr/bin/env python3
"""
Simple test script to debug the iLand demo issues
"""

import os
import sys
from pathlib import Path

print("🔍 Starting iLand Demo Debug Test")
print("=" * 50)

try:
    # Test basic imports
    print("1. Testing basic Python imports...")
    import time
    from typing import Dict, List, Any, Optional
    from dotenv import load_dotenv
    print("   ✅ Basic imports successful")
    
    # Load environment
    print("2. Loading environment variables...")
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"   ✅ OpenAI API key found (length: {len(api_key)})")
    else:
        print("   ⚠️ No OpenAI API key found")
    
    # Test src-iLand path
    print("3. Testing src-iLand path...")
    src_iland_path = Path(__file__).parent / "src-iLand"
    print(f"   Path: {src_iland_path}")
    print(f"   Exists: {src_iland_path.exists()}")
    
    if src_iland_path.exists():
        sys.path.append(str(src_iland_path))
        print("   ✅ Added src-iLand to path")
    else:
        print("   ❌ src-iLand path not found")
        sys.exit(1)
    
    # Test retrieval imports
    print("4. Testing iLand retrieval imports...")
    try:
        from retrieval import (
            iLandRouterRetriever,
            create_default_iland_classifier,
            VectorRetrieverAdapter,
            SummaryRetrieverAdapter
        )
        print("   ✅ Core retrieval imports successful")
    except Exception as e:
        print(f"   ❌ Retrieval imports failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test embedding imports
    print("5. Testing embedding imports...")
    try:
        from load_embedding import (
            load_latest_iland_embeddings,
            load_all_latest_iland_embeddings,
            get_iland_batch_summary
        )
        print("   ✅ Embedding imports successful")
    except Exception as e:
        print(f"   ❌ Embedding imports failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test embedding loading
    print("6. Testing embedding loading...")
    try:
        embeddings_data, batch_path = load_all_latest_iland_embeddings()
        if embeddings_data:
            print(f"   ✅ Loaded {len(embeddings_data)} embeddings from {batch_path}")
        else:
            print("   ⚠️ No embeddings loaded")
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ Embedding loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test strategy creation
    print("7. Testing strategy creation...")
    try:
        vector_strategy = VectorRetrieverAdapter.from_iland_embeddings(
            embeddings_data, api_key=api_key
        )
        print("   ✅ Vector strategy created successfully")
    except Exception as e:
        print(f"   ❌ Strategy creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test basic query
    print("8. Testing basic query...")
    try:
        from llama_index.core.schema import QueryBundle
        query = "ค้นหาโฉนดที่ดินในจังหวัดชัยนาท"
        query_bundle = QueryBundle(query_str=query)
        nodes = vector_strategy.retrieve(query_bundle)
        print(f"   ✅ Query successful, found {len(nodes)} nodes")
        
        if nodes:
            first_node = nodes[0]
            print(f"   📄 First result score: {getattr(first_node, 'score', 'N/A')}")
            node_content = getattr(first_node, 'node', first_node)
            if hasattr(node_content, 'text'):
                preview = node_content.text[:100] + "..." if len(node_content.text) > 100 else node_content.text
                print(f"   📄 First result preview: {preview}")
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 All tests passed! The demo should work now.")
    print("Try running: python demo_iland_retrieval_pipeline.py")
    
except Exception as e:
    print(f"\n💥 Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 