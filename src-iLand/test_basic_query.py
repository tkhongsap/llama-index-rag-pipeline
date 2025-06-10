#!/usr/bin/env python
"""
Minimal test to identify CLI issues
"""

import sys
from pathlib import Path

# Add the current directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    print("Starting minimal CLI test...")
    
    from retrieval.cli_handlers import iLandRetrievalCLI
    print("✅ Imported CLI successfully")
    
    cli = iLandRetrievalCLI()
    print("✅ Created CLI instance")
    
    print("\n📚 Loading embeddings...")
    success = cli.load_embeddings("latest")
    if not success:
        print("❌ Failed to load embeddings")
        sys.exit(1)
    print("✅ Embeddings loaded")
    
    print("\n🚀 Creating router...")
    success = cli.create_router("llm")
    if not success:
        print("❌ Failed to create router")
        sys.exit(1)
    print("✅ Router created")
    
    print("\n🔍 Testing query...")
    query = "how many documents"
    print(f"Query: '{query}'")
    
    results = cli.query(query, top_k=2)  # Limit to 2 results
    print(f"✅ Query completed with {len(results)} results")
    
    print("\n✅ Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    print(f"Full error: {traceback.format_exc()}")
    sys.exit(1) 