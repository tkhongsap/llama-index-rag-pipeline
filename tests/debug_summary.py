#!/usr/bin/env python3
"""
Debug script for summary strategy issue
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from agentic_retriever.cli import create_agentic_router

def debug_summary_strategy():
    """Debug the summary strategy to understand why it's returning empty results."""
    print("🔍 DEBUGGING SUMMARY STRATEGY")
    print("=" * 50)
    
    # Create the full agentic router
    print("1. Creating agentic router...")
    router = create_agentic_router()
    
    # Get compensation_docs retriever
    print("2. Getting compensation_docs retriever...")
    comp_retriever = router.retrievers.get('compensation_docs')
    if not comp_retriever:
        print("❌ No compensation_docs found!")
        print(f"Available indices: {list(router.retrievers.keys())}")
        return
    
    # Get summary adapter
    print("3. Getting summary adapter...")
    summary_adapter = comp_retriever.get('summary')
    if not summary_adapter:
        print("❌ No summary adapter found!")
        print(f"Available strategies: {list(comp_retriever.keys())}")
        return
    
    print(f"✅ Summary adapter found")
    print(f"   - Summary embeddings: {len(summary_adapter.summary_embeddings)}")
    print(f"   - Chunk embeddings: {len(summary_adapter.chunk_embeddings)}")
    
    # Test hierarchical_retrieve directly
    print("\n4. Testing hierarchical_retrieve directly...")
    query = "salary ranges for different positions"
    
    try:
        result = summary_adapter.retriever.hierarchical_retrieve(
            query=query,
            summary_top_k=5,
            chunks_per_doc=3
        )
        
        print(f"✅ hierarchical_retrieve completed")
        print(f"   Keys in result: {list(result.keys())}")
        print(f"   relevant_documents: {len(result.get('relevant_documents', []))}")
        print(f"   relevant_chunks: {len(result.get('relevant_chunks', []))}")
        
        # Show first few documents
        docs = result.get('relevant_documents', [])
        if docs:
            print(f"\n📄 First few documents:")
            for i, doc in enumerate(docs[:3]):
                print(f"   {i+1}. {doc.get('doc_title', 'Unknown')} (score: {doc.get('score', 0):.3f})")
        else:
            print("   ❌ No documents found!")
            
        # Show first few chunks  
        chunks = result.get('relevant_chunks', [])
        if chunks:
            print(f"\n📄 First few chunks:")
            for i, chunk in enumerate(chunks[:3]):
                text_preview = chunk.get('text', '')[:100] + "..." if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                print(f"   {i+1}. {text_preview}")
        else:
            print("   ❌ No chunks found!")
            
        # Test the adapter's retrieve method
        print(f"\n5. Testing adapter retrieve method...")
        nodes = summary_adapter.retrieve(query, top_k=5)
        print(f"   Nodes returned: {len(nodes)}")
        
        if nodes:
            print("   ✅ Summary adapter working!")
            for i, node in enumerate(nodes[:2]):
                print(f"   {i+1}. {node.node.text[:100]}...")
        else:
            print("   ❌ Adapter returning empty results")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_summary_strategy() 