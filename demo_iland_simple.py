#!/usr/bin/env python3
"""
Simple iLand Retrieval Demo - Thai Land Deed RAG System

A simplified version of the comprehensive demo that focuses on core functionality.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv(override=True)

# Add src-iLand to path
sys.path.append(str(Path(__file__).parent / "src-iLand"))

# Import iLand modules
from retrieval import VectorRetrieverAdapter, MetadataRetrieverAdapter, HybridRetrieverAdapter
from load_embedding import load_all_latest_iland_embeddings

def debug_metadata_fields(embeddings_data):
    """Debug what metadata fields are actually available."""
    print("\n🔍 DEBUGGING METADATA FIELDS")
    print("=" * 40)
    
    if not embeddings_data:
        print("❌ No embeddings data available")
        return
    
    # Sample first few embeddings to understand structure
    for i, emb in enumerate(embeddings_data[:3]):
        print(f"\n📋 Sample Embedding {i+1}:")
        metadata = emb.get('metadata', {})
        print(f"  Available metadata keys: {list(metadata.keys())}")
        
        # Show key Thai land deed fields
        for key in ['province', 'district', 'deed_type', 'land_use_type', 'deed_holding_type']:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")
        
        # Check if this is a deed with province info
        if 'province' in metadata:
            print(f"  ✅ Has province: {metadata['province']}")
        
        if i == 0:  # Show full metadata for first item
            print(f"\n  Full metadata keys: {sorted(metadata.keys())}")
    
    # Show province distribution
    provinces = {}
    for emb in embeddings_data:
        metadata = emb.get('metadata', {})
        province = metadata.get('province', 'Unknown')
        provinces[province] = provinces.get(province, 0) + 1
    
    print(f"\n🏘️ Province Distribution:")
    for province, count in sorted(provinces.items()):
        print(f"  {province}: {count} records")

def main():
    print("🏛️ iLAND SIMPLE RETRIEVAL DEMO")
    print("🇹🇭 Thai Land Deed RAG System")
    print("=" * 50)
    
    # Load embeddings
    print("\n📊 Loading Thai land deed embeddings...")
    embeddings_data, batch_path = load_all_latest_iland_embeddings()
    print(f"✅ Loaded {len(embeddings_data)} embeddings")
    
    # Debug metadata fields
    debug_metadata_fields(embeddings_data)
    
    # Create strategies
    print("\n🔧 Creating retrieval strategies...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    strategies = {}
    strategies['vector'] = VectorRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
    strategies['metadata'] = MetadataRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
    strategies['hybrid'] = HybridRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
    
    print(f"✅ Created {len(strategies)} strategies")
    
    # Test queries with explicit metadata filters
    test_queries = [
        ("ค้นหาโฉนดที่ดินในจังหวัดชัยนาท", {"province": "ชัยนาท"}),
        ("โฉนดที่ดินประเภทกรรมสิทธิ์บริษัท", None),
        ("ที่ดินในอำเภอเมืองอ่างทอง", {"province": "อ่างทอง"}),
        ("ที่ดินในจังหวัดอ่างทอง", {"province": "อ่างทอง"})
    ]
    
    print("\n🔍 Testing queries...")
    for i, (query, explicit_filters) in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        if explicit_filters:
            print(f"    Explicit filters: {explicit_filters}")
        
        for strategy_name, strategy in strategies.items():
            start_time = time.time()
            
            try:
                if strategy_name == 'metadata':
                    # Test metadata strategy with explicit filters
                    nodes = strategy.retrieve(query, top_k=5, filters=explicit_filters)
                elif strategy_name == 'hybrid':
                    # Hybrid strategy expects string
                    nodes = strategy.retrieve(query, top_k=3)
                else:
                    # Vector strategy expects QueryBundle
                    from llama_index.core.schema import QueryBundle
                    query_bundle = QueryBundle(query_str=query)
                    nodes = strategy.retrieve(query_bundle)
                
                elapsed = time.time() - start_time
                print(f"  {strategy_name}: {len(nodes)} results in {elapsed:.2f}s")
                
                if nodes:
                    first_node = nodes[0]
                    score = getattr(first_node, 'score', 0)
                    print(f"    Top result score: {score:.3f}")
                    
                    # Show metadata of first result for debugging
                    if hasattr(first_node, 'node') and hasattr(first_node.node, 'metadata'):
                        node_metadata = first_node.node.metadata
                        province = node_metadata.get('province', 'N/A')
                        district = node_metadata.get('district', 'N/A')
                        print(f"    Top result location: {province}, {district}")
                
            except Exception as e:
                print(f"  {strategy_name}: ERROR - {str(e)}")
    
    print("\n✅ Demo complete!")
    print("\n💡 Tips for metadata filtering:")
    print("- Province names are stored in English format (**: Ang Thong, **: Chai Nat)")
    print("- Use explicit filters with English province names for better matching")
    print("- Thai queries work with vector/hybrid strategies but metadata needs exact matches")

if __name__ == "__main__":
    main() 