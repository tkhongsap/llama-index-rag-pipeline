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

def main():
    print("🏛️ iLAND SIMPLE RETRIEVAL DEMO")
    print("🇹🇭 Thai Land Deed RAG System")
    print("=" * 50)
    
    # Load embeddings
    print("\n📊 Loading Thai land deed embeddings...")
    embeddings_data, batch_path = load_all_latest_iland_embeddings()
    print(f"✅ Loaded {len(embeddings_data)} embeddings")
    
    # Create strategies
    print("\n🔧 Creating retrieval strategies...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    strategies = {}
    strategies['vector'] = VectorRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
    strategies['metadata'] = MetadataRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
    strategies['hybrid'] = HybridRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
    
    print(f"✅ Created {len(strategies)} strategies")
    
    # Test queries
    test_queries = [
        "ค้นหาโฉนดที่ดินในจังหวัดชัยนาท",
        "โฉนดที่ดินประเภทกรรมสิทธิ์บริษัท",
        "ที่ดินในอำเภอเมืองอ่างทอง"
    ]
    
    print("\n🔍 Testing queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        for strategy_name, strategy in strategies.items():
            start_time = time.time()
            
            try:
                if strategy_name == 'metadata':
                    # Metadata strategy expects string
                    nodes = strategy.retrieve(query, top_k=3)
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
                
            except Exception as e:
                print(f"  {strategy_name}: ERROR - {str(e)}")
    
    print("\n✅ Demo complete!")
    print("\nFor full demo with all strategies, run:")
    print("python demo_iland_retrieval_pipeline.py")

if __name__ == "__main__":
    main() 