#!/usr/bin/env python3
"""
Working iLand Retrieval Demo - Thai Land Deed RAG System

A fully functional demo with correct province name mappings for metadata filtering.
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

# Import response synthesis
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI

def main():
    print("🏛️ iLAND WORKING RETRIEVAL DEMO")
    print("🇹🇭 Thai Land Deed RAG System")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Load embeddings
    print("\n📊 Loading Thai land deed embeddings...")
    try:
        embeddings_data, batch_path = load_all_latest_iland_embeddings()
        print(f"✅ Loaded {len(embeddings_data)} embeddings")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return
    
    # Create strategies and response synthesizer
    print("\n🔧 Creating retrieval strategies...")
    
    try:
        strategies = {}
        strategies['vector'] = VectorRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
        strategies['metadata'] = MetadataRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
        strategies['hybrid'] = HybridRetrieverAdapter.from_iland_embeddings(embeddings_data, api_key=api_key)
        
        # Create response synthesizer for generating answers
        llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=llm
        )
        
        print(f"✅ Created {len(strategies)} strategies and response synthesizer")
    except Exception as e:
        print(f"❌ Error creating strategies: {e}")
        return
    
    # Test queries with CORRECT English province names
    test_queries = [
        {
            "query": "ค้นหาโฉนดที่ดินในจังหวัดชัยนาท", 
            "description": "Find land deeds in Chai Nat province",
            "filters": {"province": "**: Chai Nat"}
        },
        {
            "query": "ที่ดินในจังหวัดอ่างทอง", 
            "description": "Land in Ang Thong province",
            "filters": {"province": "**: Ang Thong"}
        },
        {
            "query": "โฉนดที่ดินประเภทกรรมสิทธิ์บริษัท", 
            "description": "Corporate land ownership deeds",
            "filters": None
        },
        {
            "query": "ที่ดินในอำเภอเมืองอ่างทอง", 
            "description": "Land in Mueang Ang Thong district",
            "filters": {"district": "**: Mueang Ang Thong"}
        }
    ]
    
    print("\n🔍 Testing queries with correct province/district names...")
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        description = test["description"]
        explicit_filters = test["filters"]
        
        print(f"\n--- Query {i}: {query} ---")
        print(f"    📝 Description: {description}")
        if explicit_filters:
            print(f"    🔧 Filters: {explicit_filters}")
        
        for strategy_name, strategy in strategies.items():
            start_time = time.time()
            
            try:
                if strategy_name == 'metadata':
                    # Test metadata strategy with correct English filters
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
                
                if nodes:
                    first_node = nodes[0]
                    score = getattr(first_node, 'score', 0)
                    print(f"  ✅ {strategy_name}: {len(nodes)} results in {elapsed:.2f}s (score: {score:.3f})")
                    
                    # Show location of first result
                    if hasattr(first_node, 'node') and hasattr(first_node.node, 'metadata'):
                        node_metadata = first_node.node.metadata
                        province = node_metadata.get('province', 'N/A')
                        district = node_metadata.get('district', 'N/A')
                        ownership = node_metadata.get('deed_holding_type', 'N/A')
                        print(f"      📍 Location: {province}, {district}")
                        print(f"      🏢 Ownership: {ownership}")
                    
                    # Generate RAG response
                    try:
                        response = response_synthesizer.synthesize(query, nodes)
                        print(f"      🤖 RAG Response:")
                        print(f"         {response.response[:200]}...")
                    except Exception as e:
                        print(f"      ⚠️ Response generation failed: {str(e)[:100]}...")
                        
                else:
                    print(f"  ❌ {strategy_name}: 0 results in {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  💥 {strategy_name}: ERROR - {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ DEMO SUMMARY:")
    print("• Vector strategy: Works well for semantic Thai text search")
    print("• Hybrid strategy: Combines vector + keyword search effectively") 
    print("• Metadata strategy: Requires exact English province/district names")
    print("\n🔧 Metadata Filter Examples:")
    print("• Province: {'province': '**: Ang Thong'} or {'province': '**: Chai Nat'}")
    print("• District: {'district': '**: Mueang Ang Thong'}")
    print("• Multiple: {'province': '**: Ang Thong', 'district': '**: Mueang Ang Thong'}")

if __name__ == "__main__":
    main() 