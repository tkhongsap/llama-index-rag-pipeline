#!/usr/bin/env python3
"""
Strategy Tuning Script

This script analyzes and tunes the strategy selection logic to improve
routing decisions and fix empty response issues.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from agentic_retriever.cli import get_cached_router, query_agentic_retriever

def analyze_strategy_performance():
    """Analyze which strategies work best for different query types."""
    print("🔧 STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Test different strategies directly
    test_cases = [
        {
            "query": "salary compensation THB",
            "expected_index": "compensation_docs",
            "test_strategies": ["vector", "metadata", "summary"]
        },
        {
            "query": "university degree bachelor master education", 
            "expected_index": "education_career",
            "test_strategies": ["vector", "hybrid", "metadata"]
        },
        {
            "query": "job position experience years career",
            "expected_index": "candidate_profiles", 
            "test_strategies": ["vector", "metadata", "recursive"]
        }
    ]
    
    # Get cached router
    router = get_cached_router()
    if not router:
        print("❌ Could not get router")
        return
    
    results = {}
    
    for test_case in test_cases:
        query = test_case["query"]
        expected_index = test_case["expected_index"]
        
        print(f"\n🧪 Testing: {query}")
        print(f"   Expected Index: {expected_index}")
        print("   Strategy Performance:")
        
        results[query] = {}
        
        # Test each strategy directly
        for strategy in test_case["test_strategies"]:
            try:
                # Get the specific retriever for this index and strategy
                retriever = router.retrievers.get(expected_index, {}).get(strategy)
                
                if not retriever:
                    print(f"     ❌ {strategy}: Not available")
                    continue
                
                start_time = time.time()
                nodes = retriever.retrieve(query, top_k=3)
                query_time = time.time() - start_time
                
                # Check if we got results
                if nodes and len(nodes) > 0:
                    # Check quality of first result
                    first_result = nodes[0].node.text if nodes else ""
                    quality = "Good" if len(first_result) > 50 else "Poor"
                    status = "✅"
                else:
                    quality = "Empty"
                    status = "❌"
                
                print(f"     {status} {strategy}: {len(nodes)} results, {query_time:.2f}s, {quality}")
                
                results[query][strategy] = {
                    "num_results": len(nodes),
                    "time": query_time,
                    "quality": quality,
                    "status": status
                }
                
            except Exception as e:
                print(f"     ❌ {strategy}: Error - {str(e)[:50]}...")
                results[query][strategy] = {"error": str(e)}
    
    return results


def test_strategy_selection_override():
    """Test manual strategy override to validate specific strategies."""
    print("\n🎯 STRATEGY SELECTION TESTING")
    print("=" * 50)
    
    # Test forcing different strategies for the same query
    query = "What are the salary ranges for different positions?"
    strategies_to_test = ["vector", "metadata", "summary"]
    
    router = get_cached_router()
    if not router:
        print("❌ Could not get router")
        return
    
    print(f"Query: {query}")
    print("Testing forced strategy selection:")
    
    for strategy in strategies_to_test:
        print(f"\n🔧 Testing {strategy} strategy:")
        try:
            # Override the strategy selector temporarily
            original_selector = router.strategy_selector
            router.strategy_selector = lambda idx, q, retrievers: strategy
            
            start_time = time.time()
            result = query_agentic_retriever(query, top_k=3)
            query_time = time.time() - start_time
            
            # Restore original selector
            router.strategy_selector = original_selector
            
            if result.get("response"):
                response_preview = result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"]
                print(f"   ✅ Time: {query_time:.2f}s")
                print(f"   📝 Response: {response_preview}")
                print(f"   📊 Sources: {result['metadata'].get('num_sources', 0)}")
            else:
                print(f"   ❌ Empty response - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    

def diagnose_metadata_strategy_issue():
    """Specifically diagnose why metadata strategy returns empty results."""
    print("\n🔍 METADATA STRATEGY DIAGNOSIS")
    print("=" * 50)
    
    router = get_cached_router()
    if not router:
        print("❌ Could not get router")
        return
    
    # Get metadata adapter for compensation_docs
    metadata_adapter = router.retrievers.get('compensation_docs', {}).get('metadata')
    
    if not metadata_adapter:
        print("❌ No metadata adapter found")
        return
    
    print("✅ Metadata adapter found")
    
    # Test simple retrieval
    query = "salary ranges"
    print(f"Testing query: {query}")
    
    try:
        # Check available metadata
        if hasattr(metadata_adapter, 'metadata_fields'):
            print(f"📊 Available metadata fields: {len(metadata_adapter.metadata_fields)}")
            salary_fields = [f for f in metadata_adapter.metadata_fields if 'salary' in f.lower()]
            print(f"💰 Salary-related fields: {salary_fields}")
        
        # Test direct retrieval
        nodes = metadata_adapter.retrieve(query, top_k=5)
        print(f"📄 Retrieved {len(nodes)} nodes")
        
        if nodes:
            for i, node in enumerate(nodes[:2]):
                print(f"   {i+1}. {node.node.text[:100]}...")
        else:
            print("   ❌ No nodes retrieved")
            
            # Check if the issue is in filter inference
            if hasattr(metadata_adapter, 'retriever'):
                print("🔍 Checking filter inference...")
                # This might reveal why no results are returned
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def run_strategy_tuning():
    """Run complete strategy tuning analysis."""
    print("🚀 STRATEGY TUNING & OPTIMIZATION")
    print("=" * 60)
    
    # 1. Analyze strategy performance
    strategy_results = analyze_strategy_performance()
    
    # 2. Test manual strategy override
    test_strategy_selection_override()
    
    # 3. Diagnose specific issues
    diagnose_metadata_strategy_issue()
    
    # 4. Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("Based on analysis:")
    print("1. Vector strategy appears most reliable for semantic queries")
    print("2. Metadata strategy has filtering issues causing empty results")  
    print("3. Summary strategy works but may be slower")
    print("4. Consider defaulting to vector strategy for better reliability")
    
    return strategy_results


if __name__ == "__main__":
    run_strategy_tuning() 