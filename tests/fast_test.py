#!/usr/bin/env python3
"""
Fast test script for performance validation
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from agentic_retriever.cli import query_agentic_retriever, get_cached_router

def fast_performance_test():
    """Test performance improvements with caching."""
    print("🚀 FAST PERFORMANCE TEST")
    print("=" * 50)
    
    test_queries = [
        "salary compensation THB",
        "university degree bachelor master education", 
        "job position experience years career",
        "What are the salary ranges for different positions?",
        "What educational backgrounds do the candidates have?"
    ]
    
    print("🔄 First query (cold cache)...")
    start_time = time.time()
    result1 = query_agentic_retriever(test_queries[0], fast_mode=True)
    first_query_time = time.time() - start_time
    
    print(f"✅ First query completed in {first_query_time:.2f}s")
    print(f"Response: {result1['response'][:100]}...")
    print(f"Index: {result1['metadata'].get('index')}, Strategy: {result1['metadata'].get('strategy')}")
    
    print("\n🚀 Subsequent queries (warm cache)...")
    
    for i, query in enumerate(test_queries[1:], 2):
        start_time = time.time()
        result = query_agentic_retriever(query, fast_mode=True)
        query_time = time.time() - start_time
        
        print(f"✅ Query {i} completed in {query_time:.2f}s")
        print(f"   Response: {result['response'][:80]}...")
        print(f"   Index: {result['metadata'].get('index')}, Strategy: {result['metadata'].get('strategy')}")
        
        if result['error']:
            print(f"   ❌ Error: {result['error']}")

if __name__ == "__main__":
    fast_performance_test() 