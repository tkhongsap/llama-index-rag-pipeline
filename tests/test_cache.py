#!/usr/bin/env python3
"""
Test script for the persistent cache system
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_retriever.cli import cache, query_agentic_retriever

def test_cache_system():
    """Test the persistent cache system."""
    print("🧪 Testing Persistent Cache System")
    print("=" * 50)
    
    # Clear cache first
    print("1. Clearing all caches...")
    cache.clear_all_caches()
    
    # Show initial cache status
    print("\n2. Initial cache status:")
    cache.show_status()
    
    # First query (cold start)
    print("\n3. First query (COLD START):")
    print("-" * 30)
    start_time = time.time()
    result1 = query_agentic_retriever(
        query="What are these documents about?",
        show_performance=True
    )
    cold_time = time.time() - start_time
    
    if result1["error"]:
        print(f"❌ Error: {result1['error']}")
        return
    
    print(f"\n✅ Cold start completed in {cold_time*1000:.0f}ms")
    
    # Show cache status after first query
    print("\n4. Cache status after first query:")
    cache.show_status()
    
    # Second query (warm cache)
    print("\n5. Second query (WARM CACHE):")
    print("-" * 30)
    start_time = time.time()
    result2 = query_agentic_retriever(
        query="Find the best candidate",
        show_performance=True
    )
    warm_time = time.time() - start_time
    
    if result2["error"]:
        print(f"❌ Error: {result2['error']}")
        return
    
    print(f"\n✅ Warm cache completed in {warm_time*1000:.0f}ms")
    
    # Performance comparison
    print("\n6. Performance Comparison:")
    print("=" * 50)
    print(f"🔥 Cold Start: {cold_time*1000:.0f}ms")
    print(f"⚡ Warm Cache: {warm_time*1000:.0f}ms")
    speedup = cold_time / warm_time if warm_time > 0 else 0
    print(f"🚀 Speedup: {speedup:.1f}x faster")
    
    if speedup > 5:
        print("✅ Cache system working correctly!")
    else:
        print("⚠️ Cache system may not be working as expected")

if __name__ == "__main__":
    test_cache_system() 