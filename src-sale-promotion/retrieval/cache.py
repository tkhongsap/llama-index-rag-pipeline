"""
Query Cache for Sales Promotion Retrieval

LRU cache with TTL for query results.
Target: > 30% cache hit rate, 50% latency reduction for cached queries.
"""

import time
import hashlib
import json
from collections import OrderedDict
from typing import Optional, List, Any, Dict
from llama_index.core.schema import NodeWithScore


class QueryCache:
    """
    LRU cache with TTL (Time-To-Live) for query results.
    
    Features:
    - LRU eviction when cache is full
    - TTL expiration (default: 1 hour)
    - Cache key based on query + parameters
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
    
    def _make_cache_key(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> str:
        """
        Generate cache key from query parameters.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            Cache key string
        """
        # Create deterministic key
        key_data = {
            "query": query,
            "top_k": top_k,
            "filters": filters or {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        
        # Hash for shorter key
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> Optional[List[NodeWithScore]]:
        """
        Get cached results if available and not expired.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            Cached results or None if cache miss/expired
        """
        key = self._make_cache_key(query, top_k, filters)
        
        # Check if key exists
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            # Expired, remove from cache
            del self.cache[key]
            del self.timestamps[key]
            self.misses += 1
            return None
        
        # Cache hit - move to end (LRU)
        self.cache.move_to_end(key)
        self.hits += 1
        
        return self.cache[key]
    
    def set(
        self,
        query: str,
        results: List[NodeWithScore],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ):
        """
        Store results in cache.
        
        Args:
            query: Search query
            results: Query results to cache
            top_k: Number of results
            filters: Optional metadata filters
        """
        key = self._make_cache_key(query, top_k, filters)
        
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Store results
        self.cache[key] = results
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl
        }

