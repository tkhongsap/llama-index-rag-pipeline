"""
Caching System for iLand Retrieval

Provides multi-level caching for frequent queries to improve performance.
"""

import os
import time
import hashlib
from typing import Dict, List, Optional, Any
from collections import OrderedDict

from llama_index.core.schema import NodeWithScore


class TTLCache:
    """Time-to-live cache with LRU eviction policy."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def _is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache or self._is_expired(key):
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
        self.timestamps[key] = time.time()
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.timestamps.pop(oldest_key, None)
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class QueryCache:
    """Cache for query results."""
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 1800):
        self.cache = TTLCache(max_size, ttl_seconds)
        self.hit_count = 0
        self.miss_count = 0
    
    def _create_query_key(self, query: str, strategy: str, top_k: int, index: str = "default") -> str:
        normalized_query = query.strip().lower()
        key_data = f"{normalized_query}|{strategy}|{top_k}|{index}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def get(self, query: str, strategy: str, top_k: int, index: str = "default") -> Optional[List[NodeWithScore]]:
        key = self._create_query_key(query, strategy, top_k, index)
        result = self.cache.get(key)
        if result is not None:
            self.hit_count += 1
            return result
        else:
            self.miss_count += 1
            return None
    
    def put(self, query: str, strategy: str, top_k: int, results: List[NodeWithScore], index: str = "default"):
        key = self._create_query_key(query, strategy, top_k, index)
        self.cache.put(key, results)
    
    def clear(self):
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def stats(self) -> Dict[str, Any]:
        total_queries = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_queries if total_queries > 0 else 0
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_queries": total_queries,
            **self.cache.stats()
        }


class iLandCacheManager:
    """Centralized cache manager for iLand retrieval system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.query_cache = QueryCache(
            max_size=config.get("query_cache_size", 500),
            ttl_seconds=config.get("query_cache_ttl", 1800)
        )
        self.query_caching_enabled = config.get("enable_query_cache", True)
    
    def get_query_results(self, query: str, strategy: str, top_k: int, index: str = "default") -> Optional[List[NodeWithScore]]:
        if not self.query_caching_enabled:
            return None
        return self.query_cache.get(query, strategy, top_k, index)
    
    def cache_query_results(self, query: str, strategy: str, top_k: int, results: List[NodeWithScore], index: str = "default"):
        if self.query_caching_enabled:
            self.query_cache.put(query, strategy, top_k, results, index)
    
    def clear_all_caches(self):
        self.query_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "query_cache": self.query_cache.stats(),
            "caching_enabled": {
                "query_cache": self.query_caching_enabled
            }
        }
    
    @classmethod
    def from_env(cls) -> "iLandCacheManager":
        config = {
            "query_cache_size": int(os.getenv("ILAND_QUERY_CACHE_SIZE", "500")),
            "query_cache_ttl": int(os.getenv("ILAND_QUERY_CACHE_TTL", "1800")),
            "enable_query_cache": os.getenv("ILAND_ENABLE_QUERY_CACHE", "true").lower() == "true"
        }
        return cls(config) 