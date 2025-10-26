"""
Two-Stage Router for Sales Promotion Retrieval

Orchestrates two-stage routing:
Stage 1: Index Classification (which data source)
Stage 2: Strategy Selection (which retrieval method)

Includes caching and result tagging.
"""

import time
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore

from .index_classifier import IndexClassifier
from .strategy_selector import StrategySelector
from .cache import QueryCache
from .retrievers.base import BaseRetrieverAdapter


class PromotionRouter:
    """
    Two-stage router for intelligent query routing.
    
    Workflow:
    1. Check cache for cached results
    2. Stage 1: Classify index (sales_promotions)
    3. Stage 2: Select optimal strategy (GPT-5 analysis)
    4. Execute selected retriever
    5. Tag results with routing metadata
    6. Cache results
    """
    
    def __init__(
        self,
        index_classifier: IndexClassifier,
        strategy_selector: StrategySelector,
        retrievers: Dict[str, BaseRetrieverAdapter],
        cache: Optional[QueryCache] = None,
        enable_caching: bool = True,
        fallback_strategy: str = "vector",
        debug_logging: bool = False
    ):
        """
        Initialize promotion router.
        
        Args:
            index_classifier: Stage 1 classifier
            strategy_selector: Stage 2 selector
            retrievers: Dictionary of strategy_name -> retriever
            cache: Query cache (optional)
            enable_caching: Enable result caching
            fallback_strategy: Strategy to use if selection fails
            debug_logging: Enable debug output
        """
        self.index_classifier = index_classifier
        self.strategy_selector = strategy_selector
        self.retrievers = retrievers
        self.cache = cache or QueryCache()
        self.enable_caching = enable_caching
        self.fallback_strategy = fallback_strategy
        self.debug_logging = debug_logging
    
    def route(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        strategy_override: Optional[str] = None
    ) -> List[NodeWithScore]:
        """
        Route query through two-stage system and retrieve results.
        
        Args:
            query: User query
            top_k: Number of results to return
            filters: Optional explicit metadata filters
            strategy_override: Manually specify strategy (skip Stage 2)
            
        Returns:
            List of retrieved nodes with routing metadata
        """
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching:
            cached_results = self.cache.get(query, top_k, filters)
            if cached_results:
                if self.debug_logging:
                    print(f"💾 Cache hit for query: {query[:50]}...")
                
                # Update metadata for cached results
                for node in cached_results:
                    node.node.metadata['cache_hit'] = True
                    node.node.metadata['routing_latency'] = time.time() - start_time
                
                return cached_results
        
        # Stage 1: Index Classification
        index_result = self.index_classifier.classify_query(query)
        selected_index = index_result["selected_index"]
        
        if self.debug_logging:
            print(f"📊 Stage 1: Index = {selected_index}")
        
        # Stage 2: Strategy Selection (skip if manual override)
        if strategy_override:
            selected_strategy = strategy_override
            strategy_confidence = 1.0
            if self.debug_logging:
                print(f"📊 Stage 2: Strategy = {selected_strategy} (manual override)")
        else:
            strategy_result = self.strategy_selector.select_strategy(query, selected_index)
            selected_strategy = strategy_result.get("strategy", self.fallback_strategy)
            strategy_confidence = strategy_result.get("confidence", 0.5)
            
            if self.debug_logging:
                print(f"📊 Stage 2: Strategy = {selected_strategy} "
                      f"(confidence: {strategy_confidence:.2f})")
                print(f"   Reasoning: {strategy_result.get('reasoning', 'N/A')}")
        
        # Execute retrieval
        retriever = self.retrievers.get(selected_strategy)
        
        if not retriever:
            print(f"⚠️  Strategy '{selected_strategy}' not available, using fallback")
            retriever = self.retrievers[self.fallback_strategy]
            selected_strategy = self.fallback_strategy
        
        # Perform retrieval
        nodes = retriever.retrieve(query, top_k)
        
        # Calculate routing latency
        routing_latency = time.time() - start_time
        
        # Tag nodes with routing metadata
        for node in nodes:
            node.node.metadata.update({
                "selected_index": selected_index,
                "selected_strategy": selected_strategy,
                "strategy_confidence": strategy_confidence,
                "routing_latency": routing_latency,
                "cache_hit": False
            })
        
        # Cache results
        if self.enable_caching:
            self.cache.set(query, nodes, top_k, filters)
        
        return nodes
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()

