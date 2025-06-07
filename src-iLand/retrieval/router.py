"""
Router Retriever for iLand Data

Main router that implements two-stage routing: index classification → strategy selection.
Adapted from src/agentic_retriever/router.py for Thai land deed data.
"""

import os
import time
from typing import Dict, List, Optional, Any

from llama_index.core import Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.llms.openai import OpenAI

from index_classifier import iLandIndexClassifier, create_default_iland_classifier
from retrievers.base import BaseRetrieverAdapter
from cache import iLandCacheManager

# Import logging utilities from src package
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.agentic_retriever.log_utils import log_retrieval_call
except ImportError:
    # Fallback logging function
    def log_retrieval_call(query: str, selected_index: str, selected_strategy: str, 
                          latency_ms: float, confidence: float = None, **kwargs):
        print(f"iLand Retrieval: {query[:50]}... -> {selected_index}/{selected_strategy} ({latency_ms:.1f}ms)")


class iLandRouterRetriever(BaseRetriever):
    """Router retriever for iLand Thai land deed data."""
    
    def __init__(self,
                 retrievers: Dict[str, Dict[str, BaseRetrieverAdapter]],
                 index_classifier: Optional[iLandIndexClassifier] = None,
                 strategy_selector: Optional[str] = "llm",
                 llm_strategy_mode: Optional[str] = "enhanced",
                 api_key: Optional[str] = None,
                 cache_manager: Optional[iLandCacheManager] = None,
                 enable_caching: bool = True):
        """
        Initialize iLand router retriever.
        
        Args:
            retrievers: Dict mapping index_name -> {strategy_name -> adapter}
            index_classifier: Index classifier (creates default if None)
            strategy_selector: Strategy selection method ("llm", "heuristic", "round_robin")
            llm_strategy_mode: LLM strategy mode ("enhanced", "simple")
            api_key: OpenAI API key
            cache_manager: Cache manager instance (creates default if None)
            enable_caching: Whether to enable caching
        """
        super().__init__()
        
        self.retrievers = retrievers
        self.strategy_selector = strategy_selector or "llm"
        self.llm_strategy_mode = llm_strategy_mode or "enhanced"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._default_strategy = "vector"
        self.enable_caching = enable_caching
        
        # Setup index classifier
        if index_classifier is None:
            self.index_classifier = create_default_iland_classifier(api_key=self.api_key)
        else:
            self.index_classifier = index_classifier
        
        # Setup cache manager
        if cache_manager is None and enable_caching:
            self.cache_manager = iLandCacheManager.from_env()
        else:
            self.cache_manager = cache_manager
        
        # Setup models
        self._setup_models()
        
        # Round-robin state for strategy selection
        self._strategy_round_robin_state = {}
        
        # Debug logging
        self.debug_logging = False
    
    def _setup_models(self):
        """Setup LLM for strategy selection."""
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key
        )
    
    def _select_strategy_llm_enhanced(self, index: str, query: str, 
                                    available_strategies: List[str]) -> Dict[str, Any]:
        """
        Select strategy using enhanced LLM with detailed reasoning for iLand data.
        
        Args:
            index: The selected index name
            query: The user query (may contain Thai text)
            available_strategies: List of available strategies for this index
            
        Returns:
            Dict with strategy, confidence, method, and reasoning
        """
        # Strategy descriptions adapted for Thai land deed data
        strategy_descriptions = {
            "vector": "Semantic similarity search using embeddings. Best for: finding conceptually similar Thai land deed content, general queries about property ownership, when you need fast and reliable results.",
            "hybrid": "Combines semantic search with Thai keyword matching. Best for: queries that need both conceptual understanding and exact Thai term matches (โฉนด, นส.3, etc.), comprehensive search results.",
            "recursive": "Hierarchical retrieval that can drill down from summaries to details. Best for: complex queries requiring multi-level information about land ownership, when you need to explore document structure.",
            "chunk_decoupling": "Separates chunk retrieval from context synthesis. Best for: detailed analysis of specific land deed sections, when you need precise chunk-level information with broader context.",
            "planner": "Multi-step query planning and execution. Best for: complex multi-part questions about Thai land procedures, analytical tasks requiring step-by-step reasoning.",
            "metadata": "Filters documents based on Thai geographic and legal metadata. Best for: queries about specific provinces (จังหวัด), districts (อำเภอ), land deed types, or structured attributes.",
            "summary": "Retrieves from document summaries first. Best for: overview questions about land deed processes, when you need high-level information about Thai property law."
        }
        
        # Filter available strategies and build descriptions
        available_descriptions = []
        for strategy in available_strategies:
            if strategy in strategy_descriptions:
                available_descriptions.append(f"- {strategy}: {strategy_descriptions[strategy]}")
        
        if not available_descriptions:
            return {
                "strategy": available_strategies[0] if available_strategies else "vector",
                "confidence": 0.1,
                "method": "llm_fallback",
                "reasoning": "No known strategies available, using fallback"
            }
        
        # Create enhanced LLM prompt for Thai land deed strategy selection
        prompt = f"""You are an expert retrieval system for Thai land deed data that selects the best strategy for answering user queries.

Query: "{query}"
Index: {index}

Available retrieval strategies:
{chr(10).join(available_descriptions)}

Analyze the query and provide your response in this exact format:
STRATEGY: [strategy_name]
CONFIDENCE: [0.1-1.0]
REASONING: [brief explanation of why this strategy is best for Thai land deed data]

Consider:
1. Query complexity (simple vs multi-part)
2. Thai language content and terminology (โฉนด, นส.3, จังหวัด, etc.)
3. Information type needed (overview vs specific details)  
4. Search requirements (semantic vs keyword vs structured metadata)
5. Performance vs accuracy trade-offs for Thai content

Example response:
STRATEGY: vector
CONFIDENCE: 0.9
REASONING: Simple semantic query requiring fast, reliable similarity search for Thai land deed content"""

        try:
            response = Settings.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse LLM response
            strategy = None
            confidence = 0.5
            reasoning = "LLM selection for iLand data"
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('STRATEGY:'):
                    strategy = line.split(':', 1)[1].strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        confidence = max(0.1, min(1.0, confidence))  # Clamp to valid range
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Validate strategy selection
            if strategy and strategy in available_strategies:
                return {
                    "strategy": strategy,
                    "confidence": confidence,
                    "method": "llm_enhanced",
                    "reasoning": reasoning
                }
            
            # Try partial matching
            if strategy:
                for available_strategy in available_strategies:
                    if (available_strategy.lower() in strategy or 
                        strategy in available_strategy.lower()):
                        return {
                            "strategy": available_strategy,
                            "confidence": max(0.3, confidence - 0.2),  # Reduce confidence for partial match
                            "method": "llm_partial_match",
                            "reasoning": f"Partial match: {reasoning}"
                        }
            
            # Fallback to heuristic selection
            fallback_strategy = self._select_strategy_fallback(available_strategies, query)
            return {
                "strategy": fallback_strategy,
                "confidence": 0.3,
                "method": "llm_fallback_heuristic",
                "reasoning": f"LLM response invalid, using heuristic fallback: {fallback_strategy}"
            }
            
        except Exception as e:
            # Fallback to heuristic selection
            fallback_strategy = self._select_strategy_fallback(available_strategies, query)
            return {
                "strategy": fallback_strategy,
                "confidence": 0.2,
                "method": "llm_error_fallback",
                "reasoning": f"LLM error: {str(e)}, using heuristic fallback"
            }
    
    def _select_strategy_fallback(self, available_strategies: List[str], query: str) -> str:
        """
        Fallback strategy selection using heuristics for Thai land deed data.
        
        Args:
            available_strategies: List of available strategies
            query: The user query for basic heuristics
            
        Returns:
            Selected strategy name
        """
        # Strategy reliability ranking for Thai land deed data
        strategy_priority = {
            "vector": 1,        # Most reliable, fast, works well with Thai content
            "hybrid": 2,        # Good for Thai keyword + semantic search
            "recursive": 3,     # Good for complex hierarchical queries
            "chunk_decoupling": 4,  # Good for detailed chunk analysis
            "planner": 5,       # Good for multi-step Thai land deed queries
            "metadata": 6,      # Good for geographic/legal filtering
            "summary": 7        # Good for overview queries
        }
        
        # Filter available strategies by reliability
        reliable_strategies = [s for s in available_strategies if s in strategy_priority]
        
        if not reliable_strategies:
            # Fallback to first available if none are in our priority list
            return available_strategies[0] if available_strategies else "vector"
        
        # Basic heuristics for Thai land deed queries
        query_lower = query.lower()
        
        # For Thai geographic queries, prefer metadata strategy
        thai_geo_terms = ["จังหวัด", "อำเภอ", "ตำบล", "กรุงเทพ", "สมุทรปราการ", "นนทบุรี"]
        if any(term in query for term in thai_geo_terms):
            if "metadata" in reliable_strategies:
                return "metadata"
        
        # For Thai land deed type queries, prefer hybrid for keyword matching
        land_deed_terms = ["โฉนด", "นส.3", "นส.4", "ส.ค.1"]
        if any(term in query for term in land_deed_terms):
            if "hybrid" in reliable_strategies:
                return "hybrid"
        
        # For simple semantic queries, prefer vector strategy
        simple_query_indicators = ["what", "show", "list", "find", "get", "which", "อะไร", "แสดง", "หา"]
        if any(indicator in query_lower for indicator in simple_query_indicators):
            if "vector" in reliable_strategies:
                return "vector"
        
        # For complex multi-step queries, prefer planner
        complex_indicators = ["first", "then", "analyze", "compare", "breakdown", "step", "ขั้นตอน", "วิเคราะห์"]
        if any(indicator in query_lower for indicator in complex_indicators):
            if "planner" in reliable_strategies:
                return "planner"
            elif "recursive" in reliable_strategies:
                return "recursive"
        
        # Default to most reliable available strategy
        reliable_strategies.sort(key=lambda s: strategy_priority.get(s, 10))
        return reliable_strategies[0]
    
    def _select_strategy(self, query: str, index_name: str) -> Dict[str, Any]:
        """
        Select retrieval strategy for the given query and index.
        
        Args:
            query: User query
            index_name: Selected index name
            
        Returns:
            Dict with strategy selection results
        """
        # Get available strategies for this index
        available_strategies = list(self.retrievers.get(index_name, {}).keys())
        
        if not available_strategies:
            return {
                "strategy": self._default_strategy,
                "confidence": 0.1,
                "method": "no_strategies_fallback",
                "reasoning": f"No strategies available for index {index_name}"
            }
        
        # Select strategy based on configured method
        if self.strategy_selector == "llm":
            if self.llm_strategy_mode == "enhanced":
                return self._select_strategy_llm_enhanced(index_name, query, available_strategies)
            else:
                # Simple LLM mode (can be implemented later)
                return self._select_strategy_llm_enhanced(index_name, query, available_strategies)
        elif self.strategy_selector == "round_robin":
            return self._select_strategy_round_robin(index_name, available_strategies)
        else:  # heuristic
            strategy = self._select_strategy_fallback(available_strategies, query)
            return {
                "strategy": strategy,
                "confidence": 0.7,
                "method": "heuristic",
                "reasoning": f"Heuristic selection: {strategy}"
            }
    
    def _select_strategy_round_robin(self, index_name: str, available_strategies: List[str]) -> Dict[str, Any]:
        """Select strategy using round-robin for testing purposes."""
        if not available_strategies:
            return {
                "strategy": self._default_strategy,
                "confidence": 0.1,
                "method": "round_robin_fallback",
                "reasoning": "No strategies available"
            }
        
        # Initialize round-robin state for this index
        if index_name not in self._strategy_round_robin_state:
            self._strategy_round_robin_state[index_name] = 0
        
        # Get next strategy
        current_index = self._strategy_round_robin_state[index_name]
        selected_strategy = available_strategies[current_index % len(available_strategies)]
        
        # Update state
        self._strategy_round_robin_state[index_name] = (current_index + 1) % len(available_strategies)
        
        return {
            "strategy": selected_strategy,
            "confidence": 0.5,
            "method": "round_robin",
            "reasoning": f"Round-robin selection: {selected_strategy}"
        }
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Main retrieval method implementing two-stage routing with caching.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of retrieved nodes with routing metadata
        """
        query = query_bundle.query_str
        start_time = time.time()
        
        # Stage 1: Index Classification
        index_result = self.index_classifier.classify_query(query)
        selected_index = index_result["selected_index"]
        index_confidence = index_result["confidence"]
        
        # Stage 2: Strategy Selection
        strategy_result = self._select_strategy(query, selected_index)
        selected_strategy = strategy_result["strategy"]
        strategy_confidence = strategy_result["confidence"]
        
        # Check cache first if enabled
        cached_results = None
        if self.enable_caching and self.cache_manager:
            cached_results = self.cache_manager.get_query_results(
                query=query,
                strategy=selected_strategy,
                top_k=5,  # Default top_k, should be configurable
                index=selected_index
            )
            if cached_results:
                # Calculate cache hit latency
                latency = time.time() - start_time
                
                # Update metadata for cached results
                for node in cached_results:
                    if hasattr(node.node, 'metadata'):
                        node.node.metadata.update({
                            "cache_hit": True,
                            "routing_latency": latency
                        })
                
                if self.debug_logging:
                    print(f"Cache hit for query: {query[:50]}... -> {selected_index}/{selected_strategy}")
                
                return cached_results
        
        # Get the appropriate retriever adapter
        if (selected_index in self.retrievers and 
            selected_strategy in self.retrievers[selected_index]):
            adapter = self.retrievers[selected_index][selected_strategy]
        else:
            # Fallback to any available adapter
            for idx, strategies in self.retrievers.items():
                if strategies:
                    adapter = list(strategies.values())[0]
                    selected_index = idx
                    selected_strategy = list(strategies.keys())[0]
                    break
            else:
                # No adapters available
                return []
        
        # Perform retrieval
        try:
            nodes = adapter.retrieve(query)
        except Exception as e:
            if self.debug_logging:
                print(f"Retrieval error: {e}")
            nodes = []
        
        # Cache results if enabled
        if self.enable_caching and self.cache_manager and nodes:
            self.cache_manager.cache_query_results(
                query=query,
                strategy=selected_strategy,
                top_k=len(nodes),
                results=nodes,
                index=selected_index
            )
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Enrich nodes with routing metadata
        for node in nodes:
            if hasattr(node.node, 'metadata'):
                node.node.metadata.update({
                    "selected_index": selected_index,
                    "selected_strategy": selected_strategy,
                    "index_confidence": index_confidence,
                    "strategy_confidence": strategy_confidence,
                    "routing_latency": latency,
                    "index_method": index_result.get("method", "unknown"),
                    "strategy_method": strategy_result.get("method", "unknown"),
                    "data_source": "iland",
                    "cache_hit": False  # Set to False for fresh retrievals
                })
        
        # Log the retrieval call
        log_retrieval_call(
            query=query,
            selected_index=selected_index,
            selected_strategy=selected_strategy,
            latency_ms=latency * 1000,  # Convert to milliseconds
            confidence=min(index_confidence, strategy_confidence)
        )
        
        if self.debug_logging:
            print(f"iLand Router: {query[:50]}... -> {selected_index}/{selected_strategy} "
                  f"({len(nodes)} results, {latency:.2f}s)")
        
        return nodes
    
    def add_retriever(self, index_name: str, strategy_name: str, adapter: BaseRetrieverAdapter):
        """Add a retriever adapter to the router."""
        if index_name not in self.retrievers:
            self.retrievers[index_name] = {}
        self.retrievers[index_name][strategy_name] = adapter
    
    def remove_retriever(self, index_name: str, strategy_name: str):
        """Remove a retriever adapter from the router."""
        if (index_name in self.retrievers and 
            strategy_name in self.retrievers[index_name]):
            del self.retrievers[index_name][strategy_name]
            
            # Remove index if no strategies left
            if not self.retrievers[index_name]:
                del self.retrievers[index_name]
    
    def get_available_retrievers(self) -> Dict[str, List[str]]:
        """Get all available retrievers by index and strategy."""
        return {
            index: list(strategies.keys()) 
            for index, strategies in self.retrievers.items()
        }
    
    def enable_debug_logging(self, enabled: bool = True):
        """Enable or disable debug logging."""
        self.debug_logging = enabled
    
    @classmethod
    def from_adapters(cls,
                     adapters: Dict[str, Dict[str, BaseRetrieverAdapter]],
                     api_key: Optional[str] = None,
                     strategy_selector: str = "llm",
                     llm_strategy_mode: str = "enhanced",
                     enable_caching: bool = True) -> "iLandRouterRetriever":
        """
        Create router from adapter dictionary.
        
        Args:
            adapters: Dict mapping index_name -> {strategy_name -> adapter}
            api_key: OpenAI API key
            strategy_selector: Strategy selection method
            llm_strategy_mode: LLM strategy mode
            enable_caching: Whether to enable caching
            
        Returns:
            iLandRouterRetriever instance
        """
        return cls(
            retrievers=adapters,
            api_key=api_key,
            strategy_selector=strategy_selector,
            llm_strategy_mode=llm_strategy_mode,
            enable_caching=enable_caching
        ) 