"""
Router Module

Main router that combines index classification and retrieval strategy selection.
Provides both local RouterRetriever and cloud LlamaCloudCompositeRetriever.

Usage:
    The RouterRetriever is strategy-agnostic and accepts any pre-built retrievers.
    For the complete 7-strategy implementation, see cli.py which creates all adapters
    and builds the full router system.
    
    Use RouterRetriever.from_retrievers() to create a router with your retrievers.
"""

import os
import time
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from .index_classifier import IndexClassifier, create_default_classifier
from .log_utils import log_retrieval_call

# Load environment variables
load_dotenv(override=True)


class RouterRetriever(BaseRetriever):
    """
    Main router that selects the appropriate index and retrieval strategy.
    """
    
    def __init__(
        self,
        retrievers: Dict[str, Dict[str, BaseRetriever]],
        index_classifier: Optional[IndexClassifier] = None,
        strategy_selector: Optional[str] = "llm",
        llm_strategy_mode: Optional[str] = "enhanced",  # "enhanced", "simple"
        api_key: Optional[str] = None
    ):
        """
        Initialize the router retriever.
        
        Args:
            retrievers: Nested dict {index_name: {strategy_name: retriever}}
            index_classifier: Index classifier instance
            strategy_selector: Strategy selection method ("llm", "round_robin", "default")
            llm_strategy_mode: LLM strategy selection mode ("enhanced", "simple")
            api_key: OpenAI API key
        """
        super().__init__()
        self.retrievers = retrievers
        self.index_classifier = index_classifier or create_default_classifier(api_key)
        self.strategy_selector = strategy_selector
        self.llm_strategy_mode = llm_strategy_mode
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Setup models
        self._setup_models()
        
        # Strategy selection state
        self._strategy_round_robin_state = {}
        self._default_strategy = "vector"
        
        # Store last routing info for CLI display
        self.last_routing_info = {}
    
    def _setup_models(self):
        """Setup LLM and embedding models."""
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=self.api_key
        )
    
    def _select_strategy_llm(self, index: str, query: str, available_strategies: List[str]) -> str:
        """
        Select the best retrieval strategy using LLM reasoning.
        
        Args:
            index: The selected index name
            query: The user query
            available_strategies: List of available strategies for this index
            
        Returns:
            Selected strategy name
        """
        # Strategy descriptions for LLM understanding
        strategy_descriptions = {
            "vector": "Semantic similarity search using embeddings. Best for: finding conceptually similar content, general queries, when you need fast and reliable results.",
            "hybrid": "Combines semantic search with keyword matching. Best for: queries that need both conceptual and exact keyword matches, comprehensive search results.",
            "recursive": "Hierarchical retrieval that can drill down from summaries to details. Best for: complex queries requiring multi-level information, when you need to explore document structure.",
            "chunk_decoupling": "Separates chunk retrieval from context synthesis. Best for: detailed analysis, when you need precise chunk-level information with broader context.",
            "planner": "Multi-step query planning and execution. Best for: complex multi-part questions, analytical tasks requiring step-by-step reasoning.",
            "metadata": "Filters documents based on metadata attributes. Best for: queries about specific document types, dates, categories, or structured attributes.",
            "summary": "Retrieves from document summaries first. Best for: overview questions, when you need high-level information about documents."
        }
        
        # Filter available strategies and build descriptions
        available_descriptions = []
        for strategy in available_strategies:
            if strategy in strategy_descriptions:
                available_descriptions.append(f"- {strategy}: {strategy_descriptions[strategy]}")
        
        if not available_descriptions:
            # Fallback if no known strategies
            return available_strategies[0] if available_strategies else "vector"
        
        # Create LLM prompt for strategy selection
        prompt = f"""You are an expert retrieval system that selects the best strategy for answering user queries.

Query: "{query}"
Index: {index}

Available retrieval strategies:
{chr(10).join(available_descriptions)}

Analyze the query and select the SINGLE BEST strategy. Consider:
1. Query complexity (simple vs multi-part)
2. Information type needed (overview vs specific details)
3. Search requirements (semantic vs keyword vs structured)
4. Performance vs accuracy trade-offs

Respond with ONLY the strategy name (e.g., "vector", "hybrid", "recursive", etc.). No explanation needed."""

        try:
            response = Settings.llm.complete(prompt)
            selected_strategy = response.text.strip().lower()
            
            # Validate LLM response
            if selected_strategy in available_strategies:
                return selected_strategy
            
            # Try to find partial match if exact match fails
            for strategy in available_strategies:
                if strategy.lower() in selected_strategy or selected_strategy in strategy.lower():
                    return strategy
            
            # Fallback to reliability-based selection if LLM response is invalid
            return self._select_strategy_fallback(available_strategies, query)
            
        except Exception as e:
            print(f"⚠️  LLM strategy selection failed: {e}")
            return self._select_strategy_fallback(available_strategies, query)
    
    def _select_strategy_fallback(self, available_strategies: List[str], query: str) -> str:
        """
        Fallback strategy selection using reliability ranking and simple heuristics.
        
        Args:
            available_strategies: List of available strategies
            query: The user query for basic heuristics
            
        Returns:
            Selected strategy name
        """
        # Strategy reliability ranking based on performance analysis
        strategy_priority = {
            "vector": 1,        # Most reliable, fast, always works
            "hybrid": 2,        # Good results but slower  
            "recursive": 3,     # Good for complex hierarchical queries
            "chunk_decoupling": 4,  # Good for detailed chunk analysis
            "planner": 5,       # Good for multi-step queries
            "metadata": 6,      # Unreliable, often empty results
            "summary": 7        # Currently broken, empty results
        }
        
        # Filter available strategies by reliability
        reliable_strategies = [s for s in available_strategies if s in strategy_priority]
        
        if not reliable_strategies:
            # Fallback to first available if none are in our priority list
            return available_strategies[0] if available_strategies else "vector"
        
        # Basic heuristics for fallback
        query_lower = query.lower()
        
        # For simple semantic queries, prefer vector strategy
        simple_query_indicators = ["what", "show", "list", "find", "get", "which"]
        if any(indicator in query_lower for indicator in simple_query_indicators):
            if "vector" in reliable_strategies:
                return "vector"
        
        # For complex multi-step queries, prefer planner
        complex_indicators = ["first", "then", "analyze", "compare", "breakdown", "step"]
        if any(indicator in query_lower for indicator in complex_indicators):
            if "planner" in reliable_strategies:
                return "planner"
            elif "recursive" in reliable_strategies:
                return "recursive"
        
        # For comparison queries, prefer hybrid
        comparison_indicators = ["compare", "versus", "vs", "difference", "similar"]
        if any(indicator in query_lower for indicator in comparison_indicators):
            if "hybrid" in reliable_strategies:
                return "hybrid"
        
        # Default to most reliable available strategy
        reliable_strategies.sort(key=lambda s: strategy_priority.get(s, 10))
        selected = reliable_strategies[0]
        
        # Avoid metadata strategy for compensation queries due to empty results issue
        if selected == "metadata" and "compensation" in query_lower and "vector" in reliable_strategies:
            selected = "vector"
        
        return selected
    
    def _select_strategy_round_robin(self, index_name: str, available_strategies: List[str]) -> Dict[str, Any]:
        """Select strategy using round-robin."""
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
    
    def _select_strategy_llm_with_reasoning(self, index: str, query: str, available_strategies: List[str]) -> Dict[str, Any]:
        """
        Select strategy using LLM with detailed reasoning and confidence scoring.
        
        Args:
            index: The selected index name
            query: The user query
            available_strategies: List of available strategies for this index
            
        Returns:
            Dict with strategy, confidence, method, and reasoning
        """
        # Strategy descriptions for LLM understanding
        strategy_descriptions = {
            "vector": "Semantic similarity search using embeddings. Best for: finding conceptually similar content, general queries, when you need fast and reliable results.",
            "hybrid": "Combines semantic search with keyword matching. Best for: queries that need both conceptual and exact keyword matches, comprehensive search results.",
            "recursive": "Hierarchical retrieval that can drill down from summaries to details. Best for: complex queries requiring multi-level information, when you need to explore document structure.",
            "chunk_decoupling": "Separates chunk retrieval from context synthesis. Best for: detailed analysis, when you need precise chunk-level information with broader context.",
            "planner": "Multi-step query planning and execution. Best for: complex multi-part questions, analytical tasks requiring step-by-step reasoning.",
            "metadata": "Filters documents based on metadata attributes. Best for: queries about specific document types, dates, categories, or structured attributes.",
            "summary": "Retrieves from document summaries first. Best for: overview questions, when you need high-level information about documents."
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
        
        # Create enhanced LLM prompt for strategy selection with reasoning
        prompt = f"""You are an expert retrieval system that selects the best strategy for answering user queries.

Query: "{query}"
Index: {index}

Available retrieval strategies:
{chr(10).join(available_descriptions)}

Analyze the query and provide your response in this exact format:
STRATEGY: [strategy_name]
CONFIDENCE: [0.1-1.0]
REASONING: [brief explanation of why this strategy is best]

Consider:
1. Query complexity (simple vs multi-part)
2. Information type needed (overview vs specific details)  
3. Search requirements (semantic vs keyword vs structured)
4. Performance vs accuracy trade-offs

Example response:
STRATEGY: vector
CONFIDENCE: 0.9
REASONING: Simple semantic query requiring fast, reliable similarity search"""

        try:
            response = Settings.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse LLM response
            strategy = None
            confidence = 0.5
            reasoning = "LLM selection"
            
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
            # Fallback to heuristic selection on error
            fallback_strategy = self._select_strategy_fallback(available_strategies, query)
            return {
                "strategy": fallback_strategy,
                "confidence": 0.2,
                "method": "llm_error_fallback",
                "reasoning": f"LLM error ({str(e)}), using heuristic fallback: {fallback_strategy}"
            }
    
    def _select_strategy(self, query: str, index_name: str) -> Dict[str, Any]:
        """Select the best retrieval strategy for the query and index."""
        available_strategies = list(self.retrievers.get(index_name, {}).keys())
        
        if not available_strategies:
            return {
                "strategy": self._default_strategy,
                "confidence": 0.1,
                "method": "no_strategies",
                "reasoning": f"No strategies available for index {index_name}"
            }
        
        if self.strategy_selector == "llm":
            # Choose LLM strategy selection mode
            if self.llm_strategy_mode == "enhanced":
                result = self._select_strategy_llm_with_reasoning(index_name, query, available_strategies)
            else:  # simple mode
                selected_strategy = self._select_strategy_llm(index_name, query, available_strategies)
                result = {
                    "strategy": selected_strategy,
                    "confidence": 0.8,
                    "method": "llm_simple",
                    "reasoning": f"LLM selected {selected_strategy} using simple mode"
                }
            
            # Log strategy selection for debugging
            if hasattr(self, '_debug_logging') and self._debug_logging:
                print(f"🎯 Strategy Selection: {result['method']} -> {result['strategy']} "
                      f"(confidence: {result['confidence']:.2f})")
                print(f"   Reasoning: {result['reasoning']}")
            
            return result
            
        elif self.strategy_selector == "round_robin":
            return self._select_strategy_round_robin(index_name, available_strategies)
        else:  # default
            default_strategy = available_strategies[0] if available_strategies else self._default_strategy
            return {
                "strategy": default_strategy,
                "confidence": 0.5,
                "method": "default",
                "reasoning": f"Using default strategy: {default_strategy}"
            }
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Internal retrieve method."""
        query = query_bundle.query_str
        start_time = time.time()
        
        try:
            # Step 1: Classify the query to select index
            index_result = self.index_classifier.classify_query(query)
            selected_index = index_result["selected_index"]
            index_confidence = index_result.get("confidence", 0.0)
            
            # Step 2: Select retrieval strategy
            strategy_result = self._select_strategy(query, selected_index)
            selected_strategy = strategy_result["strategy"]
            strategy_confidence = strategy_result.get("confidence", 0.0)
            
            # Step 3: Get the appropriate retriever
            if (selected_index not in self.retrievers or 
                selected_strategy not in self.retrievers[selected_index]):
                # Fallback to any available retriever
                for idx, strategies in self.retrievers.items():
                    if strategies:
                        selected_index = idx
                        selected_strategy = list(strategies.keys())[0]
                        break
                else:
                    raise ValueError("No retrievers available")
            
            retriever = self.retrievers[selected_index][selected_strategy]
            
            # Step 4: Perform retrieval
            nodes = retriever.retrieve(query)
              # Step 5: Add routing metadata to nodes
            for node in nodes:
                if hasattr(node.node, 'metadata'):
                    node.node.metadata.update({
                        'selected_index': selected_index,
                        'selected_strategy': selected_strategy,
                        'index_confidence': index_confidence,
                        'strategy_confidence': strategy_confidence,
                        'router_method': 'agentic'
                    })
            
            # Store routing info for CLI access
            self.last_routing_info = {
                'selected_index': selected_index,
                'selected_strategy': selected_strategy,
                'index_confidence': index_confidence,
                'strategy_confidence': strategy_confidence
            }
            
            # Step 6: Log the retrieval
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            log_retrieval_call(
                query=query,
                selected_index=selected_index,
                selected_strategy=selected_strategy,
                latency_ms=latency_ms,
                confidence=min(index_confidence, strategy_confidence)
            )
            
            return nodes
            
        except Exception as e:
            # Log error
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            log_retrieval_call(
                query=query,
                selected_index="error",
                selected_strategy="error",
                latency_ms=latency_ms,
                error=str(e)
            )
            
            raise
    
    def add_retriever(self, index_name: str, strategy_name: str, retriever: BaseRetriever):
        """Add a new retriever to the router."""
        if index_name not in self.retrievers:
            self.retrievers[index_name] = {}
        self.retrievers[index_name][strategy_name] = retriever
    
    def remove_retriever(self, index_name: str, strategy_name: str):
        """Remove a retriever from the router."""
        if (index_name in self.retrievers and 
            strategy_name in self.retrievers[index_name]):
            del self.retrievers[index_name][strategy_name]
            
            # Clean up empty index
            if not self.retrievers[index_name]:
                del self.retrievers[index_name]
    
    def get_available_retrievers(self) -> Dict[str, List[str]]:
        """Get all available retrievers by index and strategy."""
        return {
            index: list(strategies.keys())
            for index, strategies in self.retrievers.items()
        }
    
    @classmethod
    def from_retrievers(
        cls,
        retrievers: Dict[str, Dict[str, BaseRetriever]],
        api_key: Optional[str] = None,
        strategy_selector: str = "llm",
        llm_strategy_mode: str = "enhanced"
    ) -> "RouterRetriever":
        """
        Create router from a dictionary of retrievers.
        
        Args:
            retrievers: Nested dict {index_name: {strategy_name: retriever}}
            api_key: OpenAI API key
            strategy_selector: Strategy selection method ("llm", "round_robin", "default")
            llm_strategy_mode: LLM strategy selection mode ("enhanced", "simple")
            
        Returns:
            RouterRetriever instance
        """
        index_classifier = create_default_classifier(api_key)
        
        return cls(
            retrievers=retrievers,
            index_classifier=index_classifier,
            strategy_selector=strategy_selector,
            llm_strategy_mode=llm_strategy_mode,
            api_key=api_key
        )

    def enable_debug_logging(self, enabled: bool = True):
        """Enable or disable debug logging for strategy selection."""
        self._debug_logging = enabled


class LlamaCloudCompositeRetriever:
    """
    Cloud-based composite retriever for LlamaCloud integration.
    This is a placeholder for future LlamaCloud integration.
    """
    
    def __init__(self, mode: str = "ROUTED"):
        """
        Initialize cloud composite retriever.
        
        Args:
            mode: Retrieval mode (ROUTED, ENSEMBLE, etc.)
        """
        self.mode = mode
        # TODO: Implement LlamaCloud integration
        raise NotImplementedError("LlamaCloud integration not yet implemented")


# Note: The RouterRetriever is designed to be initialized with pre-built retrievers.
# For the complete implementation with all 7 retrieval strategies, see:
# - cli.py: create_strategy_adapters_optimized() - Creates all strategy adapters
# - cli.py: setup_agentic_retriever() - Builds the complete router with all strategies
#
# The RouterRetriever.from_retrievers() classmethod is the recommended way to create
# a router instance with your custom set of retrievers.