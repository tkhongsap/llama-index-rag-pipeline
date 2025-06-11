"""
PostgreSQL Router Retriever for iLand Data

Intelligent router that selects the optimal PostgreSQL retrieval strategy
based on query analysis and available strategies.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from .config import PostgresConfig
from .retrievers import (
    BasicPostgresRetriever,
    SentenceWindowPostgresRetriever,
    RecursivePostgresRetriever,
    MetadataFilterPostgresRetriever
)

logger = logging.getLogger(__name__)


class PostgresRouterRetriever:
    """
    Intelligent router for PostgreSQL-based retrieval strategies.
    
    Analyzes queries and automatically selects the best retrieval strategy
    for Thai land deed data stored in PostgreSQL.
    """
    
    def __init__(
        self,
        config: PostgresConfig,
        strategy_selector: str = "llm",
        llm_strategy_mode: str = "enhanced"
    ):
        """
        Initialize PostgreSQL router retriever.
        
        Args:
            config: PostgreSQL configuration
            strategy_selector: Strategy selection method ("llm", "heuristic", "round_robin")
            llm_strategy_mode: LLM strategy mode ("enhanced", "simple")
        """
        self.config = config
        self.strategy_selector = strategy_selector
        self.llm_strategy_mode = llm_strategy_mode
        
        # Initialize retrievers
        self.retrievers = self._initialize_retrievers()
        
        # Setup LLM for strategy selection
        if strategy_selector == "llm":
            self._setup_llm()
        
        # Round-robin state
        self._round_robin_state = 0
        
        logger.info(f"Initialized PostgreSQL router with {len(self.retrievers)} strategies")
    
    def _initialize_retrievers(self) -> Dict[str, Any]:
        """Initialize all available retrieval strategies."""
        retrievers = {}
        
        try:
            # Basic vector similarity
            retrievers["basic"] = BasicPostgresRetriever(self.config)
            
            # Sentence window with context
            retrievers["window"] = SentenceWindowPostgresRetriever(self.config)
            
            # Recursive hierarchical search
            retrievers["recursive"] = RecursivePostgresRetriever(self.config)
            
            # Metadata-aware filtering
            retrievers["metadata"] = MetadataFilterPostgresRetriever(self.config)
            
            logger.info(f"Successfully initialized {len(retrievers)} retrieval strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize some retrievers: {e}")
        
        return retrievers
    
    def _setup_llm(self):
        """Setup LLM for strategy selection."""
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key required for LLM-based strategy selection")
        
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.config.openai_api_key
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using intelligent strategy selection.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve
            filters: Metadata filters
            strategy: Force specific strategy (overrides auto-selection)
            
        Returns:
            List of NodeWithScore objects from selected strategy
        """
        start_time = time.time()
        
        top_k = top_k or self.config.default_top_k
        
        try:
            # Select strategy
            if strategy and strategy in self.retrievers:
                selected_strategy = strategy
                selection_method = "manual"
                confidence = 1.0
            else:
                strategy_result = self._select_strategy(query, filters)
                selected_strategy = strategy_result["strategy"]
                selection_method = strategy_result["method"]
                confidence = strategy_result["confidence"]
            
            # Get retriever
            retriever = self.retrievers.get(selected_strategy)
            if not retriever:
                logger.warning(f"Strategy {selected_strategy} not available, falling back to basic")
                retriever = self.retrievers["basic"]
                selected_strategy = "basic"
            
            # Execute retrieval
            nodes = retriever.retrieve(query, top_k, filters)
            
            # Add routing metadata to nodes
            for node in nodes:
                if hasattr(node.node, 'metadata'):
                    node.node.metadata.update({
                        "selected_strategy": selected_strategy,
                        "selection_method": selection_method,
                        "selection_confidence": confidence,
                        "router_type": "postgres_router"
                    })
            
            # Log routing decision
            execution_time = time.time() - start_time
            logger.info(
                f"PostgreSQL Router: {query[:50]}... -> {selected_strategy} "
                f"({len(nodes)} results, {execution_time:.3f}s, confidence={confidence:.2f})"
            )
            
            return nodes
            
        except Exception as e:
            logger.error(f"PostgreSQL router retrieval failed: {e}")
            raise
    
    def _select_strategy(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select the best retrieval strategy for the query.
        
        Args:
            query: The search query
            filters: Metadata filters
            
        Returns:
            Dictionary with strategy selection results
        """
        if self.strategy_selector == "llm":
            return self._select_strategy_llm(query, filters)
        elif self.strategy_selector == "round_robin":
            return self._select_strategy_round_robin()
        else:  # heuristic
            return self._select_strategy_heuristic(query, filters)
    
    def _select_strategy_llm(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select strategy using LLM analysis.
        
        Args:
            query: The search query
            filters: Metadata filters
            
        Returns:
            Strategy selection results
        """
        available_strategies = list(self.retrievers.keys())
        
        # Strategy descriptions for Thai land deed data
        strategy_descriptions = {
            "basic": "Direct semantic similarity search. Best for: simple queries about Thai land deeds, fast retrieval, general property information searches.",
            "window": "Semantic search with surrounding context. Best for: queries needing broader context understanding, detailed Thai legal document analysis.",
            "recursive": "Hierarchical search from summaries to details. Best for: complex multi-level queries, exploring document structure, comprehensive property analysis.",
            "metadata": "Metadata-aware filtering with geographic/legal attributes. Best for: queries about specific provinces (จังหวัด), deed types (โฉนด, นส.3), structured property searches."
        }
        
        # Build prompt
        available_descriptions = []
        for strategy in available_strategies:
            if strategy in strategy_descriptions:
                available_descriptions.append(f"- {strategy}: {strategy_descriptions[strategy]}")
        
        filter_info = ""
        if filters:
            filter_info = f"\nMetadata filters provided: {list(filters.keys())}"
        
        prompt = f"""You are an expert retrieval system for Thai land deed data that selects the best PostgreSQL-based strategy.

Query: "{query}"{filter_info}

Available retrieval strategies:
{chr(10).join(available_descriptions)}

Analyze the query and provide your response in this exact format:
STRATEGY: [strategy_name]
CONFIDENCE: [0.1-1.0]
REASONING: [brief explanation for Thai land deed context]

Consider:
1. Query complexity and Thai language content
2. Need for context vs precision
3. Geographic/legal filtering requirements
4. Document structure navigation needs
5. Metadata filter usage"""

        try:
            response = Settings.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse response
            strategy = None
            confidence = 0.5
            reasoning = "LLM selection for Thai land deed data"
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('STRATEGY:'):
                    strategy = line.split(':', 1)[1].strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        confidence = max(0.1, min(1.0, confidence))
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Validate strategy
            if strategy and strategy in available_strategies:
                return {
                    "strategy": strategy,
                    "confidence": confidence,
                    "method": "llm_enhanced",
                    "reasoning": reasoning
                }
            
            # Fallback to heuristic
            fallback = self._select_strategy_heuristic(query, filters)
            fallback["method"] = "llm_fallback"
            return fallback
            
        except Exception as e:
            logger.warning(f"LLM strategy selection failed: {e}")
            fallback = self._select_strategy_heuristic(query, filters)
            fallback["method"] = "llm_error_fallback"
            return fallback
    
    def _select_strategy_heuristic(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select strategy using heuristic rules for Thai land deed data.
        
        Args:
            query: The search query
            filters: Metadata filters
            
        Returns:
            Strategy selection results
        """
        query_lower = query.lower()
        
        # If metadata filters provided, prefer metadata strategy
        if filters and len(filters) > 0:
            return {
                "strategy": "metadata",
                "confidence": 0.8,
                "method": "heuristic_metadata",
                "reasoning": "Metadata filters provided"
            }
        
        # Check for Thai geographic terms
        thai_geo_terms = ["จังหวัด", "อำเภอ", "ตำบล", "กรุงเทพ", "สมุทรปราการ"]
        if any(term in query for term in thai_geo_terms):
            return {
                "strategy": "metadata",
                "confidence": 0.7,
                "method": "heuristic_geographic",
                "reasoning": "Geographic Thai terms detected"
            }
        
        # Check for deed type terms
        deed_terms = ["โฉนด", "นส.3", "นส.4", "ส.ค.1"]
        if any(term in query for term in deed_terms):
            return {
                "strategy": "metadata",
                "confidence": 0.7,
                "method": "heuristic_deed_type",
                "reasoning": "Thai deed type terms detected"
            }
        
        # Check for context-requiring terms
        context_terms = ["explain", "context", "surrounding", "detail", "อธิบาย", "รายละเอียด"]
        if any(term in query_lower for term in context_terms):
            return {
                "strategy": "window",
                "confidence": 0.6,
                "method": "heuristic_context",
                "reasoning": "Context-requiring query detected"
            }
        
        # Check for hierarchical terms
        hierarchical_terms = ["overview", "summary", "breakdown", "structure", "สรุป", "โครงสร้าง"]
        if any(term in query_lower for term in hierarchical_terms):
            return {
                "strategy": "recursive",
                "confidence": 0.6,
                "method": "heuristic_hierarchical",
                "reasoning": "Hierarchical query detected"
            }
        
        # Default to basic for simple queries
        return {
            "strategy": "basic",
            "confidence": 0.5,
            "method": "heuristic_default",
            "reasoning": "Simple semantic query, using basic strategy"
        }
    
    def _select_strategy_round_robin(self) -> Dict[str, Any]:
        """Select strategy using round-robin for testing."""
        available_strategies = list(self.retrievers.keys())
        selected_strategy = available_strategies[self._round_robin_state % len(available_strategies)]
        self._round_robin_state += 1
        
        return {
            "strategy": selected_strategy,
            "confidence": 0.5,
            "method": "round_robin",
            "reasoning": f"Round-robin selection: {selected_strategy}"
        }
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available retrieval strategies."""
        return list(self.retrievers.keys())
    
    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available strategies."""
        info = {}
        for name, retriever in self.retrievers.items():
            info[name] = {
                "name": name,
                "class": retriever.__class__.__name__,
                "description": retriever.__class__.__doc__ or "No description available"
            }
        return info
    
    def close(self):
        """Close all retrievers and cleanup resources."""
        for retriever in self.retrievers.values():
            if hasattr(retriever, 'close'):
                retriever.close()
        logger.info("PostgreSQL router closed") 