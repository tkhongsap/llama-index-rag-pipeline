"""
Router Module

Main router that combines index classification and retrieval strategy selection.
Provides both local RouterRetriever and cloud LlamaCloudCompositeRetriever.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from .index_classifier import IndexClassifier, create_default_classifier
from .retrievers import (
    VectorRetrieverAdapter,
    SummaryRetrieverAdapter,
    RecursiveRetrieverAdapter,
    MetadataRetrieverAdapter,
    ChunkDecouplingRetrieverAdapter,
    HybridRetrieverAdapter,
    PlannerRetrieverAdapter
)
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
        api_key: Optional[str] = None
    ):
        """
        Initialize the router retriever.
        
        Args:
            retrievers: Nested dict {index_name: {strategy_name: retriever}}
            index_classifier: Index classifier instance
            strategy_selector: Strategy selection method ("llm", "round_robin", "default")
            api_key: OpenAI API key
        """
        super().__init__()
        self.retrievers = retrievers
        self.index_classifier = index_classifier or create_default_classifier(api_key)
        self.strategy_selector = strategy_selector
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
    
    def _select_strategy_llm(self, query: str, available_strategies: List[str]) -> Dict[str, Any]:
        """Select strategy using LLM."""
        strategies_desc = {
            "vector": "Basic vector similarity search - good for general semantic queries",
            "summary": "Document summary-first retrieval - good for document-level questions",
            "recursive": "Recursive retrieval with parent-child relationships - good for hierarchical content",
            "metadata": "Metadata-filtered retrieval - good when specific filters are needed",
            "chunk_decoupling": "Chunk decoupling with sentence windows - good for precise context",
            "hybrid": "Hybrid vector + keyword search - good for mixed semantic/exact queries",
            "planner": "Query planning agent - good for complex multi-step queries"
        }
        
        available_desc = "\n".join([
            f"- {strategy}: {strategies_desc.get(strategy, 'Advanced retrieval strategy')}"
            for strategy in available_strategies
        ])
        
        prompt = f"""
You are an expert at selecting the best retrieval strategy for different types of queries.

Available strategies:
{available_desc}

Query: "{query}"

Based on the query characteristics, which strategy would be most effective?
Consider:
- Query complexity (simple vs multi-step)
- Information type needed (specific facts vs general overview)
- Search type (semantic similarity vs exact matches)

Respond with ONLY the strategy name (exactly as listed above).
"""
        
        try:
            response = Settings.llm.complete(prompt)
            selected_strategy = response.text.strip()
            
            if selected_strategy in available_strategies:
                return {
                    "strategy": selected_strategy,
                    "confidence": 0.9,
                    "method": "llm",
                    "reasoning": f"LLM selected {selected_strategy}"
                }
            else:
                # Fallback to default
                fallback_strategy = available_strategies[0] if available_strategies else self._default_strategy
                return {
                    "strategy": fallback_strategy,
                    "confidence": 0.3,
                    "method": "llm_fallback",
                    "reasoning": f"LLM response '{selected_strategy}' invalid, using fallback"
                }
                
        except Exception as e:
            # Fallback to default strategy
            fallback_strategy = available_strategies[0] if available_strategies else self._default_strategy
            return {
                "strategy": fallback_strategy,
                "confidence": 0.1,
                "method": "error_fallback",
                "reasoning": f"Error in LLM strategy selection: {str(e)}"
            }
    
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
            return self._select_strategy_llm(query, available_strategies)
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
        strategy_selector: str = "llm"
    ) -> "RouterRetriever":
        """
        Create router from a dictionary of retrievers.
        
        Args:
            retrievers: Nested dict {index_name: {strategy_name: retriever}}
            api_key: OpenAI API key
            strategy_selector: Strategy selection method
            
        Returns:
            RouterRetriever instance
        """
        index_classifier = create_default_classifier(api_key)
        
        return cls(
            retrievers=retrievers,
            index_classifier=index_classifier,
            strategy_selector=strategy_selector,
            api_key=api_key
        )


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


def create_default_router(
    embeddings_data: Dict[str, List[dict]],
    api_key: Optional[str] = None,
    strategy_selector: str = "llm"
) -> RouterRetriever:
    """
    Create a router with default retrievers for each strategy.
    
    Args:
        embeddings_data: Dict mapping index names to embedding data
        api_key: OpenAI API key
        strategy_selector: Strategy selection method
        
    Returns:
        RouterRetriever instance
    """
    retrievers = {}
    
    for index_name, embeddings in embeddings_data.items():
        retrievers[index_name] = {}
        
        # Create all strategy adapters for this index
        retrievers[index_name]["vector"] = VectorRetrieverAdapter.from_embeddings(
            embeddings, api_key
        )
        
        # Note: Other strategies would need additional data (summary embeddings, etc.)
        # For now, we'll just include vector retrieval
        # TODO: Add other strategies when data is available
    
    return RouterRetriever.from_retrievers(
        retrievers=retrievers,
        api_key=api_key,
        strategy_selector=strategy_selector
    ) 