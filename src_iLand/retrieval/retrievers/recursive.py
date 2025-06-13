"""
Recursive Retriever Adapter for iLand Data

Implements hierarchical recursive retrieval for Thai land deed data with parent-child relationships.
Adapted from src/agentic_retriever/retrievers/recursive.py for iLand data.
"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever

from .base import BaseRetrieverAdapter


class RecursiveRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for recursive retrieval on iLand data."""
    
    def __init__(self, recursive_retriever: RecursiveRetriever, default_top_k: int = 5):
        """
        Initialize recursive retriever adapter for iLand data.
        
        Args:
            recursive_retriever: Pre-configured recursive retriever
            default_top_k: Default number of nodes to retrieve
        """
        super().__init__("recursive")
        self.recursive_retriever = recursive_retriever
        self.default_top_k = default_top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using recursive approach on iLand data.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Note: RecursiveRetriever doesn't directly support top_k in retrieve()
        # We'll retrieve and then truncate if needed
        nodes = self.recursive_retriever.retrieve(query)
        
        # Apply top_k limit if specified
        k = top_k if top_k is not None else self.default_top_k
        if k < len(nodes):
            nodes = nodes[:k]
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
    @classmethod
    def from_iland_indices(
        cls,
        leaf_index: VectorStoreIndex,
        parent_index: VectorStoreIndex,
        default_top_k: int = 5
    ) -> "RecursiveRetrieverAdapter":
        """
        Create adapter from iLand parent-child indices.
        
        Args:
            leaf_index: Vector index for leaf nodes (detailed chunks)
            parent_index: Vector index for parent nodes (summaries)
            default_top_k: Default number of nodes to retrieve
            
        Returns:
            RecursiveRetrieverAdapter instance for iLand data
        """
        # Create retrievers for each level
        leaf_retriever = leaf_index.as_retriever(similarity_top_k=default_top_k)
        
        # Create recursive retriever
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={
                "vector": leaf_retriever
            },
            query_engine_dict={},
            verbose=False
        )
        
        return cls(recursive_retriever, default_top_k) 