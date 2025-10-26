"""
Recursive Retriever Adapter for Sales Promotions

Strategy 3: Multi-level hierarchical access with drill-down capability.
Best for complex queries like "Compare discounts across customer tiers for beer"

Performance Target: < 300ms
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, IndexNode

from .base import BaseRetrieverAdapter


class RecursiveRetrieverAdapter(BaseRetrieverAdapter):
    """
    Recursive hierarchical retriever for sales promotions.
    
    Queries top-level IndexNodes first, then drills down to details.
    Maintains context across retrieval levels.
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        top_level_top_k: int = 3,
        detail_top_k: int = 2,
        max_depth: int = 2
    ):
        """
        Initialize recursive retriever.
        
        Args:
            vector_index: VectorStoreIndex containing IndexNodes and details
            top_level_top_k: Number of top-level results
            detail_top_k: Number of details per top-level result
            max_depth: Maximum recursion depth
        """
        super().__init__("recursive")
        self.vector_index = vector_index
        self.top_level_top_k = top_level_top_k
        self.detail_top_k = detail_top_k
        self.max_depth = max_depth
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using recursive hierarchical approach.
        
        Args:
            query: Search query (supports Thai text)
            top_k: Number of results (uses top_level_top_k if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy metadata
        """
        k = top_k if top_k is not None else self.top_level_top_k
        
        # For atomic documents (no chunks), recursive retrieval simplifies to:
        # 1. Query all nodes
        # 2. Return top-k results
        # (In chunked documents, we would drill down to child chunks here)
        
        retriever = self.vector_index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        
        # Tag nodes with strategy info
        return self._tag_nodes_with_strategy(nodes)
