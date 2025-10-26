"""
Vector Retriever Adapter for Sales Promotions

Strategy 1: Basic semantic similarity search using 3072-dim embeddings.
Best for simple semantic queries like "Find beer promotions" or "Show water deals".

Performance Target: < 100ms
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .base import BaseRetrieverAdapter


class VectorRetrieverAdapter(BaseRetrieverAdapter):
    """
    Basic vector similarity retriever for sales promotions.
    
    Uses semantic search with text-embedding-3-large (3072 dimensions).
    Fast and reliable for general-purpose queries.
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        similarity_top_k: int = 5,
        similarity_cutoff: float = 0.7
    ):
        """
        Initialize vector retriever.
        
        Args:
            vector_index: VectorStoreIndex built from Phase 2 embeddings
            similarity_top_k: Number of results to return
            similarity_cutoff: Minimum similarity score threshold
        """
        super().__init__("vector")
        self.vector_index = vector_index
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using vector similarity search.
        
        Args:
            query: Search query (supports Thai text)
            top_k: Number of results (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy metadata
        """
        k = top_k if top_k is not None else self.similarity_top_k
        
        # Create retriever from index
        retriever = self.vector_index.as_retriever(
            similarity_top_k=k,
            similarity_cutoff=self.similarity_cutoff
        )
        
        # Perform retrieval
        nodes = retriever.retrieve(query)
        
        # Tag nodes with strategy info
        return self._tag_nodes_with_strategy(nodes)
