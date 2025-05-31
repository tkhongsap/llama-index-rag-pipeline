"""
Hybrid Retriever Adapter

Wraps the hybrid search combining vector and keyword from 16_hybrid_search.py
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.schema import NodeWithScore, TextNode

from .base import BaseRetrieverAdapter


class HybridRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for hybrid search combining vector and keyword retrieval."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever adapter.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
        """
        super().__init__("hybrid")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # Import and create the ensemble retriever
        from hybrid_search import EnsembleRetriever
        self.retriever = EnsembleRetriever(
            embeddings=embeddings,
            api_key=api_key,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid search strategy.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.default_top_k
        
        # Perform hybrid retrieval
        result = self.retriever.ensemble_retrieve(
            query=query,
            top_k=k
        )
        
        # Convert retrieved chunks to NodeWithScore objects
        nodes = []
        for chunk_info in result.get('retrieved_chunks', []):
            # Create TextNode from chunk info
            text_node = TextNode(
                text=chunk_info.get('text', ''),
                metadata=chunk_info.get('metadata', {})
            )
            
            # Create NodeWithScore
            node_with_score = NodeWithScore(
                node=text_node,
                score=chunk_info.get('score', 0.0)
            )
            nodes.append(node_with_score)
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
    def set_weights(self, vector_weight: float, keyword_weight: float):
        """Update the weights for vector and keyword search."""
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.retriever.set_weights(vector_weight, keyword_weight)
    
    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> "HybridRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            
        Returns:
            HybridRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        ) 