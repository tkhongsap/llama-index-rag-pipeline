"""
Summary Retriever Adapter for iLand Data

Implements document summary-first retrieval for Thai land deed data.
Adapted from src/agentic_retriever/retrievers/summary.py for iLand data.
"""

from typing import List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from .base import BaseRetrieverAdapter


class SummaryRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for summary-first retrieval on iLand data."""
    
    def __init__(self, summary_index: VectorStoreIndex, default_top_k: int = 5):
        """
        Initialize summary retriever adapter for iLand data.
        
        Args:
            summary_index: Vector index of document summaries
            default_top_k: Default number of nodes to retrieve
        """
        super().__init__("summary")
        self.summary_index = summary_index
        self.default_top_k = default_top_k
        self.retriever = self.summary_index.as_retriever(similarity_top_k=default_top_k)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using summary-first approach on iLand data.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.default_top_k
        
        # Update retriever if top_k is different
        if k != self.retriever.similarity_top_k:
            self.retriever = self.summary_index.as_retriever(similarity_top_k=k)
        
        # Retrieve from summary index first
        nodes = self.retriever.retrieve(query)
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
    @classmethod
    def from_iland_embeddings(
        cls, 
        summary_embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5
    ) -> "SummaryRetrieverAdapter":
        """
        Create adapter from iLand summary embeddings.
        
        Args:
            summary_embeddings: List of iLand summary embedding dictionaries
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            
        Returns:
            SummaryRetrieverAdapter instance for iLand data
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from load_embedding import iLandIndexReconstructor
        
        from load_embedding.models import EmbeddingConfig
        config = EmbeddingConfig(api_key=api_key)
        reconstructor = iLandIndexReconstructor(config=config)
        summary_index = reconstructor.create_vector_index_from_embeddings(
            summary_embeddings, 
            show_progress=False
        )
        return cls(summary_index, default_top_k) 