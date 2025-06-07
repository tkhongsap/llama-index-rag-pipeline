"""
Vector Retriever Adapter for iLand Data

Wraps the basic vector similarity search for Thai land deed embeddings.
Adapted from src/agentic_retriever/retrievers/vector.py for iLand data.
"""

import sys
from pathlib import Path
from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex

from .base import BaseRetrieverAdapter

# Import from updated iLand embedding loading modules
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from load_embedding import iLandEmbeddingLoader, iLandIndexReconstructor, EmbeddingConfig
except ImportError as e:
    print(f"Warning: Could not import iLand embedding utilities: {e}")
    iLandEmbeddingLoader = None
    iLandIndexReconstructor = None
    EmbeddingConfig = None


class VectorRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for basic vector similarity search on iLand data."""
    
    def __init__(self, index: VectorStoreIndex, default_top_k: int = 5):
        """
        Initialize vector retriever adapter for iLand data.
        
        Args:
            index: The vector store index to use (created from iLand embeddings)
            default_top_k: Default number of nodes to retrieve
        """
        super().__init__("vector")
        self.index = index
        self.default_top_k = default_top_k
        self.retriever = self.index.as_retriever(similarity_top_k=default_top_k)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using vector similarity search on iLand data.
        
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
            self.retriever = self.index.as_retriever(similarity_top_k=k)
        
        # Retrieve nodes
        nodes = self.retriever.retrieve(query)
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
    @classmethod
    def from_iland_embeddings(
        cls, 
        embeddings: List[dict], 
        api_key: Optional[str] = None,
        default_top_k: int = 5
    ) -> "VectorRetrieverAdapter":
        """
        Create adapter from iLand embedding data.
        
        Args:
            embeddings: List of iLand embedding dictionaries
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            
        Returns:
            VectorRetrieverAdapter instance for iLand data
        """
        if not iLandIndexReconstructor or not EmbeddingConfig:
            raise ImportError("iLand embedding utilities not available")
        
        config = EmbeddingConfig(api_key=api_key)
        reconstructor = iLandIndexReconstructor(config=config)
        index = reconstructor.create_vector_index_from_embeddings(
            embeddings, 
            show_progress=False
        )
        return cls(index, default_top_k)
    
    @classmethod
    def from_latest_iland_batch(
        cls,
        api_key: Optional[str] = None,
        default_top_k: int = 5
    ) -> "VectorRetrieverAdapter":
        """
        Create adapter from latest iLand embedding batch.
        
        Args:
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            
        Returns:
            VectorRetrieverAdapter instance for latest iLand batch
        """
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from load_embedding.utils import create_iland_index_from_latest_batch
        index = create_iland_index_from_latest_batch(api_key)
        return cls(index, default_top_k) 