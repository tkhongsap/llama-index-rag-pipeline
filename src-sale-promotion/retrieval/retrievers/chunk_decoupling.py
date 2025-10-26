"""
Chunk Decoupling Retriever Adapter for Sales Promotions

Strategy 6: Context-preserving retrieval for precise answer extraction.
Adapted for atomic documents (no chunks).

Performance Target: < 200ms
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .base import BaseRetrieverAdapter


class ChunkDecouplingRetrieverAdapter(BaseRetrieverAdapter):
    """
    Chunk decoupling retriever adapted for atomic promotion documents.
    
    Since sales promotions are atomic (50-150 tokens, no chunking),
    this strategy simplifies to retrieving full documents with
    emphasis on precise answer extraction.
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        top_k: int = 3
    ):
        """
        Initialize chunk decoupling retriever.
        
        Args:
            vector_index: VectorStoreIndex for retrieval
            top_k: Number of results to return
        """
        super().__init__("chunk_decoupling")
        self.vector_index = vector_index
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes with focus on precise answer extraction.
        
        For atomic documents, this is similar to vector retrieval
        but optimized for "what is..." type questions.
        
        Args:
            query: Search query (supports Thai text)
            top_k: Number of results (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy metadata
        """
        k = top_k if top_k is not None else self.top_k
        
        # For atomic docs, retrieve full promotion documents
        retriever = self.vector_index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        
        # Tag nodes with strategy info
        return self._tag_nodes_with_strategy(nodes)
