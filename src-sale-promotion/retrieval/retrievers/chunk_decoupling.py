"""
Chunk Decoupling Retriever for detailed analysis.
"""

from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from .base import BaseRetrieverAdapter


class ChunkDecouplingRetrieverAdapter(BaseRetrieverAdapter):
    """
    Chunk decoupling retriever for precise context.
    
    Best for:
    - Detailed analysis
    - Specific information extraction
    - Context-aware retrieval
    
    Example queries:
    - "Detailed breakdown of campaign budget allocation"
    - "Explain the pricing strategy for product X"
    - "What were the exact terms of the promotion?"
    """
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using chunk decoupling."""
        if not self.index:
            return []
        
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query_bundle.query_str)
        return self._enhance_metadata(nodes)
