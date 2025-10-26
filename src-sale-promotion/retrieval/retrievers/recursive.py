"""
Recursive Retriever for hierarchical campaign structures.
"""

from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from .base import BaseRetrieverAdapter


class RecursiveRetrieverAdapter(BaseRetrieverAdapter):
    """
    Recursive retriever for hierarchical campaign/product structures.
    
    Best for:
    - Multi-level campaigns
    - Product bundles
    - Campaign hierarchies
    
    Example queries:
    - "Show Q4 holiday campaign hierarchy"
    - "Break down multi-channel campaign components"
    - "What products are in the summer sale bundle?"
    """
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using recursive strategy."""
        if not self.index:
            return []
        
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query_bundle.query_str)
        return self._enhance_metadata(nodes)
