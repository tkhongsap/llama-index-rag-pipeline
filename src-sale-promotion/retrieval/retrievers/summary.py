"""
Summary Retriever for overview questions.
"""

from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from .base import BaseRetrieverAdapter


class SummaryRetrieverAdapter(BaseRetrieverAdapter):
    """
    Summary-based retriever for overview questions.
    
    Best for:
    - High-level overviews
    - Campaign summaries
    - Performance summaries
    
    Example queries:
    - "Summarize 2024 email marketing performance"
    - "Overview of best-performing product categories"
    - "What were the main campaigns last quarter?"
    """
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using document summaries."""
        if not self.index:
            return []
        
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query_bundle.query_str)
        return self._enhance_metadata(nodes)
