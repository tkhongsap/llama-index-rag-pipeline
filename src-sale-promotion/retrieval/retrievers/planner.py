"""
Planner Retriever for complex multi-step queries.
"""

from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from .base import BaseRetrieverAdapter


class PlannerRetrieverAdapter(BaseRetrieverAdapter):
    """
    Planner retriever with multi-step query decomposition.
    
    Best for:
    - Complex analytical questions
    - Multi-step reasoning
    - Comparative analysis
    
    Example queries:
    - "Compare Q3 vs Q4 campaigns, analyze ROI trends, recommend best channel"
    - "Which products had highest discount but lowest revenue? Why?"
    - "Break down customer journey from impression to purchase"
    """
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using query planning."""
        if not self.index:
            return []
        
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query_bundle.query_str)
        return self._enhance_metadata(nodes)
