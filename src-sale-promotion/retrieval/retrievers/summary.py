"""
Summary Retriever Adapter for Sales Promotions

Strategy 2: Hierarchical retrieval using GPT-5 generated summaries.
Best for document-level queries like "Which promotions have highest discounts?"

Performance Target: < 200ms
"""

from typing import List, Optional
from llama_index.core import DocumentSummaryIndex
from llama_index.core.schema import NodeWithScore

from .base import BaseRetrieverAdapter


class SummaryRetrieverAdapter(BaseRetrieverAdapter):
    """
    Summary-based hierarchical retriever for sales promotions.
    
    Uses GPT-5 generated summaries from Phase 2 for initial document-level retrieval,
    then drills down to details.
    """
    
    def __init__(
        self,
        doc_summary_index: DocumentSummaryIndex,
        summary_top_k: int = 3,
        detail_top_k: int = 2
    ):
        """
        Initialize summary retriever.
        
        Args:
            doc_summary_index: DocumentSummaryIndex from Phase 2
            summary_top_k: Number of documents to retrieve
            detail_top_k: Number of details per document
        """
        super().__init__("summary")
        self.doc_summary_index = doc_summary_index
        self.summary_top_k = summary_top_k
        self.detail_top_k = detail_top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using summary-first hierarchical approach.
        
        Args:
            query: Search query (supports Thai text)
            top_k: Number of results (uses summary_top_k if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy metadata
        """
        k = top_k if top_k is not None else self.summary_top_k
        
        # Create retriever from DocumentSummaryIndex
        retriever = self.doc_summary_index.as_retriever(
            similarity_top_k=k
        )
        
        # Perform retrieval
        nodes = retriever.retrieve(query)
        
        # Tag nodes with strategy info
        return self._tag_nodes_with_strategy(nodes)
