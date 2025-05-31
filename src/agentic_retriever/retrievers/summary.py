"""
Summary Retriever Adapter

Wraps the document summary-first retrieval from 11_document_summary_retriever.py
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.schema import NodeWithScore, TextNode

from .base import BaseRetrieverAdapter


class SummaryRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for document summary-first retrieval."""
    
    def __init__(
        self, 
        summary_embeddings: List[dict],
        chunk_embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        chunks_per_doc: int = 3
    ):
        """
        Initialize summary retriever adapter.
        
        Args:
            summary_embeddings: Document summary embeddings
            chunk_embeddings: Chunk embeddings
            api_key: OpenAI API key
            default_top_k: Default number of documents to retrieve
            chunks_per_doc: Number of chunks to retrieve per document
        """
        super().__init__("summary")
        self.summary_embeddings = summary_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.chunks_per_doc = chunks_per_doc
        
        # Import and create the document summary retriever
        from document_summary_retriever import DocumentSummaryRetriever
        self.retriever = DocumentSummaryRetriever(
            summary_embeddings=summary_embeddings,
            chunk_embeddings=chunk_embeddings,
            api_key=api_key
        )
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using summary-first strategy.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.default_top_k
        
        # Perform hierarchical retrieval
        result = self.retriever.hierarchical_retrieve(
            query=query,
            summary_top_k=k,
            chunks_per_doc=self.chunks_per_doc
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
    
    @classmethod
    def from_embeddings(
        cls,
        summary_embeddings: List[dict],
        chunk_embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        chunks_per_doc: int = 3
    ) -> "SummaryRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            summary_embeddings: Document summary embeddings
            chunk_embeddings: Chunk embeddings
            api_key: OpenAI API key
            default_top_k: Default number of documents to retrieve
            chunks_per_doc: Number of chunks to retrieve per document
            
        Returns:
            SummaryRetrieverAdapter instance
        """
        return cls(
            summary_embeddings=summary_embeddings,
            chunk_embeddings=chunk_embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            chunks_per_doc=chunks_per_doc
        ) 