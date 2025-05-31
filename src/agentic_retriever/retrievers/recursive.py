"""
Recursive Retriever Adapter

Wraps the recursive retrieval from 12_recursive_retriever.py
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.schema import NodeWithScore, TextNode

from .base import BaseRetrieverAdapter


class RecursiveRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for recursive retrieval with parent-child relationships."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize recursive retriever adapter.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            chunk_size: Size of chunks for recursive splitting
            chunk_overlap: Overlap between chunks
        """
        super().__init__("recursive")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Import and create the recursive retriever
        from recursive_retriever import RecursiveDocumentRetriever
        self.retriever = RecursiveDocumentRetriever(
            embeddings=embeddings,
            api_key=api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using recursive strategy.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.default_top_k
        
        # Perform recursive retrieval
        result = self.retriever.recursive_retrieve(
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
    
    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> "RecursiveRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            chunk_size: Size of chunks for recursive splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            RecursiveRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        ) 