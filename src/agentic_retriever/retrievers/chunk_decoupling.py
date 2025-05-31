"""
Chunk Decoupling Retriever Adapter

Wraps the chunk decoupling strategy from 15_chunk_decoupling.py
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.schema import NodeWithScore, TextNode

from .base import BaseRetrieverAdapter


class ChunkDecouplingRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for chunk decoupling retrieval strategy."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        window_size: int = 3,
        sentence_window_size: int = 1
    ):
        """
        Initialize chunk decoupling retriever adapter.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            window_size: Size of the context window
            sentence_window_size: Size of sentence window
        """
        super().__init__("chunk_decoupling")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.window_size = window_size
        self.sentence_window_size = sentence_window_size
        
        # Import and create the sentence window retriever
        from chunk_decoupling import SentenceWindowRetriever
        self.retriever = SentenceWindowRetriever(
            embeddings=embeddings,
            api_key=api_key,
            window_size=window_size,
            sentence_window_size=sentence_window_size
        )
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using chunk decoupling strategy.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.default_top_k
        
        # Perform sentence window retrieval
        result = self.retriever.sentence_window_retrieve(
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
        window_size: int = 3,
        sentence_window_size: int = 1
    ) -> "ChunkDecouplingRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            window_size: Size of the context window
            sentence_window_size: Size of sentence window
            
        Returns:
            ChunkDecouplingRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            window_size=window_size,
            sentence_window_size=sentence_window_size
        ) 