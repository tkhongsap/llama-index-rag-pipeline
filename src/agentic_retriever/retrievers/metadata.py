"""
Metadata Retriever Adapter

Wraps the metadata-filtered retrieval from 14_metadata_filtering.py
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilters

from .base import BaseRetrieverAdapter


class MetadataRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for metadata-filtered retrieval."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        metadata_filters: Optional[MetadataFilters] = None
    ):
        """
        Initialize metadata retriever adapter.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            metadata_filters: Default metadata filters to apply
        """
        super().__init__("metadata")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.metadata_filters = metadata_filters
        
        # Import and create the metadata retriever
        from metadata_filtering import MetadataFilteredRetriever
        self.retriever = MetadataFilteredRetriever(
            embeddings=embeddings,
            api_key=api_key
        )
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        metadata_filters: Optional[MetadataFilters] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using metadata filtering.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (uses default if None)
            metadata_filters: Metadata filters to apply (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided parameters or defaults
        k = top_k if top_k is not None else self.default_top_k
        filters = metadata_filters if metadata_filters is not None else self.metadata_filters
        
        # Perform metadata-filtered retrieval
        result = self.retriever.filtered_retrieve(
            query=query,
            top_k=k,
            metadata_filters=filters
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
    
    def set_metadata_filters(self, metadata_filters: MetadataFilters):
        """Set default metadata filters."""
        self.metadata_filters = metadata_filters
    
    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        metadata_filters: Optional[MetadataFilters] = None
    ) -> "MetadataRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            metadata_filters: Default metadata filters to apply
            
        Returns:
            MetadataRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            metadata_filters=metadata_filters
        ) 