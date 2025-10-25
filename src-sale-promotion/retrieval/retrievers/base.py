"""
Base retriever adapter for sales promotion pipeline.
"""

from typing import List, Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever


class BaseRetrieverAdapter(BaseRetriever):
    """
    Base adapter for all retrieval strategies.

    Provides common functionality for sales promotion retrievers.
    """

    def __init__(
        self,
        index=None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize base retriever adapter.

        Args:
            index: LlamaIndex VectorStoreIndex
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            **kwargs: Additional strategy-specific parameters
        """
        super().__init__()
        self.index = index
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.kwargs = kwargs

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Internal retrieve method to be implemented by subclasses.

        Args:
            query_bundle: Query bundle with query string

        Returns:
            List of nodes with scores
        """
        raise NotImplementedError("Subclasses must implement _retrieve()")

    def _filter_by_threshold(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Filter nodes by similarity threshold."""
        if self.similarity_threshold > 0:
            return [node for node in nodes if node.score >= self.similarity_threshold]
        return nodes

    def _enhance_metadata(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Add sales promotion-specific metadata enhancements."""
        for node in nodes:
            if hasattr(node.node, 'metadata'):
                # Add retrieval metadata
                node.node.metadata['retriever_type'] = self.__class__.__name__
                node.node.metadata['similarity_score'] = node.score
        return nodes
