"""
Vector Retriever for sales promotion documents.

Simple semantic similarity search using vector embeddings.
Best for general queries and fast retrieval.
"""

from typing import List
from llama_index.core.schema import NodeWithScore, QueryBundle
from .base import BaseRetrieverAdapter


class VectorRetrieverAdapter(BaseRetrieverAdapter):
    """
    Vector similarity retriever for sales promotion documents.

    Uses semantic embedding similarity for retrieval.
    Fast and reliable for most queries.

    Best for:
    - Simple semantic queries
    - Product/campaign similarity
    - General search

    Example queries:
    - "Find campaigns similar to summer sale"
    - "Show products like wireless headphones"
    - "What promotions target millennials?"
    """

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve using vector similarity.

        Args:
            query_bundle: Query bundle

        Returns:
            List of similar nodes
        """
        if not self.index:
            return []

        # Use LlamaIndex vector retriever
        retriever = self.index.as_retriever(
            similarity_top_k=self.top_k
        )

        nodes = retriever.retrieve(query_bundle.query_str)
        nodes = self._filter_by_threshold(nodes)
        nodes = self._enhance_metadata(nodes)

        return nodes


# Standalone function for backward compatibility
def create_vector_retriever(index, top_k: int = 5):
    """Create a vector retriever from an index."""
    return VectorRetrieverAdapter(index=index, top_k=top_k)
