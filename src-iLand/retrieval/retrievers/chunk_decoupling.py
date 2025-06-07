"""
Chunk Decoupling Retriever Adapter for iLand Data

Implements chunk decoupling strategy for Thai land deed data, separating chunk retrieval from context synthesis.
Adapted from src/agentic_retriever/retrievers/chunk_decoupling.py for iLand data.
"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank

from .base import BaseRetrieverAdapter


class ChunkDecouplingRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for chunk decoupling retrieval on iLand data."""
    
    def __init__(self, 
                 chunk_index: VectorStoreIndex, 
                 context_index: VectorStoreIndex,
                 default_top_k: int = 5,
                 rerank_top_n: int = 3):
        """
        Initialize chunk decoupling retriever adapter for iLand data.
        
        Args:
            chunk_index: Vector index for individual chunks
            context_index: Vector index for broader context
            default_top_k: Default number of nodes to retrieve
            rerank_top_n: Number of nodes to rerank
        """
        super().__init__("chunk_decoupling")
        self.chunk_index = chunk_index
        self.context_index = context_index
        self.default_top_k = default_top_k
        self.rerank_top_n = rerank_top_n
        
        # Create retrievers
        self.chunk_retriever = chunk_index.as_retriever(similarity_top_k=default_top_k * 2)
        self.context_retriever = context_index.as_retriever(similarity_top_k=default_top_k)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using chunk decoupling approach on iLand data.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Step 1: Retrieve chunks independently
        chunk_nodes = self.chunk_retriever.retrieve(query)
        
        # Step 2: Get context for understanding
        context_nodes = self.context_retriever.retrieve(query)
        
        # Step 3: Combine and deduplicate
        all_nodes = chunk_nodes + context_nodes
        seen_node_ids = set()
        unique_nodes = []
        
        for node in all_nodes:
            node_id = getattr(node.node, 'node_id', str(hash(node.node.text)))
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                unique_nodes.append(node)
        
        # Step 4: Truncate to requested size
        final_nodes = unique_nodes[:k]
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(final_nodes)
    
    def _merge_chunk_context(self, chunk_nodes: List[NodeWithScore], 
                           context_nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Merge chunk and context nodes intelligently.
        
        Args:
            chunk_nodes: Nodes from chunk retrieval
            context_nodes: Nodes from context retrieval
            
        Returns:
            Merged list of nodes
        """
        # Simple merging strategy: alternate between chunk and context
        merged = []
        max_len = max(len(chunk_nodes), len(context_nodes))
        
        for i in range(max_len):
            if i < len(chunk_nodes):
                merged.append(chunk_nodes[i])
            if i < len(context_nodes):
                merged.append(context_nodes[i])
        
        return merged
    
    @classmethod
    def from_iland_embeddings(
        cls,
        chunk_embeddings: List[dict],
        context_embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5
    ) -> "ChunkDecouplingRetrieverAdapter":
        """
        Create adapter from iLand chunk and context embeddings.
        
        Args:
            chunk_embeddings: List of chunk embedding dictionaries
            context_embeddings: List of context embedding dictionaries
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            
        Returns:
            ChunkDecouplingRetrieverAdapter instance for iLand data
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from load_embedding import iLandIndexReconstructor
        
        reconstructor = iLandIndexReconstructor(api_key=api_key)
        
        chunk_index = reconstructor.create_vector_index_from_embeddings(
            chunk_embeddings, show_progress=False
        )
        context_index = reconstructor.create_vector_index_from_embeddings(
            context_embeddings, show_progress=False
        )
        
        return cls(chunk_index, context_index, default_top_k) 