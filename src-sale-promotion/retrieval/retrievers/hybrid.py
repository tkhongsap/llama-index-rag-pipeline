"""
Hybrid Retriever Adapter for Sales Promotions

Strategy 5: Combines vector similarity (semantic) + BM25 (keyword) search.
Best for queries with Thai keywords like "น้ำดื่มช้าง PET promotions"

Performance Target: < 150ms
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import QueryFusionRetriever

from .base import BaseRetrieverAdapter


class HybridRetrieverAdapter(BaseRetrieverAdapter):
    """
    Hybrid retriever combining vector and BM25 keyword search.
    
    Optimized for Thai language queries with specific product keywords.
    Uses weighted fusion (alpha parameter) to balance semantic vs keyword.
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        nodes: List,
        alpha: float = 0.5,
        top_k: int = 5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_index: VectorStoreIndex for semantic search
            nodes: List of nodes for BM25 search
            alpha: Weight for vector vs keyword (0=keyword only, 1=vector only)
            top_k: Number of results to return
        """
        super().__init__("hybrid")
        self.vector_index = vector_index
        self.nodes = nodes
        self.alpha = alpha
        self.top_k = top_k
        
        # Try to import BM25 retriever
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            
            # Create BM25 retriever (no Thai stemmer available)
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=top_k
            )
            self.bm25_available = True
        except ImportError:
            print("⚠️  BM25Retriever not available, falling back to vector-only")
            self.bm25_available = False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid vector + keyword search.
        
        Args:
            query: Search query (supports Thai text)
            top_k: Number of results (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy metadata
        """
        k = top_k if top_k is not None else self.top_k
        
        # If BM25 not available, fall back to vector-only
        if not self.bm25_available:
            retriever = self.vector_index.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(query)
            return self._tag_nodes_with_strategy(nodes)
        
        # Perform vector search
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=k)
        vector_nodes = vector_retriever.retrieve(query)
        
        # Perform BM25 search
        bm25_nodes = self.bm25_retriever.retrieve(query)
        
        # Fuse results using weighted combination
        fused_nodes = self._fuse_results(vector_nodes, bm25_nodes, k)
        
        # Tag nodes with strategy info
        return self._tag_nodes_with_strategy(fused_nodes)
    
    def _fuse_results(
        self,
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore],
        top_k: int
    ) -> List[NodeWithScore]:
        """
        Fuse vector and BM25 results using weighted combination.
        
        Args:
            vector_nodes: Results from vector search
            bm25_nodes: Results from BM25 search
            top_k: Number of final results
            
        Returns:
            Fused and ranked results
        """
        # Create score map
        node_scores = {}
        
        # Add vector scores (weighted by alpha)
        for node in vector_nodes:
            node_id = node.node.id_
            node_scores[node_id] = {
                'node': node,
                'vector_score': node.score * self.alpha,
                'bm25_score': 0
            }
        
        # Add BM25 scores (weighted by 1-alpha)
        for node in bm25_nodes:
            node_id = node.node.id_
            if node_id in node_scores:
                node_scores[node_id]['bm25_score'] = node.score * (1 - self.alpha)
            else:
                node_scores[node_id] = {
                    'node': node,
                    'vector_score': 0,
                    'bm25_score': node.score * (1 - self.alpha)
                }
        
        # Calculate combined scores
        for node_id in node_scores:
            node_scores[node_id]['combined_score'] = (
                node_scores[node_id]['vector_score'] + 
                node_scores[node_id]['bm25_score']
            )
        
        # Sort by combined score
        sorted_items = sorted(
            node_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        # Create fused nodes with updated scores
        fused_nodes = []
        for node_id, scores in sorted_items[:top_k]:
            node = scores['node']
            node.score = scores['combined_score']
            fused_nodes.append(node)
        
        return fused_nodes
