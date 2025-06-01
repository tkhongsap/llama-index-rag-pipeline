"""
Hybrid Retriever Adapter

Wraps the hybrid search combining vector and keyword from 16_hybrid_search.py
"""

import sys
from pathlib import Path
from typing import List, Optional



from llama_index.core.schema import NodeWithScore, TextNode

from .base import BaseRetrieverAdapter


class HybridRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for hybrid search combining vector and keyword retrieval."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        fusion_method: str = "rrf"
    ):
        """
        Initialize hybrid retriever adapter.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            fusion_method: Method for fusing results ("rrf", "weighted", "borda")
        """
        super().__init__("hybrid")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.fusion_method = fusion_method
        
        # Import and create the hybrid search engine from pipeline script
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "hybrid_search", 
                Path(__file__).parent.parent.parent / "16_hybrid_search.py"
            )
            hybrid_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hybrid_module)
            
            # Convert embeddings to nodes and create index
            from llama_index.core import VectorStoreIndex
            from llama_index.core.schema import TextNode
            
            nodes = []
            for emb_data in embeddings:
                node = TextNode(
                    text=emb_data.get('text', ''),
                    metadata=emb_data.get('metadata', {}),
                    embedding=emb_data.get('embedding')
                )
                nodes.append(node)
            
            # Create index from nodes
            index = VectorStoreIndex(nodes)
            
            # Create hybrid search engine
            self.retriever = hybrid_module.HybridSearchEngine(
                index=index,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                fusion_method=fusion_method,
                similarity_top_k=default_top_k
            )
            self.use_advanced_retrieval = True
            
        except Exception as e:
            print(f"⚠️ Advanced hybrid search not available: {e}")
            print("⚠️ Using basic vector retrieval as fallback")
            
            # Fallback to basic vector retrieval
            from llama_index.core import VectorStoreIndex
            from llama_index.core.schema import TextNode
            
            # Convert embeddings to nodes
            nodes = []
            for emb_data in embeddings:
                node = TextNode(
                    text=emb_data.get('text', ''),
                    metadata=emb_data.get('metadata', {}),
                    embedding=emb_data.get('embedding')
                )
                nodes.append(node)
            
            # Create index from nodes
            index = VectorStoreIndex(nodes)
            
            # Create basic retriever
            self.retriever = index.as_retriever(similarity_top_k=default_top_k)
            self.use_advanced_retrieval = False
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid search strategy.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.default_top_k
        
        # Perform hybrid retrieval
        try:
            if hasattr(self, 'use_advanced_retrieval') and self.use_advanced_retrieval:
                # Use advanced hybrid search
                result = self.retriever.query(
                    query=query,
                    top_k=k,
                    show_details=False
                )
                
                # Convert sources to NodeWithScore objects
                nodes = []
                for source_info in result.get('sources', []):
                    # Create TextNode from source info
                    text_node = TextNode(
                        text=source_info.get('text_preview', ''),
                        metadata=source_info.get('metadata', {})
                    )
                    
                    # Create NodeWithScore
                    node_with_score = NodeWithScore(
                        node=text_node,
                        score=source_info.get('score', 0.0)
                    )
                    nodes.append(node_with_score)
            else:
                # Use basic vector retrieval
                from llama_index.core import QueryBundle
                query_bundle = QueryBundle(query_str=query)
                nodes = self.retriever.retrieve(query_bundle)
                
        except Exception as e:
            print(f"⚠️ Error in hybrid retrieval: {e}")
            # Fallback to empty results
            nodes = []
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
    def set_weights(self, semantic_weight: float, keyword_weight: float):
        """Update the weights for semantic and keyword search."""
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        if hasattr(self.retriever, 'semantic_weight'):
            self.retriever.semantic_weight = semantic_weight
            self.retriever.keyword_weight = keyword_weight
    
    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        fusion_method: str = "rrf"
    ) -> "HybridRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            fusion_method: Method for fusing results ("rrf", "weighted", "borda")
            
        Returns:
            HybridRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            fusion_method=fusion_method
        )