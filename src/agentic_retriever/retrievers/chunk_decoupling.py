"""
Chunk Decoupling Retriever Adapter

Wraps the chunk decoupling strategy from 15_chunk_decoupling.py
"""

import sys
from pathlib import Path
from typing import List, Optional



from llama_index.core.schema import NodeWithScore, TextNode

from .base import BaseRetrieverAdapter


class ChunkDecouplingRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for chunk decoupling retrieval strategy."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        window_size: int = 3
    ):
        """
        Initialize chunk decoupling retriever adapter.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            window_size: Size of the context window
        """
        super().__init__("chunk_decoupling")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.window_size = window_size
        
        # Import and create the sentence window retriever from pipeline script
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "chunk_decoupling", 
                Path(__file__).parent.parent.parent / "15_chunk_decoupling.py"
            )
            chunk_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chunk_module)
            
            # Convert embeddings to documents
            from llama_index.core import Document
            documents = []
            for emb_data in embeddings:
                doc = Document(
                    text=emb_data.get('text', ''),
                    metadata=emb_data.get('metadata', {})
                )
                documents.append(doc)
            
            # Create sentence window retriever
            self.retriever = chunk_module.SentenceWindowRetriever(
                documents=documents,
                window_size=window_size
            )
            self.use_advanced_retrieval = True
            
        except Exception as e:
            print(f"⚠️ Advanced chunk decoupling not available: {e}")
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
        try:
            if hasattr(self, 'use_advanced_retrieval') and self.use_advanced_retrieval:
                # Use advanced sentence window retrieval
                result = self.retriever.query(
                    query=query,
                    similarity_top_k=k,
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
            print(f"⚠️ Error in chunk decoupling retrieval: {e}")
            # Fallback to empty results
            nodes = []
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        window_size: int = 3
    ) -> "ChunkDecouplingRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            window_size: Size of the context window
            
        Returns:
            ChunkDecouplingRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            window_size=window_size
        )