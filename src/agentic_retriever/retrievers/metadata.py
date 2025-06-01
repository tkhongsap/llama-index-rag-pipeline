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
        
        # Create a simple metadata-aware retriever
        # For now, we'll use a basic vector retriever with metadata support
        try:
            # Try to import and use the metadata filtering functionality
            import sys
            from pathlib import Path
            
            # Add the src directory to path to import 14_metadata_filtering
            src_path = Path(__file__).parent.parent.parent
            if str(src_path) not in sys.path:
                sys.path.append(str(src_path))
            
            # Import from the correct module name (14_metadata_filtering.py)
            import importlib.util
            metadata_filtering_path = src_path / "14_metadata_filtering.py"
            
            if metadata_filtering_path.exists():
                spec = importlib.util.spec_from_file_location("metadata_filtering_14", metadata_filtering_path)
                metadata_filtering_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_filtering_module)
                
                AutoRetrievalQueryEngine = metadata_filtering_module.AutoRetrievalQueryEngine
                
                # Create a simple index from embeddings for the AutoRetrievalQueryEngine
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
                
                # Create the auto retrieval query engine
                self.retriever = AutoRetrievalQueryEngine(index, top_k=default_top_k)
                self.use_advanced_filtering = True
            else:
                raise FileNotFoundError("14_metadata_filtering.py not found")
                
        except Exception as e:
            print(f"⚠️ Advanced metadata filtering not available: {e}")
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
            self.use_advanced_filtering = False
    
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
        try:
            if hasattr(self, 'use_advanced_filtering') and self.use_advanced_filtering:
                # Use advanced metadata filtering
                # Convert MetadataFilters to dict format if provided
                manual_filters = None
                if filters:
                    manual_filters = {}
                    for filter_obj in filters.filters:
                        manual_filters[filter_obj.key] = filter_obj.value
                
                # Use the auto retrieval query engine
                result = self.retriever.query_with_auto_filters(
                    query=query,
                    manual_filters=manual_filters,
                    show_filters=False
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
            print(f"⚠️ Error in metadata retrieval: {e}")
            # Fallback to empty results
            nodes = []
        
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