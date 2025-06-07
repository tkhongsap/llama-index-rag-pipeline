"""
index_reconstructor.py - LlamaIndex reconstruction from iLand embeddings

This module contains the iLandIndexReconstructor class responsible for
reconstructing various LlamaIndex indices from saved iLand embeddings.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import LlamaIndex components
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext
)
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .models import EmbeddingConfig
from .embedding_loader import iLandEmbeddingLoader

# Import iLand specific modules with fallback
try:
    from ..docs_embedding import iLandMetadataExtractor
except ImportError:
    try:
        # Try absolute import as backup
        import sys
        from pathlib import Path
        docs_embedding_path = Path(__file__).parent.parent / "docs_embedding"
        sys.path.insert(0, str(docs_embedding_path))
        from metadata_extractor import iLandMetadataExtractor
    except ImportError:
        print("⚠️ Warning: Could not import iLandMetadataExtractor. Some features may be limited.")
        class iLandMetadataExtractor:
            pass

# ---------- ILAND INDEX RECONSTRUCTION CLASS --------------------------------

class iLandIndexReconstructor:
    """Reconstruct various LlamaIndex indices from saved iLand embeddings."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize iLand index reconstructor."""
        self.config = config or EmbeddingConfig()
        
        try:
            self.metadata_extractor = iLandMetadataExtractor()
        except:
            self.metadata_extractor = None
            
        if self.config.api_key:
            self._setup_llm_settings()
        else:
            print("⚠️ Warning: No OpenAI API key found. Index creation will be limited.")
    
    def _setup_llm_settings(self):
        """Configure LLM settings for Thai land deed processing."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key not found")
        
        # Configure models
        Settings.llm = OpenAI(
            model=self.config.llm_model, 
            temperature=0, 
            api_key=self.config.api_key
        )
        Settings.embed_model = OpenAIEmbedding(
            model=self.config.embed_model, 
            api_key=self.config.api_key
        )
    
    def embeddings_to_nodes(
        self, 
        embeddings: List[Dict[str, Any]]
    ) -> List[Union[TextNode, IndexNode]]:
        """Convert iLand embedding dictionaries to LlamaIndex nodes."""
        nodes = []
        
        for emb in embeddings:
            # Get text content based on embedding type
            if emb.get("type") == "summary":
                text_content = emb.get("summary_text", "")
            elif emb.get("doc_type") == "summary":
                text_content = emb.get("text", "")
            else:
                text_content = emb.get("text", "")
            
            # Create node based on type
            if emb.get("type") == "indexnode" or emb.get("doc_type") == "indexnode":
                # Create IndexNode
                node = IndexNode(
                    text=text_content,
                    index_id=emb.get("index_id", emb.get("doc_id", "")),
                    metadata=emb.get("metadata", {}),
                    embedding=emb.get("embedding_vector"),
                    id_=emb.get("node_id", emb.get("doc_id", ""))
                )
            else:
                # Create TextNode
                node = TextNode(
                    text=text_content,
                    metadata=emb.get("metadata", {}),
                    embedding=emb.get("embedding_vector"),
                    id_=emb.get("node_id", emb.get("doc_id", ""))
                )
            
            nodes.append(node)
        
        return nodes
    
    def create_vector_index_from_embeddings(
        self, 
        embeddings: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """Create a VectorStoreIndex from iLand embeddings."""
        # Convert to nodes
        nodes = self.embeddings_to_nodes(embeddings)
        
        if show_progress:
            print(f"🔄 Creating VectorStoreIndex from {len(nodes)} iLand nodes...")
        
        # Create vector store with embeddings
        vector_store = SimpleVectorStore()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=show_progress
        )
        
        if show_progress:
            print(f"✅ iLand VectorStoreIndex created with {len(nodes)} nodes")
        
        return index
    
    def create_combined_iland_index(
        self,
        chunk_embeddings: List[Dict[str, Any]],
        summary_embeddings: Optional[List[Dict[str, Any]]] = None,
        indexnode_embeddings: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """Create a combined index from multiple iLand embedding types."""
        all_embeddings = []
        
        # Add chunks (primary content)
        if chunk_embeddings:
            all_embeddings.extend(chunk_embeddings)
            if show_progress:
                print(f"📄 Added {len(chunk_embeddings)} iLand chunk embeddings")
        
        # Add summaries (optional)
        if summary_embeddings:
            all_embeddings.extend(summary_embeddings)
            if show_progress:
                print(f"📋 Added {len(summary_embeddings)} iLand summary embeddings")
        
        # Add index nodes (optional)
        if indexnode_embeddings:
            all_embeddings.extend(indexnode_embeddings)
            if show_progress:
                print(f"📊 Added {len(indexnode_embeddings)} iLand index node embeddings")
        
        return self.create_vector_index_from_embeddings(
            all_embeddings, 
            show_progress=show_progress
        )
    
    def create_province_specific_index(
        self,
        embeddings: List[Dict[str, Any]],
        provinces: Union[str, List[str]],
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """Create an index filtered by specific Thai provinces."""
        if isinstance(provinces, str):
            provinces = [provinces]
        
        # Filter embeddings by province
        loader = iLandEmbeddingLoader(self.config)
        filtered_embeddings = loader.filter_embeddings_by_province(embeddings, provinces)
        
        if show_progress:
            print(f"🌏 Creating province-specific index for: {', '.join(provinces)}")
            print(f"📄 Filtered to {len(filtered_embeddings)} embeddings from {len(embeddings)} total")
        
        return self.create_vector_index_from_embeddings(
            filtered_embeddings,
            show_progress=show_progress
        )
    
    def create_deed_type_specific_index(
        self,
        embeddings: List[Dict[str, Any]],
        deed_types: Union[str, List[str]],
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """Create an index filtered by specific deed types."""
        if isinstance(deed_types, str):
            deed_types = [deed_types]
        
        # Filter embeddings by deed type
        loader = iLandEmbeddingLoader(self.config)
        filtered_embeddings = loader.filter_embeddings_by_deed_type(embeddings, deed_types)
        
        if show_progress:
            print(f"📋 Creating deed-type-specific index for: {', '.join(deed_types)}")
            print(f"📄 Filtered to {len(filtered_embeddings)} embeddings from {len(embeddings)} total")
        
        return self.create_vector_index_from_embeddings(
            filtered_embeddings,
            show_progress=show_progress
        )
    
    def create_multi_filtered_index(
        self,
        embeddings: List[Dict[str, Any]],
        provinces: Optional[Union[str, List[str]]] = None,
        deed_types: Optional[Union[str, List[str]]] = None,
        min_area_rai: Optional[float] = None,
        max_area_rai: Optional[float] = None,
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """Create an index with multiple filters applied."""
        loader = iLandEmbeddingLoader(self.config)
        filtered_embeddings = embeddings
        
        # Apply province filter
        if provinces:
            filtered_embeddings = loader.filter_embeddings_by_province(filtered_embeddings, provinces)
            if show_progress:
                print(f"🌏 Filtered by provinces: {provinces} -> {len(filtered_embeddings)} embeddings")
        
        # Apply deed type filter
        if deed_types:
            filtered_embeddings = loader.filter_embeddings_by_deed_type(filtered_embeddings, deed_types)
            if show_progress:
                print(f"📋 Filtered by deed types: {deed_types} -> {len(filtered_embeddings)} embeddings")
        
        # Apply area filter
        if min_area_rai is not None or max_area_rai is not None:
            filtered_embeddings = loader.filter_embeddings_by_area_range(
                filtered_embeddings, min_area_rai, max_area_rai
            )
            if show_progress:
                print(f"📏 Filtered by area range -> {len(filtered_embeddings)} embeddings")
        
        return self.create_vector_index_from_embeddings(
            filtered_embeddings,
            show_progress=show_progress
        )
    
    def create_query_engine_with_thai_context(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 5,
        response_mode: str = "compact"
    ):
        """Create a query engine optimized for Thai land deed queries."""
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode
        )
        
        # Add Thai-specific system prompt if LLM is configured
        if hasattr(Settings, 'llm') and Settings.llm:
            # This would be implemented based on LlamaIndex's system prompt capabilities
            pass
        
        return query_engine 