"""
Metadata Retriever Adapter for iLand Data

Implements metadata-filtered retrieval for Thai land deed data with province, district, and other filters.
Adapted from src/agentic_retriever/retrievers/metadata.py for iLand data.
"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

from .base import BaseRetrieverAdapter


class MetadataRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for metadata-filtered retrieval on iLand data."""
    
    def __init__(self, index: VectorStoreIndex, default_top_k: int = 5):
        """
        Initialize metadata retriever adapter for iLand data.
        
        Args:
            index: Vector store index with Thai land deed metadata
            default_top_k: Default number of nodes to retrieve
        """
        super().__init__("metadata")
        self.index = index
        self.default_top_k = default_top_k
        
        # Thai provinces for metadata filtering
        self.thai_provinces = [
            "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี", "พระนครศรีอยุธยา",
            "อ่างทอง", "ลพบุรี", "สิงห์บุรี", "ชัยนาท", "สระบุรี", "ชลบุรี", "ระยอง",
            "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "ปราจีนบุรี", "นครนายก", "สระแก้ว"
            # Add more provinces as needed
        ]
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                 filters: Optional[Dict[str, Any]] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using metadata filtering on iLand data.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (uses default if None)
            filters: Metadata filters to apply (province, district, etc.)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Build metadata filters for Thai land deed data
        metadata_filters = self._build_metadata_filters(filters, query)
        
        # Create retriever with filters
        if metadata_filters:
            retriever = self.index.as_retriever(
                similarity_top_k=k,
                filters=metadata_filters
            )
        else:
            retriever = self.index.as_retriever(similarity_top_k=k)
        
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
    def _build_metadata_filters(self, filters: Optional[Dict[str, Any]], 
                               query: str) -> Optional[MetadataFilters]:
        """
        Build metadata filters for Thai land deed data.
        
        Args:
            filters: Explicit filters provided
            query: Query text to extract implicit filters from
            
        Returns:
            MetadataFilters object or None
        """
        filter_conditions = []
        
        # Apply explicit filters
        if filters:
            for key, value in filters.items():
                if key == "province" and value:
                    filter_conditions.append(
                        MetadataFilter(key="province", value=value, operator=FilterOperator.EQ)
                    )
                elif key == "district" and value:
                    filter_conditions.append(
                        MetadataFilter(key="district", value=value, operator=FilterOperator.EQ)
                    )
                elif key == "land_type" and value:
                    filter_conditions.append(
                        MetadataFilter(key="land_type", value=value, operator=FilterOperator.EQ)
                    )
                elif key == "year" and value:
                    filter_conditions.append(
                        MetadataFilter(key="year", value=value, operator=FilterOperator.EQ)
                    )
        
        # Extract implicit filters from query (Thai province names)
        query_lower = query.lower()
        for province in self.thai_provinces:
            if province in query or province.lower() in query_lower:
                filter_conditions.append(
                    MetadataFilter(key="province", value=province, operator=FilterOperator.EQ)
                )
                break  # Use first matching province
        
        # Check for land type keywords in Thai
        land_type_keywords = {
            "ที่ดิน": "land",
            "บ้าน": "house", 
            "อาคาร": "building",
            "คอนโด": "condo"
        }
        
        for thai_keyword, eng_type in land_type_keywords.items():
            if thai_keyword in query:
                filter_conditions.append(
                    MetadataFilter(key="land_type", value=eng_type, operator=FilterOperator.EQ)
                )
                break
        
        if filter_conditions:
            return MetadataFilters(filters=filter_conditions)
        
        return None
    
    @classmethod
    def from_iland_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5
    ) -> "MetadataRetrieverAdapter":
        """
        Create adapter from iLand embeddings with metadata.
        
        Args:
            embeddings: List of iLand embedding dictionaries with metadata
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            
        Returns:
            MetadataRetrieverAdapter instance for iLand data
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from load_embedding import iLandIndexReconstructor
        
        from load_embedding.models import EmbeddingConfig
        config = EmbeddingConfig(api_key=api_key)
        reconstructor = iLandIndexReconstructor(config=config)
        index = reconstructor.create_vector_index_from_embeddings(
            embeddings,
            show_progress=False
        )
        return cls(index, default_top_k) 