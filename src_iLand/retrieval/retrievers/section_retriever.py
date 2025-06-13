"""
Section-based retriever adapter for structured Thai land deed documents.
Optimized for section-aware retrieval with metadata filtering.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator

from .base import BaseRetrieverAdapter


class SectionBasedRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for section-based retrieval of land deed documents."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        enable_metadata_filtering: bool = True,
        preferred_sections: List[str] = None
    ):
        """
        Initialize section-based retriever adapter.
        
        Args:
            embeddings: Embedding data (should include section-based chunks)
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            enable_metadata_filtering: Whether to use metadata filtering
            preferred_sections: List of sections to prioritize in results
        """
        super().__init__("section_based")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.enable_metadata_filtering = enable_metadata_filtering
        
        # Default preferred sections for land deed queries
        self.preferred_sections = preferred_sections or [
            "key_info",      # Always prioritize key info chunks
            "location",      # Location queries are common
            "deed_info",     # Deed identification
            "area_measurements",  # Size queries
            "geolocation"    # Coordinate queries
        ]
        
        # Section priority mapping for query routing
        self.section_query_mapping = {
            # Location-related queries
            "location": ["ที่ตั้ง", "location", "province", "จังหวัด", "อำเภอ", "district", "ตำบล", "subdistrict"],
            "geolocation": ["พิกัด", "coordinates", "ลองจิจูด", "longitude", "ละติจูด", "latitude", "maps"],
            
            # Deed information queries
            "deed_info": ["โฉนด", "deed", "เลขที่", "number", "ประเภท", "type", "เล่ม", "book", "หน้า", "page"],
            
            # Area-related queries
            "area_measurements": ["ขนาด", "size", "area", "เนื้อที่", "ไร่", "rai", "งาน", "ngan", "ตารางวา", "wa"],
            
            # Classification queries
            "classification": ["หมวดหมู่", "category", "ประเภท", "type", "classification"],
            
            # Date queries
            "dates": ["วันที่", "date", "เมื่อไหร่", "when", "ได้มา", "received"],
            
            # Financial queries
            "financial": ["ราคา", "price", "value", "มูลค่า", "การเงิน", "financial"]
        }
        
        # Filter section-based embeddings
        self.section_embeddings = [
            emb for emb in embeddings 
            if emb.get('type') == 'section_chunk' or emb.get('metadata', {}).get('chunk_type') in ['key_info', 'section', 'section_part']
        ]
        
        print(f"🎯 SectionBasedRetriever initialized with {len(self.section_embeddings)} section chunks")
        
        # Build the retriever
        self._build_retriever()
    
    def _build_retriever(self):
        """Build the section-based retriever."""
        if not self.section_embeddings:
            print("⚠️ No section-based embeddings found, using all embeddings")
            self.section_embeddings = self.embeddings
        
        # Import and create the retriever from embedding data
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "load_embeddings", 
            Path(__file__).parent.parent.parent.parent / "src" / "load_embeddings.py"
        )
        load_embeddings_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_embeddings_module)
        
        # Create index from section embeddings
        try:
            self.index = load_embeddings_module.create_index_from_embeddings(
                self.section_embeddings,
                api_key=self.api_key,
                show_progress=False
            )
            
            self.base_retriever = self.index.as_retriever(
                similarity_top_k=self.default_top_k
            )
            print(f"✅ Section-based retriever built successfully")
            
        except Exception as e:
            print(f"❌ Error building section-based retriever: {e}")
            self.index = None
            self.base_retriever = None
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using section-based strategy with intelligent routing.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        if not self.base_retriever:
            print("⚠️ Section-based retriever not available")
            return []
        
        k = top_k if top_k is not None else self.default_top_k
        
        # Detect preferred sections based on query content
        query_sections = self._detect_query_sections(query)
        
        # Apply metadata filtering if enabled
        metadata_filters = None
        if self.enable_metadata_filtering:
            metadata_filters = self._create_section_filters(query_sections)
        
        # Perform retrieval
        try:
            if metadata_filters:
                # Use filtered retrieval
                retrieved_nodes = self._retrieve_with_section_filters(query, k, metadata_filters)
            else:
                # Standard retrieval
                self.base_retriever.similarity_top_k = k
                retrieved_nodes = self.base_retriever.retrieve(query)
            
            # Post-process and re-rank based on section preferences
            ranked_nodes = self._rank_by_section_preference(retrieved_nodes, query_sections)
            
            # Tag nodes with strategy
            return self._tag_nodes_with_strategy(ranked_nodes)
            
        except Exception as e:
            print(f"⚠️ Error in section-based retrieval: {e}")
            return []
    
    def _detect_query_sections(self, query: str) -> List[str]:
        """Detect which sections are most relevant to the query."""
        query_lower = query.lower()
        relevant_sections = []
        
        for section, keywords in self.section_query_mapping.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if section not in relevant_sections:
                        relevant_sections.append(section)
                    break
        
        # Always include key_info for comprehensive results
        if "key_info" not in relevant_sections:
            relevant_sections.insert(0, "key_info")
        
        return relevant_sections
    
    def _create_section_filters(self, preferred_sections: List[str]) -> MetadataFilters:
        """Create metadata filters for section-based retrieval."""
        if not preferred_sections:
            return None
        
        # Create filter for preferred sections
        section_filters = []
        for section in preferred_sections:
            section_filter = MetadataFilter(
                key="section",
                value=section,
                operator=FilterOperator.EQ
            )
            section_filters.append(section_filter)
        
        # Combine with OR logic (retrieve from any of the preferred sections)
        return MetadataFilters(
            filters=section_filters,
            condition="or"
        )
    
    def _retrieve_with_section_filters(
        self, 
        query: str, 
        top_k: int, 
        metadata_filters: MetadataFilters
    ) -> List[NodeWithScore]:
        """Retrieve with section-based metadata filtering."""
        
        # First, try to retrieve from preferred sections
        try:
            filtered_retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=metadata_filters
            )
            filtered_nodes = filtered_retriever.retrieve(query)
            
            # If we got good results, return them
            if len(filtered_nodes) >= top_k // 2:  # At least half the requested results
                return filtered_nodes
            
        except Exception as e:
            print(f"⚠️ Filtered retrieval failed: {e}")
        
        # Fallback: retrieve without filters and post-filter
        self.base_retriever.similarity_top_k = top_k * 2  # Get more for filtering
        all_nodes = self.base_retriever.retrieve(query)
        
        # Filter by preferred sections
        filtered_nodes = []
        preferred_sections_set = set(metadata_filters.filters[0].value for f in metadata_filters.filters)
        
        for node in all_nodes:
            node_section = node.metadata.get('section', '')
            if node_section in preferred_sections_set:
                filtered_nodes.append(node)
                if len(filtered_nodes) >= top_k:
                    break
        
        return filtered_nodes[:top_k]
    
    def _rank_by_section_preference(
        self, 
        nodes: List[NodeWithScore], 
        preferred_sections: List[str]
    ) -> List[NodeWithScore]:
        """Re-rank nodes based on section preferences."""
        
        # Create section priority map
        section_priority = {section: i for i, section in enumerate(preferred_sections)}
        
        def get_section_score(node):
            section = node.metadata.get('section', '')
            chunk_type = node.metadata.get('chunk_type', '')
            
            # Key info chunks always have highest priority
            if chunk_type == 'key_info':
                return 0
            
            # Use section priority, with unknown sections having low priority
            return section_priority.get(section, len(preferred_sections))
        
        # Sort by: section priority (ascending), then similarity score (descending)
        ranked_nodes = sorted(
            nodes,
            key=lambda x: (get_section_score(x), -getattr(x, 'score', 0.0))
        )
        
        return ranked_nodes
    
    def retrieve_by_section(
        self, 
        query: str, 
        section: str, 
        top_k: int = 5
    ) -> List[NodeWithScore]:
        """Retrieve specifically from a named section."""
        
        if not self.base_retriever:
            return []
        
        # Create filter for specific section
        section_filter = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="section",
                    value=section,
                    operator=FilterOperator.EQ
                )
            ]
        )
        
        try:
            section_retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=section_filter
            )
            nodes = section_retriever.retrieve(query)
            return self._tag_nodes_with_strategy(nodes)
            
        except Exception as e:
            print(f"⚠️ Section-specific retrieval failed for '{section}': {e}")
            return []
    
    def get_available_sections(self) -> Dict[str, int]:
        """Get count of chunks available by section."""
        section_counts = {}
        
        for emb in self.section_embeddings:
            section = emb.get('metadata', {}).get('section', 'unknown')
            section_counts[section] = section_counts.get(section, 0) + 1
        
        return dict(sorted(section_counts.items(), key=lambda x: x[1], reverse=True))
    
    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        enable_metadata_filtering: bool = True,
        preferred_sections: List[str] = None
    ) -> "SectionBasedRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            enable_metadata_filtering: Whether to use metadata filtering
            preferred_sections: List of sections to prioritize
            
        Returns:
            SectionBasedRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            enable_metadata_filtering=enable_metadata_filtering,
            preferred_sections=preferred_sections
        ) 