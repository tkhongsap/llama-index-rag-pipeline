"""
Metadata Filter PostgreSQL Retriever for iLand Data

Implements metadata-aware retrieval with strong filtering capabilities
for Thai geographic, legal, and document-specific attributes.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from llama_index.core.schema import NodeWithScore

from ..base_retriever import BasePostgresRetriever
from ..config import PostgresConfig

logger = logging.getLogger(__name__)


class MetadataFilterPostgresRetriever(BasePostgresRetriever):
    """
    Metadata-aware retriever with sophisticated filtering for Thai land deed data.
    
    Applies metadata filters before vector search for more precise results
    based on geographic, legal, and document attributes.
    """
    
    def __init__(
        self,
        config: PostgresConfig,
        require_filters: bool = False,
        filter_first: bool = True
    ):
        """
        Initialize metadata filter PostgreSQL retriever.
        
        Args:
            config: PostgreSQL configuration
            require_filters: Whether metadata filters are required
            filter_first: Whether to apply filters before vector search
        """
        super().__init__(config, "metadata_filter_postgres")
        self.require_filters = require_filters
        self.filter_first = filter_first
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes with metadata filtering.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve
            filters: Metadata filters (required if require_filters=True)
            
        Returns:
            List of NodeWithScore objects with metadata filtering applied
        """
        start_time = time.time()
        
        top_k = top_k or self.config.default_top_k
        
        # Validate filters
        if self.require_filters and not filters:
            raise ValueError("Metadata filters are required for this retriever")
        
        validated_filters = self._validate_filters(filters)
        
        # Extract auto-detected filters from query
        auto_filters = self._extract_filters_from_query(query)
        
        # Combine explicit and auto-detected filters
        combined_filters = self._combine_filters(validated_filters, auto_filters)
        
        try:
            query_embedding = self._get_query_embedding(query)
            
            if self.filter_first:
                # Apply filters during vector search
                results = self._filtered_vector_search(
                    query_embedding=query_embedding,
                    filters=combined_filters,
                    top_k=top_k
                )
            else:
                # Search first, then filter
                results = self._vector_then_filter(
                    query_embedding=query_embedding,
                    filters=combined_filters,
                    top_k=top_k
                )
            
            # Apply additional post-processing filters
            processed_results = self._post_process_results(results, combined_filters)
            
            # Convert to nodes
            nodes = self._format_results_as_nodes(processed_results)
            
            # Log metrics
            execution_time = time.time() - start_time
            self._log_retrieval_metrics(
                query=query,
                num_results=len(nodes),
                execution_time=execution_time,
                additional_info={
                    "explicit_filters": len(validated_filters),
                    "auto_detected_filters": len(auto_filters),
                    "combined_filters": len(combined_filters),
                    "filter_first": self.filter_first
                }
            )
            
            logger.info(f"Metadata filter retrieval completed: {len(nodes)} nodes with {len(combined_filters)} filters")
            return nodes
            
        except Exception as e:
            logger.error(f"Metadata filter retrieval failed: {e}")
            raise
    
    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata filters from Thai text query.
        
        Args:
            query: Search query text
            
        Returns:
            Dictionary of auto-detected filters
        """
        auto_filters = {}
        
        try:
            # Extract geographic metadata
            geo_metadata = self.metadata_utils.extract_geographic_metadata(query)
            if geo_metadata:
                if "primary_province" in geo_metadata:
                    auto_filters["province"] = geo_metadata["primary_province"]
                if "districts" in geo_metadata:
                    auto_filters["district"] = geo_metadata["districts"]
            
            # Extract deed metadata
            deed_metadata = self.metadata_utils.extract_deed_metadata(query)
            if deed_metadata:
                if "primary_deed_type" in deed_metadata:
                    auto_filters["deed_type"] = deed_metadata["primary_deed_type"]
                if "deed_numbers" in deed_metadata:
                    auto_filters["deed_number"] = deed_metadata["deed_numbers"]
            
            logger.debug(f"Auto-detected {len(auto_filters)} filters from query")
            return auto_filters
            
        except Exception as e:
            logger.warning(f"Failed to extract filters from query: {e}")
            return {}
    
    def _combine_filters(
        self,
        explicit_filters: Dict[str, Any],
        auto_filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine explicit and auto-detected filters intelligently.
        
        Args:
            explicit_filters: User-provided filters
            auto_filters: Auto-detected filters from query
            
        Returns:
            Combined filter dictionary
        """
        combined = explicit_filters.copy()
        
        # Add auto-detected filters if not explicitly specified
        for key, value in auto_filters.items():
            if key not in combined:
                combined[key] = value
            elif isinstance(combined[key], list) and not isinstance(value, list):
                # Add to existing list
                if value not in combined[key]:
                    combined[key].append(value)
            elif not isinstance(combined[key], list) and isinstance(value, list):
                # Convert to list and merge
                combined[key] = [combined[key]] + [v for v in value if v != combined[key]]
        
        return combined
    
    def _filtered_vector_search(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search with metadata pre-filtering.
        
        Args:
            query_embedding: Query embedding vector
            filters: Metadata filters to apply
            top_k: Number of results to retrieve
            
        Returns:
            Filtered search results
        """
        # Search multiple tables if no specific table preference
        table_preference = self._determine_table_preference(filters)
        
        if table_preference:
            # Search specific table
            results = self._vector_similarity_search(
                query_embedding=query_embedding,
                table_name=table_preference,
                top_k=top_k,
                similarity_threshold=self.config.similarity_threshold,
                metadata_filters=filters
            )
        else:
            # Search combined table or multiple tables
            results = self._search_multiple_tables(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k
            )
        
        return results
    
    def _vector_then_filter(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search first, then apply metadata filtering.
        
        Args:
            query_embedding: Query embedding vector
            filters: Metadata filters to apply
            top_k: Number of results to retrieve
            
        Returns:
            Search results with post-filtering
        """
        # Get more results initially for filtering
        initial_top_k = min(top_k * 3, self.config.max_top_k)
        
        # Search without metadata filters
        initial_results = self._vector_similarity_search(
            query_embedding=query_embedding,
            table_name=self.config.combined_table,
            top_k=initial_top_k,
            similarity_threshold=self.config.similarity_threshold
        )
        
        # Apply metadata filtering
        filtered_results = self._apply_metadata_filters(initial_results, filters)
        
        # Return top results after filtering
        return filtered_results[:top_k]
    
    def _apply_metadata_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to search results.
        
        Args:
            results: Search results to filter
            filters: Metadata filters to apply
            
        Returns:
            Filtered results
        """
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            # Check each filter
            matches_all_filters = True
            
            for filter_key, filter_value in filters.items():
                metadata_value = metadata.get(filter_key)
                
                if not self._check_filter_match(metadata_value, filter_value):
                    matches_all_filters = False
                    break
            
            if matches_all_filters:
                filtered_results.append(result)
        
        return filtered_results
    
    def _check_filter_match(self, metadata_value: Any, filter_value: Any) -> bool:
        """
        Check if metadata value matches filter criteria.
        
        Args:
            metadata_value: Value from metadata
            filter_value: Filter criteria
            
        Returns:
            True if matches, False otherwise
        """
        if filter_value is None:
            return metadata_value is None
        
        if isinstance(filter_value, list):
            # OR condition - metadata value should match any in the list
            return metadata_value in filter_value
        
        if isinstance(filter_value, dict):
            # Range or special conditions
            if metadata_value is None:
                return False
            
            try:
                numeric_value = float(metadata_value)
                
                if "gte" in filter_value and numeric_value < filter_value["gte"]:
                    return False
                if "lte" in filter_value and numeric_value > filter_value["lte"]:
                    return False
                if "gt" in filter_value and numeric_value <= filter_value["gt"]:
                    return False
                if "lt" in filter_value and numeric_value >= filter_value["lt"]:
                    return False
                
                return True
                
            except (ValueError, TypeError):
                return False
        
        # Exact match
        return str(metadata_value) == str(filter_value)
    
    def _determine_table_preference(self, filters: Dict[str, Any]) -> Optional[str]:
        """
        Determine the best table to search based on filters.
        
        Args:
            filters: Metadata filters
            
        Returns:
            Preferred table name or None for multiple tables
        """
        # If looking for summaries specifically
        if "result_type" in filters and filters["result_type"] == "summary":
            return self.config.summaries_table
        
        # If looking for specific chunk information
        if "chunk_index" in filters:
            return self.config.chunks_table
        
        # Default to combined table for most filters
        return self.config.combined_table
    
    def _search_multiple_tables(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search multiple tables and combine results.
        
        Args:
            query_embedding: Query embedding vector
            filters: Metadata filters
            top_k: Number of results to retrieve
            
        Returns:
            Combined search results
        """
        all_results = []
        
        # Define tables to search
        tables_to_search = [
            self.config.chunks_table,
            self.config.summaries_table
        ]
        
        # Search each table
        for table in tables_to_search:
            try:
                table_results = self._vector_similarity_search(
                    query_embedding=query_embedding,
                    table_name=table,
                    top_k=max(1, top_k // len(tables_to_search)),
                    similarity_threshold=self.config.similarity_threshold,
                    metadata_filters=filters
                )
                
                # Add table source
                for result in table_results:
                    result["source_table"] = table
                
                all_results.extend(table_results)
                
            except Exception as e:
                logger.warning(f"Failed to search table {table}: {e}")
                continue
        
        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return all_results[:top_k]
    
    def _post_process_results(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply additional post-processing to results.
        
        Args:
            results: Search results
            filters: Applied filters
            
        Returns:
            Post-processed results
        """
        # Add filter information to metadata
        for result in results:
            if isinstance(result.get("metadata"), dict):
                result["metadata"]["applied_filters"] = filters
                result["metadata"]["filter_method"] = "metadata_filter_postgres"
        
        return results 