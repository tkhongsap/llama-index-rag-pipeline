"""
Fast Metadata Index Manager for iLand Retrieval

Implements efficient metadata indexing for sub-50ms filtering on 50k documents.
Built to work ON TOP of LlamaIndex framework as a pre-filtering layer.

Based on PRD 2: Metadata Indices for Fast Filtering of 50k Documents
"""

import time
import bisect
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from collections import defaultdict
from pathlib import Path
import pickle
import json

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator


class FastMetadataIndexManager:
    """
    Fast metadata indexing for pre-filtering before vector search.
    
    Implements:
    - Inverted indices for categorical fields (province, deed_type, etc.)
    - B-tree style indices for numeric fields (area, coordinates)
    - Sub-50ms filtering performance for 50k documents
    """
    
    def __init__(self):
        # Core index structures (as per PRD)
        self.keyword_indices: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.numeric_indices: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        
        # Index metadata
        self.indexed_fields: Set[str] = set()
        self.total_documents: int = 0
        self.build_timestamp: Optional[float] = None
        
        # Performance tracking
        self.filter_stats = {
            "total_queries": 0,
            "avg_filter_time_ms": 0.0,
            "avg_reduction_ratio": 0.0
        }
    
    def build_indices_from_llamaindex_nodes(self, nodes: List[TextNode]) -> None:
        """
        Build fast indices from LlamaIndex nodes.
        
        Args:
            nodes: List of LlamaIndex TextNode objects with metadata
        """
        start_time = time.time()
        
        print(f"🔨 Building fast metadata indices from {len(nodes)} nodes...")
        
        # Clear existing indices
        self.keyword_indices.clear()
        self.numeric_indices.clear()
        self.indexed_fields.clear()
        
        # Build indices from each node
        for node in nodes:
            if hasattr(node, 'metadata') and node.metadata:
                self._index_document(node.node_id, node.metadata)
        
        # Sort numeric indices for binary search
        for field in self.numeric_indices:
            self.numeric_indices[field].sort(key=lambda x: x[0])
        
        self.total_documents = len(nodes)
        self.build_timestamp = time.time()
        
        build_time = (self.build_timestamp - start_time) * 1000
        print(f"✅ Fast indices built in {build_time:.2f}ms")
        print(f"   • Categorical fields: {len([f for f in self.indexed_fields if f in self.keyword_indices])}")
        print(f"   • Numeric fields: {len([f for f in self.indexed_fields if f in self.numeric_indices])}")
    
    def _index_document(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Index a single document's metadata."""
        for field, value in metadata.items():
            if value is None:
                continue
                
            self.indexed_fields.add(field)
            
            if isinstance(value, (int, float)):
                self._index_numeric(field, float(value), doc_id)
            else:
                self._index_keyword(field, str(value), doc_id)
    
    def _index_numeric(self, field: str, value: float, doc_id: str) -> None:
        """Add numeric value to B-tree style index."""
        self.numeric_indices[field].append((value, doc_id))
    
    def _index_keyword(self, field: str, value: str, doc_id: str) -> None:
        """Add keyword value to inverted index."""
        # Normalize value (lowercase, strip)
        normalized_value = value.lower().strip()
        self.keyword_indices[field][normalized_value].add(doc_id)
        
        # Also index original value for exact matches
        if normalized_value != value:
            self.keyword_indices[field][value].add(doc_id)
    
    def pre_filter_node_ids(self, filters: MetadataFilters) -> Set[str]:
        """
        Fast pre-filtering that returns matching LlamaIndex node IDs in <50ms.
        
        Args:
            filters: LlamaIndex MetadataFilters object
            
        Returns:
            Set of node IDs that match the filters
        """
        start_time = time.time()
        
        if not filters or not filters.filters:
            return set()
        
        # Process each filter and get matching document sets
        filter_results = []
        
        for metadata_filter in filters.filters:
            matching_docs = self._process_single_filter(metadata_filter)
            filter_results.append(matching_docs)
        
        # Combine results based on condition (AND/OR)
        if not filter_results:
            result_set = set()
        elif filters.condition.upper() == "OR":
            result_set = set.union(*filter_results) if filter_results else set()
        else:  # Default to AND
            result_set = set.intersection(*filter_results) if filter_results else set()
        
        # Update performance stats
        filter_time_ms = (time.time() - start_time) * 1000
        reduction_ratio = 1.0 - (len(result_set) / self.total_documents) if self.total_documents > 0 else 0.0
        
        self._update_filter_stats(filter_time_ms, reduction_ratio)
        
        print(f"🚀 Fast filter: {len(result_set)}/{self.total_documents} docs ({reduction_ratio:.1%} reduction, {filter_time_ms:.2f}ms)")
        
        return result_set
    
    def _process_single_filter(self, metadata_filter: MetadataFilter) -> Set[str]:
        """Process a single metadata filter and return matching doc IDs."""
        field = metadata_filter.key
        value = metadata_filter.value
        operator = metadata_filter.operator
        
        if field not in self.indexed_fields:
            return set()
        
        # Handle different operators
        if operator == FilterOperator.EQ:
            return self._filter_equals(field, value)
        elif operator == FilterOperator.IN:
            return self._filter_in(field, value)
        elif operator in [FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE]:
            return self._filter_numeric_range(field, operator, value)
        else:
            # Fallback for unsupported operators
            return set()
    
    def _filter_equals(self, field: str, value: Any) -> Set[str]:
        """Filter for exact equality."""
        if field in self.keyword_indices:
            # Keyword index lookup
            normalized_value = str(value).lower().strip()
            return self.keyword_indices[field].get(normalized_value, set()).copy()
        elif field in self.numeric_indices:
            # Numeric exact match
            target_value = float(value)
            return {doc_id for val, doc_id in self.numeric_indices[field] if val == target_value}
        return set()
    
    def _filter_in(self, field: str, values: List[Any]) -> Set[str]:
        """Filter for value in list."""
        if not isinstance(values, (list, tuple)):
            values = [values]
        
        result_set = set()
        for value in values:
            result_set.update(self._filter_equals(field, value))
        return result_set
    
    def _filter_numeric_range(self, field: str, operator: FilterOperator, value: Any) -> Set[str]:
        """Filter numeric field with range operators using binary search."""
        if field not in self.numeric_indices:
            return set()
        
        try:
            target_value = float(value)
        except (ValueError, TypeError):
            return set()
        
        sorted_values = self.numeric_indices[field]
        result_set = set()
        
        if operator == FilterOperator.GT:
            # Find insertion point for target_value + small epsilon
            idx = bisect.bisect_right(sorted_values, (target_value, ''))
            result_set.update(doc_id for _, doc_id in sorted_values[idx:])
        elif operator == FilterOperator.GTE:
            # Find insertion point for target_value
            idx = bisect.bisect_left(sorted_values, (target_value, ''))
            result_set.update(doc_id for _, doc_id in sorted_values[idx:])
        elif operator == FilterOperator.LT:
            # Take all values less than target
            idx = bisect.bisect_left(sorted_values, (target_value, ''))
            result_set.update(doc_id for _, doc_id in sorted_values[:idx])
        elif operator == FilterOperator.LTE:
            # Take all values less than or equal to target
            idx = bisect.bisect_right(sorted_values, (target_value, ''))
            result_set.update(doc_id for _, doc_id in sorted_values[:idx])
        
        return result_set
    
    def _update_filter_stats(self, filter_time_ms: float, reduction_ratio: float) -> None:
        """Update running filter performance statistics."""
        self.filter_stats["total_queries"] += 1
        total_queries = self.filter_stats["total_queries"]
        
        # Update running averages
        current_avg_time = self.filter_stats["avg_filter_time_ms"]
        self.filter_stats["avg_filter_time_ms"] = (
            (current_avg_time * (total_queries - 1) + filter_time_ms) / total_queries
        )
        
        current_avg_reduction = self.filter_stats["avg_reduction_ratio"]
        self.filter_stats["avg_reduction_ratio"] = (
            (current_avg_reduction * (total_queries - 1) + reduction_ratio) / total_queries
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        return {
            "total_documents": self.total_documents,
            "indexed_fields": list(self.indexed_fields),
            "categorical_fields": list(self.keyword_indices.keys()),
            "numeric_fields": list(self.numeric_indices.keys()),
            "build_timestamp": self.build_timestamp,
            "performance_stats": self.filter_stats.copy(),
            "index_size_estimate": self._estimate_index_size()
        }
    
    def _estimate_index_size(self) -> Dict[str, int]:
        """Estimate memory usage of indices."""
        keyword_entries = sum(
            len(value_dict) for value_dict in self.keyword_indices.values()
        )
        numeric_entries = sum(
            len(value_list) for value_list in self.numeric_indices.values()
        )
        
        return {
            "keyword_entries": keyword_entries,
            "numeric_entries": numeric_entries,
            "total_entries": keyword_entries + numeric_entries
        } 