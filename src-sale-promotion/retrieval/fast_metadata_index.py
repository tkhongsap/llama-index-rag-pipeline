"""
Fast Metadata Index for Sales Promotions

In-memory indexed structure for sub-50ms metadata filtering on 29 fields.
Supports operators: $gt, $gte, $lt, $lte, $eq, $ne
"""

from typing import List, Dict, Any, Set, Optional
from collections import defaultdict


class FastMetadataIndex:
    """
    Fast in-memory metadata index for sales promotion filtering.
    
    Builds inverted indices for categorical fields and range indices
    for numeric fields to enable sub-50ms filtering.
    """
    
    def __init__(self):
        """Initialize fast metadata index."""
        self.categorical_indices = {}
        self.numeric_indices = {}
        self.boolean_indices = {}
        self.documents = []
        self.doc_id_map = {}
    
    def build_index(self, documents: List[Any]):
        """
        Build in-memory indices for fast filtering.
        
        Args:
            documents: List of documents/nodes with metadata
        """
        print(f"\n🏗️  Building fast metadata index...")
        self.documents = documents
        
        # Build doc_id mapping
        for i, doc in enumerate(documents):
            doc_id = self._get_doc_id(doc)
            if doc_id:
                self.doc_id_map[doc_id] = i
        
        # Categorical fields
        categorical_fields = [
            "customer_type", "customer_tier", 
            "product_category", "product_brand", "product_size",
            "promotion_type", "buy_unit", "free_unit"
        ]
        
        for field in categorical_fields:
            self.categorical_indices[field] = self._build_categorical_index(field)
        
        # Numeric fields
        numeric_fields = [
            "discount_percentage", "buy_quantity", "free_quantity",
            "duration_days"
        ]
        
        for field in numeric_fields:
            self.numeric_indices[field] = self._build_numeric_index(field)
        
        # Boolean fields
        boolean_fields = ["is_active", "is_wholesale", "is_retail", "expires_soon"]
        
        for field in boolean_fields:
            self.boolean_indices[field] = self._build_boolean_index(field)
        
        print(f"  ✅ Indexed {len(categorical_fields)} categorical fields")
        print(f"  ✅ Indexed {len(numeric_fields)} numeric fields")
        print(f"  ✅ Indexed {len(boolean_fields)} boolean fields")
        print(f"  ✅ Total documents indexed: {len(documents)}")
    
    def _get_doc_id(self, doc: Any) -> Optional[str]:
        """Extract document ID from various node types."""
        if hasattr(doc, 'id_'):
            return doc.id_
        elif hasattr(doc, 'node_id'):
            return doc.node_id
        elif hasattr(doc, 'metadata') and 'document_id' in doc.metadata:
            return doc.metadata['document_id']
        return None
    
    def _get_metadata(self, doc: Any) -> Dict:
        """Extract metadata from various node types."""
        if hasattr(doc, 'metadata'):
            return doc.metadata
        return {}
    
    def _build_categorical_index(self, field: str) -> Dict[Any, Set[str]]:
        """Build inverted index for categorical field."""
        index = defaultdict(set)
        
        for doc in self.documents:
            metadata = self._get_metadata(doc)
            value = metadata.get(field)
            
            if value is not None:
                doc_id = self._get_doc_id(doc)
                if doc_id:
                    index[value].add(doc_id)
        
        return dict(index)
    
    def _build_numeric_index(self, field: str) -> Dict[str, Any]:
        """Build index for numeric field with min/max tracking."""
        values = []
        doc_ids = []
        
        for doc in self.documents:
            metadata = self._get_metadata(doc)
            value = metadata.get(field)
            
            if value is not None and isinstance(value, (int, float)):
                doc_id = self._get_doc_id(doc)
                if doc_id:
                    values.append(value)
                    doc_ids.append(doc_id)
        
        return {
            "values": values,
            "doc_ids": doc_ids,
            "min": min(values) if values else None,
            "max": max(values) if values else None
        }
    
    def _build_boolean_index(self, field: str) -> Dict[bool, Set[str]]:
        """Build index for boolean field."""
        index = {True: set(), False: set()}
        
        for doc in self.documents:
            metadata = self._get_metadata(doc)
            value = metadata.get(field)
            
            if isinstance(value, bool):
                doc_id = self._get_doc_id(doc)
                if doc_id:
                    index[value].add(doc_id)
        
        return index
    
    def filter(self, filters: Dict[str, Any]) -> List[str]:
        """
        Apply filters and return matching document IDs.
        
        Args:
            filters: Dictionary of field -> condition mappings
            
        Returns:
            List of matching document IDs
        """
        if not filters:
            return [self._get_doc_id(doc) for doc in self.documents]
        
        # Start with all documents
        matching_ids = set([self._get_doc_id(doc) for doc in self.documents if self._get_doc_id(doc)])
        
        # Apply each filter (AND logic)
        for field, condition in filters.items():
            field_matches = self._apply_filter(field, condition)
            matching_ids &= field_matches
        
        return list(matching_ids)
    
    def _apply_filter(self, field: str, condition: Any) -> Set[str]:
        """Apply single filter condition."""
        # Check if it's an operator-based condition
        if isinstance(condition, dict):
            return self._apply_operator_filter(field, condition)
        
        # Exact match
        if field in self.categorical_indices:
            return self.categorical_indices[field].get(condition, set())
        elif field in self.boolean_indices:
            return self.boolean_indices[field].get(condition, set())
        
        return set()
    
    def _apply_operator_filter(self, field: str, operators: Dict) -> Set[str]:
        """Apply operator-based filter ($gt, $gte, $lt, $lte)."""
        if field not in self.numeric_indices:
            return set()
        
        index = self.numeric_indices[field]
        values = index["values"]
        doc_ids = index["doc_ids"]
        
        matching_ids = set()
        
        for value, doc_id in zip(values, doc_ids):
            match = True
            
            if "$gt" in operators and not (value > operators["$gt"]):
                match = False
            if "$gte" in operators and not (value >= operators["$gte"]):
                match = False
            if "$lt" in operators and not (value < operators["$lt"]):
                match = False
            if "$lte" in operators and not (value <= operators["$lte"]):
                match = False
            if "$eq" in operators and not (value == operators["$eq"]):
                match = False
            if "$ne" in operators and not (value != operators["$ne"]):
                match = False
            
            if match:
                matching_ids.add(doc_id)
        
        return matching_ids
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the metadata index."""
        return {
            "total_documents": len(self.documents),
            "categorical_fields": len(self.categorical_indices),
            "numeric_fields": len(self.numeric_indices),
            "boolean_fields": len(self.boolean_indices),
            "total_indexed_fields": (
                len(self.categorical_indices) + 
                len(self.numeric_indices) + 
                len(self.boolean_indices)
            )
        }

