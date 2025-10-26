"""
Metadata Retriever Adapter for Sales Promotions

Strategy 4: Metadata-filtered retrieval with fast in-memory indexing.
Best for analytical queries like "Active promotions for เอเย่นต์ with discount > 25%"

Performance Target: < 50ms
"""

import json
from typing import List, Optional, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from .base import BaseRetrieverAdapter
from ..fast_metadata_index import FastMetadataIndex


class MetadataRetrieverAdapter(BaseRetrieverAdapter):
    """
    Metadata-filtered retriever for sales promotions.
    
    Uses fast in-memory metadata index for sub-50ms filtering on 29 fields.
    Supports auto-filter detection from natural language queries using GPT-5.
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        metadata_index: FastMetadataIndex,
        llm: Optional[OpenAI] = None,
        auto_detect_filters: bool = True,
        top_k: int = 5
    ):
        """
        Initialize metadata retriever.
        
        Args:
            vector_index: VectorStoreIndex for semantic search after filtering
            metadata_index: FastMetadataIndex for fast filtering
            llm: LLM for auto filter detection (GPT-5)
            auto_detect_filters: Enable automatic filter detection from query
            top_k: Number of results to return
        """
        super().__init__("metadata")
        self.vector_index = vector_index
        self.metadata_index = metadata_index
        self.llm = llm
        self.auto_detect_filters = auto_detect_filters
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using metadata filtering + semantic search.
        
        Args:
            query: Search query (supports Thai text)
            top_k: Number of results (uses default if None)
            filters: Explicit metadata filters (optional, auto-detected if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy metadata
        """
        k = top_k if top_k is not None else self.top_k
        
        # Auto-detect filters if enabled and not provided
        if filters is None and self.auto_detect_filters and self.llm:
            filters = self._auto_detect_filters(query)
        
        # If no filters, fall back to basic vector search
        if not filters:
            retriever = self.vector_index.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(query)
        else:
            # Apply metadata filtering first
            matching_doc_ids = set(self.metadata_index.filter(filters))
            print(f"  🔍 Metadata filter matched {len(matching_doc_ids)} documents")
            
            # Then perform vector search within filtered results
            # Retrieve more results and filter post-retrieval
            retriever = self.vector_index.as_retriever(similarity_top_k=min(k * 5, 50))
            all_nodes = retriever.retrieve(query)
            
            # Filter to matching doc_ids (check multiple possible ID fields)
            filtered_nodes = []
            for node in all_nodes:
                node_id = self._get_node_doc_id(node)
                # Check if node_id or any metadata ID matches
                if (node_id in matching_doc_ids or 
                    node.node.metadata.get('document_id') in matching_doc_ids or
                    node.node.metadata.get('doc_id') in matching_doc_ids):
                    filtered_nodes.append(node)
                    if len(filtered_nodes) >= k:
                        break
            
            nodes = filtered_nodes
        
        # Tag nodes with strategy info and applied filters
        tagged_nodes = self._tag_nodes_with_strategy(nodes)
        
        # Add filter info to metadata
        if filters:
            for node in tagged_nodes:
                node.node.metadata['applied_filters'] = filters
        
        return tagged_nodes
    
    def _get_node_doc_id(self, node: NodeWithScore) -> Optional[str]:
        """Extract document ID from node."""
        # Try node ID first
        if hasattr(node.node, 'id_'):
            node_id = node.node.id_
            # Return node_id if it exists
            if node_id:
                return node_id
        
        # Fall back to metadata
        if hasattr(node.node, 'metadata'):
            return (
                node.node.metadata.get('document_id') or
                node.node.metadata.get('doc_id')
            )
        return None
    
    def _auto_detect_filters(self, query: str) -> Dict[str, Any]:
        """
        Auto-detect metadata filters from natural language query using GPT-5.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary of detected filters
        """
        if not self.llm:
            return {}
        
        prompt = f"""
Extract metadata filters from this sales promotion query.

Query: {query}

Available fields and values:
- customer_type: "เอเย่นต์", "ซับเอเย่นต์", "Super Sub", "รถขายปลีก", "Super Shop"
- customer_tier: "tier_1", "tier_2", "tier_3"
- product_category: "water", "beer", "spirits", "soda", "other"
- product_brand: "ช้าง", "สิงห์", "ลีโอ", etc.
- discount_percentage: numeric (0-100)
- is_active: boolean
- is_wholesale: boolean
- is_retail: boolean
- expires_soon: boolean
- buy_quantity: numeric
- free_quantity: numeric
- duration_days: numeric
- start_date_gregorian: ISO date string
- end_date_gregorian: ISO date string

Operators:
- $gt: greater than
- $gte: greater than or equal
- $lt: less than
- $lte: less than or equal
- $eq: equal
- $ne: not equal

Return ONLY valid JSON with detected filters, or empty dict {{}} if no filters:

Examples:
- "Active promotions for เอเย่นต์" → {{"is_active": true, "customer_type": "เอเย่นต์"}}
- "Beer with discount > 25%" → {{"product_category": "beer", "discount_percentage": {{"$gt": 25}}}}
- "Water promotions" → {{}}  (no specific filters, just semantic search)

Your response (JSON only):
"""
        
        try:
            response = self.llm.complete(prompt)
            filters = json.loads(response.text.strip())
            
            if filters:
                print(f"  🔍 Auto-detected filters: {filters}")
            
            return filters
        except Exception as e:
            print(f"  ⚠️  Filter auto-detection failed: {e}")
            return {}
