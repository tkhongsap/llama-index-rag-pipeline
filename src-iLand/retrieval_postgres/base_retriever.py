"""
Base PostgreSQL Retriever for iLand Data

Base class for all PostgreSQL-based retrieval strategies that query directly
from the database instead of loading embeddings from files.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore, TextNode

from .config import PostgresConfig
from .utils.db_connection import PostgresConnectionManager
from .utils.vector_ops import generate_embedding, cosine_similarity
from .utils.metadata_utils import MetadataUtils

logger = logging.getLogger(__name__)


class BasePostgresRetriever(ABC):
    """Base class for all PostgreSQL-based retrieval strategies."""
    
    def __init__(
        self,
        config: PostgresConfig,
        strategy_name: str
    ):
        """
        Initialize base PostgreSQL retriever.
        
        Args:
            config: PostgreSQL configuration
            strategy_name: Name of the retrieval strategy
        """
        self.config = config
        self.strategy_name = strategy_name
        
        # Initialize components
        self.connection_manager = PostgresConnectionManager(config)
        self.metadata_utils = MetadataUtils()
        
        # Validate configuration
        self.config.validate()
        
        logger.info(f"Initialized {strategy_name} PostgreSQL retriever")
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for the given query.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (optional)
            filters: Metadata filters (optional)
            
        Returns:
            List of NodeWithScore objects with strategy metadata
        """
        pass
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query."""
        try:
            print(f"[DEBUG] Using embedding model: {self.config.embedding_model}")
            return generate_embedding(query, self.config.embedding_model)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def _vector_similarity_search(
        self,
        query_embedding: List[float],
        table_name: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search against PostgreSQL table.
        
        Args:
            query_embedding: Query vector
            table_name: Target table name
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            metadata_filters: JSONB metadata filters
            select_fields: Fields to select
            
        Returns:
            List of result dictionaries
        """
        try:
            # Use connection manager for vector search
            results = self.connection_manager.vector_similarity_search(
                query_embedding=query_embedding,
                table_name=table_name,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                metadata_filters=metadata_filters,
                select_fields=select_fields
            )
            
            logger.debug(f"Vector search returned {len(results)} results from {table_name}")
            return results
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            raise
    
    def _format_results_as_nodes(
        self,
        results: List[Dict[str, Any]],
        include_embeddings: bool = False
    ) -> List[NodeWithScore]:
        """
        Convert database results to NodeWithScore objects.
        
        Args:
            results: Database query results
            include_embeddings: Whether to include embedding in metadata
            
        Returns:
            List of NodeWithScore objects
        """
        nodes = []
        
        for result in results:
            try:
                # Extract required fields
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                similarity_score = result.get("similarity_score", 0.0)
                
                # Create enriched metadata
                node_metadata = {
                    "retrieval_strategy": self.strategy_name,
                    "data_source": "iland_postgres",
                    "deed_id": result.get("deed_id", "unknown"),
                    "database_id": result.get("id"),
                    "similarity_score": similarity_score,
                    "chunk_index": result.get("chunk_index"),
                    "table_source": result.get("table_name")
                }
                
                # Merge with existing metadata
                if isinstance(metadata, dict):
                    node_metadata.update(metadata)
                
                # Include embedding if requested
                if include_embeddings and "embedding" in result:
                    node_metadata["embedding"] = result["embedding"]
                
                # Create TextNode
                text_node = TextNode(
                    text=text,
                    metadata=node_metadata
                )
                
                # Create NodeWithScore
                node_with_score = NodeWithScore(
                    node=text_node,
                    score=similarity_score
                )
                
                nodes.append(node_with_score)
                
            except Exception as e:
                logger.warning(f"Failed to format result as node: {e}")
                continue
        
        logger.debug(f"Formatted {len(nodes)} results as nodes")
        return nodes
    
    def _get_document_chunks(
        self,
        deed_id: str,
        chunk_indices: Optional[List[int]] = None,
        table_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get chunks for a specific document."""
        table_name = table_name or self.config.chunks_table
        return self.connection_manager.get_document_chunks(
            deed_id=deed_id,
            chunk_indices=chunk_indices,
            table_name=table_name
        )
    
    def _get_context_window(
        self,
        deed_id: str,
        center_chunk_index: int,
        window_size: int = 2,
        table_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get chunks around a center chunk for context window."""
        table_name = table_name or self.config.chunks_table
        return self.connection_manager.get_context_window(
            deed_id=deed_id,
            center_chunk_index=center_chunk_index,
            window_size=window_size,
            table_name=table_name
        )
    
    def _get_document_summary(
        self,
        deed_id: str,
        table_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get summary for a specific document."""
        table_name = table_name or self.config.summaries_table
        return self.connection_manager.get_document_summary(
            deed_id=deed_id,
            table_name=table_name
        )
    
    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        key_field: str = "id"
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on key field."""
        seen_keys = set()
        deduplicated = []
        
        for result in results:
            key = result.get(key_field)
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append(result)
        
        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
        return deduplicated
    
    def _apply_similarity_threshold(
        self,
        results: List[Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Filter results by similarity threshold."""
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        filtered = [
            result for result in results
            if result.get("similarity_score", 0.0) >= threshold
        ]
        
        logger.debug(f"Applied similarity threshold {threshold}: {len(results)} -> {len(filtered)} results")
        return filtered
    
    def _validate_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and clean metadata filters."""
        if not filters:
            return {}
        
        # Validate filters using metadata utils
        validation_errors = self.metadata_utils.validate_metadata_filters(filters)
        
        if validation_errors:
            error_msg = "; ".join([
                f"{field}: {', '.join(errors)}"
                for field, errors in validation_errors.items()
            ])
            raise ValueError(f"Invalid metadata filters: {error_msg}")
        
        return filters
    
    def _log_retrieval_metrics(
        self,
        query: str,
        num_results: int,
        execution_time: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log retrieval performance metrics."""
        log_info = {
            "strategy": self.strategy_name,
            "query_length": len(query),
            "results_count": num_results,
            "execution_time_ms": execution_time * 1000,
        }
        
        if additional_info:
            log_info.update(additional_info)
        
        logger.info(f"Retrieval metrics: {log_info}")
    
    @property
    def name(self) -> str:
        """Get the strategy name."""
        return self.strategy_name
    
    def close(self) -> None:
        """Close connections and cleanup resources."""
        if hasattr(self, 'connection_manager'):
            self.connection_manager.close()
        logger.info(f"Closed {self.strategy_name} retriever")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 