"""
Basic PostgreSQL Retriever for iLand Data

Implements direct vector similarity search using PostgreSQL/pgVector.
This is the most straightforward retrieval strategy.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from llama_index.core.schema import NodeWithScore

from ..base_retriever import BasePostgresRetriever
from ..config import PostgresConfig

logger = logging.getLogger(__name__)


class BasicPostgresRetriever(BasePostgresRetriever):
    """
    Basic vector similarity retriever using PostgreSQL/pgVector.
    
    This strategy performs direct cosine similarity search against
    the embedding vectors stored in PostgreSQL tables.
    """
    
    def __init__(
        self,
        config: PostgresConfig,
        default_table: str = "iland_chunks"
    ):
        """
        Initialize basic PostgreSQL retriever.
        
        Args:
            config: PostgreSQL configuration
            default_table: Default table for search (chunks, summaries, combined)
        """
        super().__init__(config, "basic_postgres")
        self.default_table = default_table
        
        # Validate table name
        valid_tables = [
            config.chunks_table,
            config.summaries_table,
            config.indexnodes_table,
            config.combined_table
        ]
        
        if default_table not in valid_tables:
            logger.warning(f"Table {default_table} not in valid tables: {valid_tables}")
            self.default_table = config.chunks_table
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using basic vector similarity search.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (uses config default if None)
            filters: Metadata filters for pre-filtering results
            
        Returns:
            List of NodeWithScore objects with similarity scores
        """
        start_time = time.time()
        
        # Set defaults
        top_k = top_k or self.config.default_top_k
        top_k = min(top_k, self.config.max_top_k)  # Enforce max limit
        
        # Validate filters
        validated_filters = self._validate_filters(filters)
        
        try:
            # Generate query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Execute vector similarity search
            results = self._vector_similarity_search(
                query_embedding=query_embedding,
                table_name=self.default_table,
                top_k=top_k,
                similarity_threshold=self.config.similarity_threshold,
                metadata_filters=validated_filters
            )
            
            # Apply additional similarity threshold if needed
            filtered_results = self._apply_similarity_threshold(results)
            
            # Convert to NodeWithScore objects
            nodes = self._format_results_as_nodes(filtered_results)
            
            # Log metrics
            execution_time = time.time() - start_time
            self._log_retrieval_metrics(
                query=query,
                num_results=len(nodes),
                execution_time=execution_time,
                additional_info={
                    "table": self.default_table,
                    "requested_top_k": top_k,
                    "filters_applied": len(validated_filters) > 0
                }
            )
            
            logger.info(f"Basic retrieval completed: {len(nodes)} nodes in {execution_time:.3f}s")
            return nodes
            
        except Exception as e:
            logger.error(f"Basic retrieval failed: {e}")
            raise
    
    def retrieve_from_table(
        self,
        query: str,
        table_name: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes from a specific table.
        
        Args:
            query: The search query
            table_name: Specific table to search
            top_k: Number of nodes to retrieve
            filters: Metadata filters
            
        Returns:
            List of NodeWithScore objects
        """
        # Temporarily change default table
        original_table = self.default_table
        self.default_table = table_name
        
        try:
            return self.retrieve(query, top_k, filters)
        finally:
            # Restore original table
            self.default_table = original_table
    
    def retrieve_multi_table(
        self,
        query: str,
        tables: List[str],
        top_k_per_table: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        merge_results: bool = True
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes from multiple tables.
        
        Args:
            query: The search query
            tables: List of table names to search
            top_k_per_table: Number of results per table
            filters: Metadata filters
            merge_results: Whether to merge and rerank results
            
        Returns:
            List of NodeWithScore objects
        """
        start_time = time.time()
        
        # Set defaults
        top_k_per_table = top_k_per_table or max(1, self.config.default_top_k // len(tables))
        
        all_results = []
        
        try:
            # Generate query embedding once
            query_embedding = self._get_query_embedding(query)
            validated_filters = self._validate_filters(filters)
            
            # Search each table
            for table in tables:
                try:
                    table_results = self._vector_similarity_search(
                        query_embedding=query_embedding,
                        table_name=table,
                        top_k=top_k_per_table,
                        similarity_threshold=self.config.similarity_threshold,
                        metadata_filters=validated_filters
                    )
                    
                    # Add table source to results
                    for result in table_results:
                        result["table_name"] = table
                    
                    all_results.extend(table_results)
                    
                except Exception as e:
                    logger.warning(f"Failed to search table {table}: {e}")
                    continue
            
            if merge_results:
                # Sort by similarity score (descending)
                all_results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
                
                # Take top results across all tables
                final_top_k = self.config.default_top_k
                all_results = all_results[:final_top_k]
                
                # Deduplicate by content or deed_id
                all_results = self._deduplicate_results(all_results, key_field="deed_id")
            
            # Convert to nodes
            nodes = self._format_results_as_nodes(all_results)
            
            # Log metrics
            execution_time = time.time() - start_time
            self._log_retrieval_metrics(
                query=query,
                num_results=len(nodes),
                execution_time=execution_time,
                additional_info={
                    "tables_searched": tables,
                    "top_k_per_table": top_k_per_table,
                    "merged_results": merge_results
                }
            )
            
            logger.info(f"Multi-table retrieval completed: {len(nodes)} nodes from {len(tables)} tables")
            return nodes
            
        except Exception as e:
            logger.error(f"Multi-table retrieval failed: {e}")
            raise
    
    @classmethod
    def create_for_chunks(cls, config: PostgresConfig) -> "BasicPostgresRetriever":
        """Create retriever for chunk table."""
        return cls(config, default_table=config.chunks_table)
    
    @classmethod
    def create_for_summaries(cls, config: PostgresConfig) -> "BasicPostgresRetriever":
        """Create retriever for summaries table."""
        return cls(config, default_table=config.summaries_table)
    
    @classmethod
    def create_for_combined(cls, config: PostgresConfig) -> "BasicPostgresRetriever":
        """Create retriever for combined table."""
        return cls(config, default_table=config.combined_table) 