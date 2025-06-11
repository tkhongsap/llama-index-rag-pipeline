"""
Database connection management for PostgreSQL retrieval

Handles connection pooling, transaction management, and database operations.
"""

import logging
import psycopg2
import psycopg2.pool
from typing import List, Dict, Any, Optional, Union, ContextManager
from contextlib import contextmanager
from ..config import PostgresConfig

logger = logging.getLogger(__name__)


class PostgresConnectionManager:
    """Manages PostgreSQL connections with pooling for retrieval operations."""
    
    def __init__(self, config: PostgresConfig):
        """Initialize connection manager with configuration."""
        self.config = config
        self.pool = None
        self._setup_connection_pool()
    
    def _setup_connection_pool(self) -> None:
        """Setup connection pool with configuration."""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                **self.config.connection_params
            )
            logger.info(f"Created connection pool")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> ContextManager[psycopg2.extensions.connection]:
        """Get a database connection from the pool."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self) -> ContextManager[psycopg2.extensions.cursor]:
        """Get a database cursor with automatic connection management."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Cursor operation error: {e}")
                raise
            finally:
                cursor.close()
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[tuple] = None,
        fetch_results: bool = True
    ) -> Optional[List[tuple]]:
        """Execute a single query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            if fetch_results:
                return cursor.fetchall()
            return None
    
    def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> None:
        """Execute multiple queries with the same statement."""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
    
    def vector_similarity_search(
        self,
        query_embedding: List[float],
        table_name: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search with optional filtering.
        
        Args:
            query_embedding: Query vector
            table_name: Target table name
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            metadata_filters: JSONB metadata filters
            select_fields: Fields to select (defaults to common fields)
            
        Returns:
            List of result dictionaries
        """
        # Default fields to select
        if select_fields is None:
            select_fields = ["id", "text", "metadata", "node_id"]
        
        # แปลง query_embedding เป็น list ของ float ธรรมดา
        query_embedding = [float(x) for x in query_embedding]
        
        # Build base query
        fields_sql = ", ".join(select_fields)
        similarity_sql = "1 - (embedding <=> %s::vector) AS similarity_score"
        
        sql = f"""
        SELECT {fields_sql}, {similarity_sql}
        FROM {table_name}
        WHERE 1=1
        """
        
        params = [query_embedding]
        
        # Add similarity threshold filter
        if similarity_threshold is not None:
            sql += " AND (1 - (embedding <=> %s::vector)) >= %s"
            params.extend([query_embedding, similarity_threshold])
        
        # Add metadata filters
        if metadata_filters:
            for key, value in metadata_filters.items():
                if isinstance(value, (list, tuple)):
                    # Handle IN queries
                    placeholders = ", ".join(["%s"] * len(value))
                    sql += f" AND metadata->>'{key}' IN ({placeholders})"
                    params.extend(value)
                else:
                    # Handle exact match
                    sql += f" AND metadata->>'{key}' = %s"
                    params.append(str(value))
        
        # Add ordering and limit
        sql += f"""
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        params.extend([query_embedding, top_k])
        
        # Execute query
        results = self.execute_query(sql, tuple(params))
        
        # Format results as dictionaries
        formatted_results = []
        for row in results:
            result_dict = {}
            for i, field in enumerate(select_fields + ["similarity_score"]):
                result_dict[field] = row[i]
            formatted_results.append(result_dict)
        
        return formatted_results
    
    def get_document_chunks(
        self,
        deed_id: str,
        chunk_indices: Optional[List[int]] = None,
        table_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get chunks for a specific document."""
        table_name = table_name or self.config.chunks_table
        # ไม่มี deed_id ใน schema จริง
        sql = f"""
        SELECT id, text, metadata, node_id, chunk_index
        FROM {table_name}
        WHERE node_id = %s
        """
        params = [deed_id]
        if chunk_indices:
            placeholders = ", ".join(["%s"] * len(chunk_indices))
            sql += f" AND chunk_index IN ({placeholders})"
            params.extend(chunk_indices)
        sql += " ORDER BY chunk_index"
        results = self.execute_query(sql, tuple(params))
        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": row[0],
                "text": row[1],
                "metadata": row[2],
                "node_id": row[3],
                "chunk_index": row[4]
            })
        return formatted_results
    
    def get_context_window(
        self,
        deed_id: str,
        center_chunk_index: int,
        window_size: int = 2,
        table_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get chunks around a center chunk for context window."""
        table_name = table_name or self.config.chunks_table
        start_index = max(0, center_chunk_index - window_size)
        end_index = center_chunk_index + window_size
        sql = f"""
        SELECT id, text, metadata, node_id, chunk_index
        FROM {table_name}
        WHERE node_id = %s 
        AND chunk_index BETWEEN %s AND %s
        ORDER BY chunk_index
        """
        results = self.execute_query(sql, (deed_id, start_index, end_index))
        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": row[0],
                "text": row[1],
                "metadata": row[2],
                "node_id": row[3],
                "chunk_index": row[4]
            })
        return formatted_results
    
    def get_document_summary(
        self,
        deed_id: str,
        table_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get summary for a specific document."""
        table_name = table_name or self.config.summaries_table
        sql = f"""
        SELECT id, text, metadata, node_id, summary_text
        FROM {table_name}
        WHERE node_id = %s
        LIMIT 1
        """
        results = self.execute_query(sql, (deed_id,))
        if results:
            row = results[0]
            return {
                "id": row[0],
                "text": row[1],
                "metadata": row[2],
                "node_id": row[3],
                "summary_text": row[4]
            }
        return None
    
    def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            self.pool.closeall()
            self.pool = None
            logger.info("Connection pool closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 