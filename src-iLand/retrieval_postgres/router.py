"""
PostgreSQL Router Retriever for iLand Data

Extends the local router to work with PostgreSQL-stored data.
Maintains complete parity with local implementation while adding PostgreSQL-specific features.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import QueryBundle, NodeWithScore

from ..retrieval.router import iLandRouterRetriever
from ..retrieval.retrievers.base import BaseRetrieverAdapter
from ..retrieval.cache import iLandCacheManager
from .index_classifier import PostgresIndexClassifier, create_postgres_classifier
from .config import PostgresRetrievalConfig


class PostgresRouterRetriever(iLandRouterRetriever):
    """PostgreSQL-based router retriever maintaining parity with local implementation."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 retrievers: Optional[Dict[str, Dict[str, BaseRetrieverAdapter]]] = None,
                 index_classifier: Optional[PostgresIndexClassifier] = None,
                 strategy_selector: Optional[str] = "llm",
                 llm_strategy_mode: Optional[str] = "enhanced",
                 api_key: Optional[str] = None,
                 cache_manager: Optional[iLandCacheManager] = None,
                 enable_caching: bool = True,
                 enable_query_logging: bool = True):
        """
        Initialize PostgreSQL router retriever.
        
        Args:
            config: PostgreSQL configuration
            retrievers: Dict mapping index_name -> {strategy_name -> adapter}
            index_classifier: Index classifier (creates PostgreSQL version if None)
            strategy_selector: Strategy selection method ("llm", "heuristic", "round_robin")
            llm_strategy_mode: LLM strategy mode ("enhanced", "simple")
            api_key: OpenAI API key
            cache_manager: Cache manager instance
            enable_caching: Whether to enable caching
            enable_query_logging: Whether to log queries to PostgreSQL
        """
        # Initialize configuration
        self.config = config or PostgresRetrievalConfig()
        self.enable_query_logging = enable_query_logging
        
        # Create PostgreSQL index classifier if not provided
        if index_classifier is None:
            index_classifier = create_postgres_classifier(
                config=self.config,
                api_key=api_key
            )
        
        # Initialize parent class
        super().__init__(
            retrievers=retrievers or {},
            index_classifier=index_classifier,
            strategy_selector=strategy_selector,
            llm_strategy_mode=llm_strategy_mode,
            api_key=api_key,
            cache_manager=cache_manager,
            enable_caching=enable_caching
        )
        
        # Create query logging table if enabled
        if self.enable_query_logging:
            self._create_query_log_table()
    
    def _create_query_log_table(self):
        """Create query log table in PostgreSQL."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iland_query_log (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    selected_index VARCHAR(255),
                    selected_strategy VARCHAR(255),
                    index_confidence FLOAT,
                    strategy_confidence FLOAT,
                    index_method VARCHAR(50),
                    strategy_method VARCHAR(50),
                    result_count INTEGER,
                    latency_ms FLOAT,
                    cache_hit BOOLEAN DEFAULT false,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_log_created_at 
                ON iland_query_log(created_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_log_index_strategy 
                ON iland_query_log(selected_index, selected_strategy)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not create query log table: {e}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Main retrieval method with PostgreSQL logging and enhanced caching.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of retrieved nodes with routing metadata
        """
        query = query_bundle.query_str
        start_time = time.time()
        
        # Use parent's retrieval logic
        nodes = super()._retrieve(query_bundle)
        
        # Calculate final latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log to PostgreSQL if enabled
        if self.enable_query_logging and nodes:
            # Extract metadata from first node
            first_node = nodes[0]
            metadata = first_node.node.metadata if hasattr(first_node.node, 'metadata') else {}
            
            self._log_query(
                query=query,
                selected_index=metadata.get("selected_index", "unknown"),
                selected_strategy=metadata.get("selected_strategy", "unknown"),
                index_confidence=metadata.get("index_confidence", 0),
                strategy_confidence=metadata.get("strategy_confidence", 0),
                index_method=metadata.get("index_method", "unknown"),
                strategy_method=metadata.get("strategy_method", "unknown"),
                result_count=len(nodes),
                latency_ms=latency_ms,
                cache_hit=metadata.get("cache_hit", False)
            )
        
        return nodes
    
    def _log_query(self, **kwargs):
        """Log query details to PostgreSQL."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Extract metadata fields
            metadata = {
                "llm_strategy_mode": self.llm_strategy_mode,
                "strategy_selector": self.strategy_selector,
                "caching_enabled": self.enable_caching
            }
            
            cursor.execute("""
                INSERT INTO iland_query_log 
                (query, selected_index, selected_strategy, index_confidence, 
                 strategy_confidence, index_method, strategy_method, result_count,
                 latency_ms, cache_hit, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                kwargs.get("query"),
                kwargs.get("selected_index"),
                kwargs.get("selected_strategy"),
                kwargs.get("index_confidence"),
                kwargs.get("strategy_confidence"),
                kwargs.get("index_method"),
                kwargs.get("strategy_method"),
                kwargs.get("result_count"),
                kwargs.get("latency_ms"),
                kwargs.get("cache_hit"),
                json.dumps(metadata)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Query logging error: {e}")
    
    def get_query_stats(self, 
                       hours: int = 24,
                       index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get query statistics from PostgreSQL.
        
        Args:
            hours: Number of hours to look back
            index_name: Optional filter by index name
            
        Returns:
            Dictionary with query statistics
        """
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Base WHERE clause
            where_clauses = ["created_at > NOW() - INTERVAL '%s hours'"]
            params = [hours]
            
            if index_name:
                where_clauses.append("selected_index = %s")
                params.append(index_name)
            
            where_clause = " AND ".join(where_clauses)
            
            # Get overall stats
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_queries,
                    COUNT(DISTINCT query) as unique_queries,
                    AVG(latency_ms) as avg_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    MAX(latency_ms) as max_latency_ms,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency_ms,
                    AVG(result_count) as avg_results,
                    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                    AVG(index_confidence) as avg_index_confidence,
                    AVG(strategy_confidence) as avg_strategy_confidence
                FROM iland_query_log
                WHERE {where_clause}
            """, params)
            
            overall_stats = dict(cursor.fetchone())
            
            # Calculate cache hit rate
            if overall_stats['total_queries'] > 0:
                overall_stats['cache_hit_rate'] = (
                    overall_stats['cache_hits'] / overall_stats['total_queries']
                )
            else:
                overall_stats['cache_hit_rate'] = 0
            
            # Get strategy distribution
            cursor.execute(f"""
                SELECT 
                    selected_strategy,
                    COUNT(*) as count,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(result_count) as avg_results
                FROM iland_query_log
                WHERE {where_clause}
                GROUP BY selected_strategy
                ORDER BY count DESC
            """, params)
            
            strategy_stats = [dict(row) for row in cursor.fetchall()]
            
            # Get index distribution
            cursor.execute(f"""
                SELECT 
                    selected_index,
                    COUNT(*) as count,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(result_count) as avg_results
                FROM iland_query_log
                WHERE {where_clause}
                GROUP BY selected_index
                ORDER BY count DESC
            """, params)
            
            index_stats = [dict(row) for row in cursor.fetchall()]
            
            # Get method distribution
            cursor.execute(f"""
                SELECT 
                    strategy_method,
                    COUNT(*) as count
                FROM iland_query_log
                WHERE {where_clause}
                GROUP BY strategy_method
                ORDER BY count DESC
            """, params)
            
            method_stats = [dict(row) for row in cursor.fetchall()]
            
            # Get hourly distribution
            cursor.execute(f"""
                SELECT 
                    DATE_TRUNC('hour', created_at) as hour,
                    COUNT(*) as query_count,
                    AVG(latency_ms) as avg_latency_ms
                FROM iland_query_log
                WHERE {where_clause}
                GROUP BY hour
                ORDER BY hour DESC
                LIMIT 24
            """, params)
            
            hourly_stats = [
                {
                    "hour": row['hour'].isoformat(),
                    "query_count": row['query_count'],
                    "avg_latency_ms": float(row['avg_latency_ms'])
                }
                for row in cursor.fetchall()
            ]
            
            cursor.close()
            conn.close()
            
            return {
                "overall": overall_stats,
                "by_strategy": strategy_stats,
                "by_index": index_stats,
                "by_method": method_stats,
                "hourly": hourly_stats,
                "period_hours": hours,
                "filtered_by_index": index_name
            }
            
        except Exception as e:
            print(f"Error getting query stats: {e}")
            return {}
    
    def get_popular_queries(self, 
                          limit: int = 10,
                          hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get most popular queries from PostgreSQL.
        
        Args:
            limit: Number of queries to return
            hours: Number of hours to look back
            
        Returns:
            List of popular queries with counts
        """
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    query,
                    COUNT(*) as count,
                    MAX(created_at) as last_seen,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(result_count) as avg_results,
                    array_agg(DISTINCT selected_strategy) as strategies_used
                FROM iland_query_log
                WHERE created_at > NOW() - INTERVAL '%s hours'
                GROUP BY query
                ORDER BY count DESC
                LIMIT %s
            """, (hours, limit))
            
            popular_queries = []
            for row in cursor.fetchall():
                popular_queries.append({
                    "query": row['query'],
                    "count": row['count'],
                    "last_seen": row['last_seen'].isoformat(),
                    "avg_latency_ms": float(row['avg_latency_ms']),
                    "avg_results": float(row['avg_results']),
                    "strategies_used": row['strategies_used']
                })
            
            cursor.close()
            conn.close()
            
            return popular_queries
            
        except Exception as e:
            print(f"Error getting popular queries: {e}")
            return []
    
    def cleanup_old_logs(self, days: int = 30):
        """
        Clean up old query logs from PostgreSQL.
        
        Args:
            days: Number of days to keep
        """
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM iland_query_log
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days,))
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"Cleaned up {deleted_count} old query log entries")
            
        except Exception as e:
            print(f"Error cleaning up logs: {e}")
    
    @classmethod
    def from_config(cls,
                   config: Optional[PostgresRetrievalConfig] = None,
                   api_key: Optional[str] = None,
                   strategy_selector: str = "llm",
                   llm_strategy_mode: str = "enhanced",
                   enable_caching: bool = True,
                   enable_query_logging: bool = True) -> "PostgresRouterRetriever":
        """
        Create PostgreSQL router from configuration.
        
        Args:
            config: PostgreSQL configuration
            api_key: OpenAI API key
            strategy_selector: Strategy selection method
            llm_strategy_mode: LLM strategy mode
            enable_caching: Whether to enable caching
            enable_query_logging: Whether to log queries
            
        Returns:
            PostgresRouterRetriever instance
        """
        return cls(
            config=config,
            api_key=api_key,
            strategy_selector=strategy_selector,
            llm_strategy_mode=llm_strategy_mode,
            enable_caching=enable_caching,
            enable_query_logging=enable_query_logging
        )