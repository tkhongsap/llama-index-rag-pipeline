"""
PostgreSQL Index Classifier for iLand Data

Extends the local index classifier to work with PostgreSQL-stored indices.
Maintains complete parity with local implementation while adding PostgreSQL-specific features.
"""

import os
import asyncio
import asyncpg
from typing import Dict, Any, Optional, List
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import QueryBundle
from llama_index.core.llms import LLM

from src_iLand.retrieval.index_classifier import iLandIndexClassifier, DEFAULT_ILAND_INDICES
from .config import PostgresRetrievalConfig


class PostgresIndexClassifier(iLandIndexClassifier):
    """PostgreSQL-based index classifier maintaining parity with local implementation."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 available_indices: Optional[Dict[str, str]] = None,
                 api_key: Optional[str] = None,
                 mode: str = "llm"):
        """
        Initialize PostgreSQL index classifier.
        
        Args:
            config: PostgreSQL configuration
            available_indices: Dict mapping index names to descriptions
            api_key: OpenAI API key
            mode: Classification mode ("llm" or "embedding")
        """
        # Initialize configuration
        self.config = config or PostgresRetrievalConfig()
        
        # Get indices from PostgreSQL or use defaults
        if available_indices is None:
            available_indices = self._load_indices_from_postgres()
            if not available_indices:
                available_indices = DEFAULT_ILAND_INDICES
        
        # Initialize parent class
        super().__init__(available_indices, api_key, mode)
        
        # PostgreSQL-specific attributes
        self._conn_pool = None
        self._async_pool = None
        
        # Verify indices exist in PostgreSQL
        self._verify_postgres_indices()
    
    def _load_indices_from_postgres(self) -> Dict[str, str]:
        """Load available indices from PostgreSQL metadata."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check if we have a metadata table for indices
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'iland_indices'
                )
            """)
            
            if cursor.fetchone()['exists']:
                # Load indices from metadata table
                cursor.execute("""
                    SELECT index_name, description, is_active
                    FROM iland_indices
                    WHERE is_active = true
                """)
                
                indices = {}
                for row in cursor.fetchall():
                    indices[row['index_name']] = row['description']
                
                cursor.close()
                conn.close()
                return indices
            else:
                cursor.close()
                conn.close()
                return {}
                
        except Exception as e:
            print(f"Warning: Could not load indices from PostgreSQL: {e}")
            return {}
    
    def _verify_postgres_indices(self):
        """Verify that indices exist in PostgreSQL."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Create indices metadata table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iland_indices (
                    index_name VARCHAR(255) PRIMARY KEY,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT true,
                    metadata_ JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            # Insert or update index metadata
            for index_name, description in self.available_indices.items():
                cursor.execute("""
                    INSERT INTO iland_indices (index_name, description)
                    VALUES (%s, %s)
                    ON CONFLICT (index_name) 
                    DO UPDATE SET 
                        description = EXCLUDED.description,
                        updated_at = CURRENT_TIMESTAMP
                """, (index_name, description))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not verify PostgreSQL indices: {e}")
    
    def classify_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Classify query with optional PostgreSQL caching.
        
        Args:
            query: The user query (may contain Thai text)
            use_cache: Whether to use cached results
            
        Returns:
            Dict with selected index and metadata
        """
        # Check cache first if enabled
        if use_cache and self.config.enable_cache:
            cached_result = self._get_cached_classification(query)
            if cached_result:
                return cached_result
        
        # Perform classification using parent method
        result = super().classify_query(query)
        
        # Cache the result if enabled
        if use_cache and self.config.enable_cache:
            self._cache_classification(query, result)
        
        # Log classification to PostgreSQL for analytics
        self._log_classification(query, result)
        
        return result
    
    def _get_cached_classification(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached classification result from PostgreSQL."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Create cache table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iland_classification_cache (
                    query_hash VARCHAR(64) PRIMARY KEY,
                    query TEXT,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Get cached result
            import hashlib
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            
            cursor.execute("""
                SELECT result 
                FROM iland_classification_cache
                WHERE query_hash = %s
                AND created_at > NOW() - INTERVAL '%s seconds'
            """, (query_hash, self.config.cache_ttl))
            
            row = cursor.fetchone()
            if row:
                # Update access stats
                cursor.execute("""
                    UPDATE iland_classification_cache
                    SET accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE query_hash = %s
                """, (query_hash,))
                conn.commit()
                
                result = dict(row['result'])
                result['from_cache'] = True
                
                cursor.close()
                conn.close()
                return result
            
            cursor.close()
            conn.close()
            return None
            
        except Exception as e:
            print(f"Cache lookup error: {e}")
            return None
    
    def _cache_classification(self, query: str, result: Dict[str, Any]):
        """Cache classification result in PostgreSQL."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            import hashlib
            import json
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            
            cursor.execute("""
                INSERT INTO iland_classification_cache (query_hash, query, result)
                VALUES (%s, %s, %s)
                ON CONFLICT (query_hash) 
                DO UPDATE SET 
                    result = EXCLUDED.result,
                    created_at = CURRENT_TIMESTAMP,
                    access_count = 1
            """, (query_hash, query, json.dumps(result)))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def _log_classification(self, query: str, result: Dict[str, Any]):
        """Log classification for analytics."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Create log table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iland_classification_log (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    selected_index VARCHAR(255),
                    confidence FLOAT,
                    method VARCHAR(50),
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_ JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            cursor.execute("""
                INSERT INTO iland_classification_log 
                (query, selected_index, confidence, method, reasoning, metadata_)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                query,
                result['selected_index'],
                result.get('confidence', 0),
                result.get('method', 'unknown'),
                result.get('reasoning', ''),
                json.dumps({
                    'all_similarities': result.get('all_similarities', {}),
                    'from_cache': result.get('from_cache', False)
                })
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Classification logging error: {e}")
    
    def add_index(self, name: str, description: str):
        """Add a new index to both memory and PostgreSQL."""
        # Add to parent class
        super().add_index(name, description)
        
        # Update PostgreSQL
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO iland_indices (index_name, description)
                VALUES (%s, %s)
                ON CONFLICT (index_name) 
                DO UPDATE SET 
                    description = EXCLUDED.description,
                    updated_at = CURRENT_TIMESTAMP,
                    is_active = true
            """, (name, description))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error adding index to PostgreSQL: {e}")
    
    def remove_index(self, name: str):
        """Remove an index from both memory and PostgreSQL."""
        # Remove from parent class
        super().remove_index(name)
        
        # Mark as inactive in PostgreSQL (soft delete)
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE iland_indices 
                SET is_active = false, updated_at = CURRENT_TIMESTAMP
                WHERE index_name = %s
            """, (name,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error removing index from PostgreSQL: {e}")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics from PostgreSQL."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_classifications,
                    COUNT(DISTINCT query) as unique_queries,
                    AVG(confidence) as avg_confidence,
                    MAX(created_at) as last_classification
                FROM iland_classification_log
            """)
            overall_stats = dict(cursor.fetchone())
            
            # Get per-index stats
            cursor.execute("""
                SELECT 
                    selected_index,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM iland_classification_log
                GROUP BY selected_index
                ORDER BY count DESC
            """)
            index_stats = [dict(row) for row in cursor.fetchall()]
            
            # Get method distribution
            cursor.execute("""
                SELECT 
                    method,
                    COUNT(*) as count
                FROM iland_classification_log
                GROUP BY method
                ORDER BY count DESC
            """)
            method_stats = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return {
                "overall": overall_stats,
                "by_index": index_stats,
                "by_method": method_stats,
                "cache_enabled": self.config.enable_cache,
                "cache_ttl": self.config.cache_ttl
            }
            
        except Exception as e:
            print(f"Error getting classification stats: {e}")
            return {}
    
    async def classify_query_async(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Async version of classify_query for better performance."""
        # For now, wrap the sync method
        # TODO: Implement true async classification
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify_query, query, use_cache)
    
    def close(self):
        """Close any open connections."""
        # PostgreSQL connections are closed after each operation
        # This is a placeholder for future connection pooling
        pass


def create_postgres_classifier(
    config: Optional[PostgresRetrievalConfig] = None,
    api_key: Optional[str] = None,
    mode: Optional[str] = None
) -> PostgresIndexClassifier:
    """
    Create a PostgreSQL-based classifier with default configurations.
    
    Args:
        config: PostgreSQL configuration
        api_key: OpenAI API key
        mode: Classification mode (defaults to env var CLASSIFIER_MODE or "llm")
        
    Returns:
        PostgresIndexClassifier instance
    """
    # Get mode from environment variable or default to "llm"
    if mode is None:
        mode = os.getenv("CLASSIFIER_MODE", "llm")
    
    return PostgresIndexClassifier(
        config=config,
        available_indices=None,  # Will load from PostgreSQL or use defaults
        api_key=api_key,
        mode=mode
    )