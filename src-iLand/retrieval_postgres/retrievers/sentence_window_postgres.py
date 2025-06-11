"""
Sentence Window PostgreSQL Retriever

This retriever expands retrieved chunks by including surrounding context
(previous and next chunks) to provide better context for the LLM.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from ..base_retriever import BasePostgresRetriever
from ..config import PostgresConfig
from ..utils.db_connection import PostgresConnectionManager
from ..utils.vector_ops import generate_embedding, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval with score and metadata."""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


class SentenceWindowPostgresRetriever(BasePostgresRetriever):
    """
    Sentence Window retrieval strategy for PostgreSQL.
    
    This retriever:
    1. Performs initial vector similarity search
    2. For each retrieved chunk, fetches surrounding chunks (window)
    3. Combines chunks to provide expanded context
    4. Maintains original relevance scores
    """
    
    def __init__(self, config: PostgresConfig, window_size: int = 1):
        """
        Initialize sentence window retriever.
        
        Args:
            config: PostgreSQL configuration
            window_size: Number of chunks to include before/after each result
        """
        super().__init__(config, "sentence_window_postgres")
        self.window_size = window_size
        
        logger.info(f"Initialized SentenceWindowPostgresRetriever with window_size={window_size}")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Retrieve documents with sentence window expansion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters
                - window_size: Override default window size
                - min_score: Minimum similarity score threshold
        
        Returns:
            List of RetrievalResult objects with expanded context
        """
        window_size = kwargs.get('window_size', self.window_size)
        min_score = kwargs.get('min_score', self.config.similarity_threshold)
        
        logger.info(f"Retrieving with sentence window: query='{query[:50]}...', top_k={top_k}, window_size={window_size}")
        
        try:
            # Generate query embedding
            query_embedding = generate_embedding(query, self.config.embedding_model)
            
            with PostgresConnectionManager(self.config) as conn_manager:
                conn = conn_manager.get_connection()
                cursor = conn.cursor()
                
                # Step 1: Get initial similar chunks
                initial_results = self._get_similar_chunks(cursor, query_embedding, top_k * 2, min_score)
                
                if not initial_results:
                    logger.warning("No similar chunks found")
                    return []
                
                # Step 2: Expand each result with window context
                expanded_results = []
                processed_chunks = set()  # Avoid duplicates
                
                for chunk_data in initial_results[:top_k]:
                    chunk_id = chunk_data['chunk_id']
                    
                    if chunk_id in processed_chunks:
                        continue
                    
                    # Get window context for this chunk
                    window_context = self._get_window_context(cursor, chunk_data, window_size)
                    
                    if window_context:
                        expanded_results.append(window_context)
                        processed_chunks.add(chunk_id)
                
                conn_manager.return_connection(conn)
                
                logger.info(f"Retrieved {len(expanded_results)} expanded results")
                return expanded_results[:top_k]
                
        except Exception as e:
            logger.error(f"Error in sentence window retrieval: {e}")
            raise
    
    def _get_similar_chunks(self, cursor, query_embedding: List[float], limit: int, min_score: float) -> List[Dict[str, Any]]:
        """Get initial similar chunks from database."""
        
        # Convert embedding to PostgreSQL array format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        query = f"""
            SELECT 
                chunk_id,
                content,
                embedding,
                source_file,
                chunk_index,
                metadata,
                1 - (embedding <=> %s::vector) as similarity_score
            FROM {self.config.chunks_table}
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        cursor.execute(query, (embedding_str, embedding_str, min_score, embedding_str, limit))
        results = cursor.fetchall()
        
        columns = ['chunk_id', 'content', 'embedding', 'source_file', 'chunk_index', 'metadata', 'similarity_score']
        return [dict(zip(columns, row)) for row in results]
    
    def _get_window_context(self, cursor, chunk_data: Dict[str, Any], window_size: int) -> Optional[RetrievalResult]:
        """
        Get window context around a specific chunk.
        
        Args:
            cursor: Database cursor
            chunk_data: Original chunk data
            window_size: Number of chunks before/after to include
        
        Returns:
            RetrievalResult with expanded context
        """
        try:
            source_file = chunk_data['source_file']
            chunk_index = chunk_data['chunk_index']
            original_score = chunk_data['similarity_score']
            
            # Calculate window range
            start_index = max(0, chunk_index - window_size)
            end_index = chunk_index + window_size
            
            # Get chunks in window range from same source file
            window_query = f"""
                SELECT 
                    chunk_id,
                    content,
                    chunk_index,
                    metadata
                FROM {self.config.chunks_table}
                WHERE source_file = %s 
                AND chunk_index BETWEEN %s AND %s
                ORDER BY chunk_index
            """
            
            cursor.execute(window_query, (source_file, start_index, end_index))
            window_chunks = cursor.fetchall()
            
            if not window_chunks:
                # Fallback to original chunk
                return RetrievalResult(
                    text=chunk_data['content'],
                    score=original_score,
                    metadata=chunk_data.get('metadata', {}),
                    chunk_id=chunk_data['chunk_id']
                )
            
            # Combine chunks into expanded context
            combined_text = []
            combined_metadata = chunk_data.get('metadata', {}).copy()
            
            for chunk in window_chunks:
                chunk_content = chunk[1]  # content
                chunk_meta = chunk[3] if chunk[3] else {}  # metadata
                
                # Mark the original chunk
                if chunk[2] == chunk_index:  # chunk_index
                    combined_text.append(f"[MAIN] {chunk_content}")
                else:
                    combined_text.append(chunk_content)
                
                # Merge metadata
                if isinstance(chunk_meta, dict):
                    combined_metadata.update(chunk_meta)
            
            # Create expanded context
            expanded_text = "\n\n".join(combined_text)
            
            # Add window information to metadata
            combined_metadata.update({
                'retrieval_strategy': 'sentence_window',
                'window_size': window_size,
                'original_chunk_index': chunk_index,
                'window_start': start_index,
                'window_end': end_index,
                'chunks_in_window': len(window_chunks),
                'source_file': source_file
            })
            
            return RetrievalResult(
                text=expanded_text,
                score=original_score,
                metadata=combined_metadata,
                chunk_id=chunk_data['chunk_id']
            )
            
        except Exception as e:
            logger.error(f"Error getting window context: {e}")
            # Return original chunk as fallback
            return RetrievalResult(
                text=chunk_data['content'],
                score=chunk_data['similarity_score'],
                metadata=chunk_data.get('metadata', {}),
                chunk_id=chunk_data['chunk_id']
            )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this retrieval strategy."""
        return {
            "name": "sentence_window",
            "description": "Expands retrieved chunks with surrounding context",
            "parameters": {
                "window_size": self.window_size,
                "embedding_model": self.config.embedding_model,
                "similarity_threshold": self.config.similarity_threshold
            },
            "features": [
                "Vector similarity search",
                "Context expansion with surrounding chunks",
                "Maintains original relevance scores",
                "Supports Thai language content",
                "Configurable window size"
            ]
        } 