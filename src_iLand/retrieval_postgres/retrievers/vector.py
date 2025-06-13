"""
PostgreSQL Vector Retriever for iLand Data

Implements vector similarity search using pgVector with BGE-M3 embeddings.
Maintains parity with local vector retriever while leveraging PostgreSQL capabilities.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding

from ..config import PostgresRetrievalConfig
from src_iLand.docs_embedding_postgres.bge_embedding_processor import BGEEmbeddingProcessor


class PostgresVectorRetriever(BaseRetriever):
    """PostgreSQL-based vector retriever using pgVector for similarity search."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 default_top_k: int = 5,
                 similarity_threshold: Optional[float] = None,
                 use_bge_embeddings: bool = True):
        """
        Initialize PostgreSQL vector retriever.
        
        Args:
            config: PostgreSQL configuration
            default_top_k: Default number of nodes to retrieve
            similarity_threshold: Minimum similarity score threshold
            use_bge_embeddings: Whether to use BGE-M3 embeddings (True) or OpenAI (False)
        """
        super().__init__()
        
        self.config = config or PostgresRetrievalConfig()
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold or self.config.similarity_threshold
        self.use_bge_embeddings = use_bge_embeddings
        
        # Initialize embedding processor
        if self.use_bge_embeddings:
            self.embedding_processor = BGEEmbeddingProcessor({
                "provider": "bge",
                "model_name": self.config.embedding_model,
                "cache_folder": "./cache/bge_models"
            })
        else:
            self.embedding_processor = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            )
        
        # Register pgVector extension
        self._register_vector()
        
        # Verify table structure
        self._verify_table_structure()
    
    def _register_vector(self):
        """Register pgVector extension with psycopg2."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            register_vector(conn)
            conn.close()
        except Exception as e:
            print(f"Warning: Could not register pgVector: {e}")
    
    def _verify_table_structure(self):
        """Verify that the required tables and columns exist."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Check if chunks table exists with required columns
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s
                AND column_name IN ('id', 'content', 'embedding', 'metadata_', 'document_id')
            """, (self.config.chunks_table,))
            
            columns = {row[0]: row[1] for row in cursor.fetchall()}
            
            if len(columns) < 5:
                print(f"Warning: Table {self.config.chunks_table} missing required columns")
            
            # Create index for vector similarity search if it doesn't exist
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_embedding 
                ON {self.config.chunks_table} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not verify table structure: {e}")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text."""
        if self.use_bge_embeddings:
            # Use BGE-M3 embeddings
            embeddings = self.embedding_processor.get_text_embeddings([query])
            return embeddings[0]
        else:
            # Use OpenAI embeddings
            return self.embedding_processor.get_text_embedding(query)
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using vector similarity search in PostgreSQL.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of nodes with similarity scores
        """
        query = query_bundle.query_str
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Perform vector similarity search
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata_,
                    c.document_id,
                    c.chunk_index,
                    c.embedding <=> %s::vector as distance,
                    1 - (c.embedding <=> %s::vector) as similarity,
                    d.title as document_title,
                    d.file_path as document_path
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE 1 - (c.embedding <=> %s::vector) >= %s
                ORDER BY distance
                LIMIT %s
            """, (
                query_embedding,
                query_embedding,
                query_embedding,
                self.similarity_threshold,
                self.default_top_k
            ))
            
            nodes = []
            for row in cursor.fetchall():
                # Create text node
                node = TextNode(
                    text=row['content'],
                    id_=f"postgres_chunk_{row['id']}",
                    metadata={
                        **row['metadata_'],
                        'chunk_id': row['id'],
                        'document_id': row['document_id'],
                        'chunk_index': row['chunk_index'],
                        'document_title': row['document_title'],
                        'document_path': row['document_path'],
                        'retrieval_strategy': 'vector',
                        'similarity_score': float(row['similarity']),
                        'distance': float(row['distance']),
                        'source': 'postgres'
                    }
                )
                
                # Create node with score
                node_with_score = NodeWithScore(
                    node=node,
                    score=float(row['similarity'])
                )
                
                nodes.append(node_with_score)
            
            cursor.close()
            conn.close()
            
            return nodes
            
        except Exception as e:
            print(f"Error in vector retrieval: {e}")
            return []
    
    def retrieve_with_metadata_filter(self,
                                    query: str,
                                    metadata_filters: Dict[str, Any],
                                    top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve with additional metadata filtering.
        
        Args:
            query: The search query
            metadata_filters: Dictionary of metadata filters
            top_k: Number of results to retrieve
            
        Returns:
            List of nodes with scores
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        k = top_k or self.default_top_k
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build metadata filter conditions
            filter_conditions = []
            filter_params = [query_embedding, query_embedding, query_embedding, self.similarity_threshold]
            
            # Add metadata filters if provided
            if metadata_filters:
                for key, value in metadata_filters.items():
                    filter_conditions.append("c.metadata_->>%s = %s")
                    filter_params.extend([key, value])
            
            where_clause = " AND ".join([
                f"1 - (c.embedding <=> %s::vector) >= %s"
            ] + filter_conditions)
            
            filter_params.append(k)
            
            # Execute query with filters
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata_,
                    c.document_id,
                    c.chunk_index,
                    c.embedding <=> %s::vector as distance,
                    1 - (c.embedding <=> %s::vector) as similarity,
                    d.title as document_title,
                    d.file_path as document_path
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE {where_clause}
                ORDER BY distance
                LIMIT %s
            """, filter_params)
            
            nodes = []
            for row in cursor.fetchall():
                # Create text node
                node = TextNode(
                    text=row['content'],
                    id_=f"postgres_chunk_{row['id']}",
                    metadata={
                        **row['metadata_'],
                        'chunk_id': row['id'],
                        'document_id': row['document_id'],
                        'chunk_index': row['chunk_index'],
                        'document_title': row['document_title'],
                        'document_path': row['document_path'],
                        'retrieval_strategy': 'vector_filtered',
                        'similarity_score': float(row['similarity']),
                        'distance': float(row['distance']),
                        'metadata_filters': metadata_filters,
                        'source': 'postgres'
                    }
                )
                
                # Create node with score
                node_with_score = NodeWithScore(
                    node=node,
                    score=float(row['similarity'])
                )
                
                nodes.append(node_with_score)
            
            cursor.close()
            conn.close()
            
            return nodes
            
        except Exception as e:
            print(f"Error in filtered vector retrieval: {e}")
            return []
    
    def get_similar_chunks(self,
                          chunk_id: int,
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to retrieve
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get embedding of reference chunk
            cursor.execute(f"""
                SELECT embedding 
                FROM {self.config.chunks_table}
                WHERE id = %s
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if not row:
                return []
            
            reference_embedding = row['embedding']
            
            # Find similar chunks
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata_,
                    c.document_id,
                    c.embedding <=> %s::vector as distance,
                    1 - (c.embedding <=> %s::vector) as similarity,
                    d.title as document_title
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE c.id != %s
                ORDER BY distance
                LIMIT %s
            """, (reference_embedding, reference_embedding, chunk_id, top_k))
            
            similar_chunks = []
            for row in cursor.fetchall():
                similar_chunks.append({
                    'id': row['id'],
                    'content': row['content'],
                    'metadata_': row['metadata_'],
                    'document_id': row['document_id'],
                    'document_title': row['document_title'],
                    'similarity': float(row['similarity']),
                    'distance': float(row['distance'])
                })
            
            cursor.close()
            conn.close()
            
            return similar_chunks
            
        except Exception as e:
            print(f"Error finding similar chunks: {e}")
            return []
    
    def update_similarity_threshold(self, threshold: float):
        """Update the similarity threshold for retrieval."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
    
    def update_top_k(self, top_k: int):
        """Update the default top_k value."""
        self.default_top_k = max(1, top_k)