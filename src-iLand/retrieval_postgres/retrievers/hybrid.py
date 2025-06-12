"""
PostgreSQL Hybrid Retriever for iLand Data

Implements hybrid search combining vector similarity and full-text search.
Uses pgVector for embeddings and PostgreSQL's full-text search capabilities.
"""

import os
import re
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever

from ..config import PostgresRetrievalConfig
from .vector import PostgresVectorRetriever


class PostgresHybridRetriever(BaseRetriever):
    """PostgreSQL-based hybrid retriever combining vector and full-text search."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 default_top_k: int = 5,
                 alpha: float = 0.7,
                 use_bge_embeddings: bool = True):
        """
        Initialize PostgreSQL hybrid retriever.
        
        Args:
            config: PostgreSQL configuration
            default_top_k: Default number of nodes to retrieve
            alpha: Weight for vector search (1-alpha for keyword search)
            use_bge_embeddings: Whether to use BGE-M3 embeddings
        """
        super().__init__()
        
        self.config = config or PostgresRetrievalConfig()
        self.default_top_k = default_top_k
        self.alpha = alpha
        
        # Initialize vector retriever component
        self.vector_retriever = PostgresVectorRetriever(
            config=config,
            default_top_k=default_top_k * 2,  # Get more candidates for reranking
            use_bge_embeddings=use_bge_embeddings
        )
        
        # Register pgVector
        self._register_vector()
        
        # Ensure full-text search is configured
        self._setup_fulltext_search()
    
    def _register_vector(self):
        """Register pgVector extension."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            register_vector(conn)
            conn.close()
        except Exception as e:
            print(f"Warning: Could not register pgVector: {e}")
    
    def _setup_fulltext_search(self):
        """Setup full-text search configuration for Thai and English."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Add text search column if it doesn't exist
            cursor.execute(f"""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = '{self.config.chunks_table}' 
                        AND column_name = 'search_vector'
                    ) THEN
                        ALTER TABLE {self.config.chunks_table} 
                        ADD COLUMN search_vector tsvector;
                    END IF;
                END $$;
            """)
            
            # Update search vectors
            cursor.execute(f"""
                UPDATE {self.config.chunks_table}
                SET search_vector = to_tsvector('simple', content)
                WHERE search_vector IS NULL
            """)
            
            # Create GIN index for full-text search
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_search 
                ON {self.config.chunks_table} 
                USING GIN (search_vector)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not setup full-text search: {e}")
    
    def _extract_thai_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Extract Thai and English keywords from query.
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (thai_keywords, english_keywords)
        """
        thai_keywords = []
        english_keywords = []
        
        # Important Thai land deed terms
        thai_land_terms = {
            "ที่ดิน": 1.5,    # land
            "โฉนด": 2.0,     # title deed
            "นส.3": 2.0,     # Nor Sor 3
            "นส3": 2.0,      # alternate
            "นส.4": 2.0,     # Nor Sor 4
            "นส4": 2.0,      # alternate
            "ส.ค.1": 2.0,    # Sor Kor 1
            "สค1": 2.0,      # alternate
            "อำเภอ": 1.2,    # district
            "ตำบล": 1.2,     # subdistrict
            "จังหวัด": 1.2,  # province
            "ไร่": 1.3,      # rai (area unit)
            "งาน": 1.3,      # ngan (area unit)
            "ตารางวา": 1.3, # square wah
            "เจ้าของ": 1.4, # owner
            "ผู้ถือกรรมสิทธิ์": 1.4  # rights holder
        }
        
        # Extract weighted Thai keywords
        query_lower = query.lower()
        for term, weight in thai_land_terms.items():
            if term in query_lower:
                thai_keywords.append((term, weight))
        
        # Extract general Thai words (simple tokenization)
        words = re.findall(r'[\u0e00-\u0e7f]+', query)
        for word in words:
            if len(word) > 1 and not any(word == kw[0] for kw in thai_keywords):
                thai_keywords.append((word.lower(), 1.0))
        
        # Extract English keywords
        english_words = re.findall(r'[a-zA-Z]+', query)
        for word in english_words:
            if len(word) > 2:
                english_keywords.append(word.lower())
        
        return thai_keywords, english_keywords
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve using hybrid search combining vector and full-text search.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of nodes with combined scores
        """
        query = query_bundle.query_str
        
        # Get vector search results
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        
        # Get keyword search results
        keyword_nodes = self._keyword_search(query)
        
        # Combine results
        combined_nodes = self._combine_results(vector_nodes, keyword_nodes)
        
        # Sort by combined score and return top_k
        combined_nodes.sort(key=lambda x: x.score, reverse=True)
        return combined_nodes[:self.default_top_k]
    
    def _keyword_search(self, query: str) -> List[NodeWithScore]:
        """
        Perform keyword-based full-text search.
        
        Args:
            query: Search query
            
        Returns:
            List of nodes with keyword scores
        """
        # Extract keywords
        thai_keywords, english_keywords = self._extract_thai_keywords(query)
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build search query
            search_terms = []
            
            # Add Thai keywords with weights
            for keyword, weight in thai_keywords:
                search_terms.append(f"{keyword}:*")
            
            # Add English keywords
            for keyword in english_keywords:
                search_terms.append(f"{keyword}:*")
            
            if not search_terms:
                return []
            
            # Combine search terms
            search_query = ' | '.join(search_terms)
            
            # Perform full-text search
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata,
                    c.document_id,
                    c.chunk_index,
                    ts_rank(c.search_vector, to_tsquery('simple', %s)) as rank,
                    d.title as document_title,
                    d.file_path as document_path
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE c.search_vector @@ to_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s
            """, (search_query, search_query, self.default_top_k * 2))
            
            nodes = []
            max_rank = 0.0
            
            # Get max rank for normalization
            rows = cursor.fetchall()
            if rows:
                max_rank = float(rows[0]['rank'])
            
            for row in rows:
                # Normalize rank to 0-1 score
                score = float(row['rank']) / max_rank if max_rank > 0 else 0.0
                
                # Apply Thai keyword weights
                content_lower = row['content'].lower()
                for keyword, weight in thai_keywords:
                    if keyword in content_lower:
                        # Boost score based on keyword weight and frequency
                        count = content_lower.count(keyword)
                        score *= (1 + (weight - 1) * min(count, 3) / 3)
                
                # Create text node
                node = TextNode(
                    text=row['content'],
                    id_=f"postgres_chunk_{row['id']}",
                    metadata={
                        **row['metadata'],
                        'chunk_id': row['id'],
                        'document_id': row['document_id'],
                        'chunk_index': row['chunk_index'],
                        'document_title': row['document_title'],
                        'document_path': row['document_path'],
                        'retrieval_strategy': 'keyword',
                        'keyword_rank': float(row['rank']),
                        'normalized_score': score,
                        'source': 'postgres'
                    }
                )
                
                # Create node with score
                node_with_score = NodeWithScore(
                    node=node,
                    score=min(score, 1.0)  # Cap at 1.0
                )
                
                nodes.append(node_with_score)
            
            cursor.close()
            conn.close()
            
            return nodes
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def _combine_results(self,
                        vector_nodes: List[NodeWithScore],
                        keyword_nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Combine vector and keyword search results.
        
        Args:
            vector_nodes: Nodes from vector search
            keyword_nodes: Nodes from keyword search
            
        Returns:
            Combined and deduplicated nodes
        """
        # Create a map of chunk_id to nodes and scores
        node_scores: Dict[int, Dict[str, Any]] = {}
        
        # Process vector results
        for node in vector_nodes:
            chunk_id = node.node.metadata.get('chunk_id')
            if chunk_id:
                node_scores[chunk_id] = {
                    'node': node.node,
                    'vector_score': node.score,
                    'keyword_score': 0.0
                }
        
        # Process keyword results
        for node in keyword_nodes:
            chunk_id = node.node.metadata.get('chunk_id')
            if chunk_id:
                if chunk_id in node_scores:
                    # Update keyword score for existing node
                    node_scores[chunk_id]['keyword_score'] = node.score
                else:
                    # Add new node from keyword search
                    node_scores[chunk_id] = {
                        'node': node.node,
                        'vector_score': 0.0,
                        'keyword_score': node.score
                    }
        
        # Combine scores and create final nodes
        combined_nodes = []
        for chunk_id, data in node_scores.items():
            # Calculate hybrid score
            vector_score = data['vector_score']
            keyword_score = data['keyword_score']
            
            # Use alpha weighting
            hybrid_score = self.alpha * vector_score + (1 - self.alpha) * keyword_score
            
            # Update metadata
            node = data['node']
            if hasattr(node, 'metadata'):
                node.metadata.update({
                    'retrieval_strategy': 'hybrid',
                    'vector_score': vector_score,
                    'keyword_score': keyword_score,
                    'hybrid_score': hybrid_score,
                    'alpha': self.alpha
                })
            
            # Create node with hybrid score
            combined_node = NodeWithScore(
                node=node,
                score=hybrid_score
            )
            
            combined_nodes.append(combined_node)
        
        return combined_nodes
    
    def update_alpha(self, alpha: float):
        """Update the alpha weighting parameter."""
        self.alpha = max(0.0, min(1.0, alpha))
    
    def retrieve_with_filters(self,
                            query: str,
                            metadata_filters: Optional[Dict[str, Any]] = None,
                            keyword_boost: Optional[Dict[str, float]] = None) -> List[NodeWithScore]:
        """
        Retrieve with additional filters and keyword boosting.
        
        Args:
            query: Search query
            metadata_filters: Metadata filters to apply
            keyword_boost: Additional keyword weights
            
        Returns:
            List of filtered and boosted results
        """
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Get base results
        nodes = self._retrieve(query_bundle)
        
        # Apply metadata filters if provided
        if metadata_filters:
            filtered_nodes = []
            for node in nodes:
                if hasattr(node.node, 'metadata'):
                    match = True
                    for key, value in metadata_filters.items():
                        if node.node.metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_nodes.append(node)
            nodes = filtered_nodes
        
        # Apply keyword boosting if provided
        if keyword_boost:
            for node in nodes:
                text_lower = node.node.text.lower()
                boost_factor = 1.0
                
                for keyword, boost in keyword_boost.items():
                    if keyword.lower() in text_lower:
                        boost_factor *= boost
                
                # Update score with boost
                node.score = min(node.score * boost_factor, 1.0)
        
        # Re-sort after filtering/boosting
        nodes.sort(key=lambda x: x.score, reverse=True)
        return nodes[:self.default_top_k]