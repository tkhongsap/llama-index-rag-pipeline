"""
PostgreSQL Recursive Retriever for iLand Data

Implements hierarchical recursive retrieval using parent-child relationships in PostgreSQL.
Supports document -> section -> chunk hierarchy for Thai land deed data.
"""

import json
from typing import List, Optional, Dict, Any, Set
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever

from ..config import PostgresRetrievalConfig
from .vector import PostgresVectorRetriever


class PostgresRecursiveRetriever(BaseRetriever):
    """PostgreSQL-based recursive retriever with hierarchical traversal."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 default_top_k: int = 5,
                 max_depth: int = 3,
                 use_bge_embeddings: bool = True):
        """
        Initialize PostgreSQL recursive retriever.
        
        Args:
            config: PostgreSQL configuration
            default_top_k: Default number of nodes to retrieve at each level
            max_depth: Maximum recursion depth
            use_bge_embeddings: Whether to use BGE-M3 embeddings
        """
        super().__init__()
        
        self.config = config or PostgresRetrievalConfig()
        self.default_top_k = default_top_k
        self.max_depth = max_depth
        
        # Use vector retriever for similarity search at each level
        self.vector_retriever = PostgresVectorRetriever(
            config=config,
            default_top_k=default_top_k,
            use_bge_embeddings=use_bge_embeddings
        )
        
        # Ensure hierarchical relationships are set up
        self._setup_hierarchical_structure()
    
    def _setup_hierarchical_structure(self):
        """Ensure tables have proper hierarchical relationships."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Add parent_chunk_id if not exists
            cursor.execute(f"""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = '{self.config.chunks_table}' 
                        AND column_name = 'parent_chunk_id'
                    ) THEN
                        ALTER TABLE {self.config.chunks_table} 
                        ADD COLUMN parent_chunk_id INTEGER REFERENCES {self.config.chunks_table}(id);
                    END IF;
                END $$;
            """)
            
            # Add chunk_level if not exists (document=0, section=1, chunk=2)
            cursor.execute(f"""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = '{self.config.chunks_table}' 
                        AND column_name = 'chunk_level'
                    ) THEN
                        ALTER TABLE {self.config.chunks_table} 
                        ADD COLUMN chunk_level INTEGER DEFAULT 2;
                    END IF;
                END $$;
            """)
            
            # Create index for hierarchical queries
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_hierarchy 
                ON {self.config.chunks_table}(parent_chunk_id, chunk_level)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not setup hierarchical structure: {e}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using recursive hierarchical search.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of nodes with scores from multiple hierarchy levels
        """
        query = query_bundle.query_str
        
        # Start with vector search to find relevant leaf nodes
        initial_nodes = self.vector_retriever.retrieve(query_bundle)
        
        if not initial_nodes:
            return []
        
        # Collect all nodes including ancestors
        all_nodes = {}  # chunk_id -> NodeWithScore
        visited_ids = set()
        
        # Process initial nodes and their ancestors
        for node in initial_nodes:
            chunk_id = node.node.metadata.get('chunk_id')
            if chunk_id:
                self._collect_ancestors(chunk_id, query, all_nodes, visited_ids, 0)
        
        # Also search for relevant parent nodes directly
        parent_nodes = self._search_parent_nodes(query)
        for node in parent_nodes:
            chunk_id = node.node.metadata.get('chunk_id')
            if chunk_id and chunk_id not in visited_ids:
                all_nodes[chunk_id] = node
                # Collect children of relevant parents
                self._collect_descendants(chunk_id, query, all_nodes, visited_ids, 0)
        
        # Convert to list and sort by combined score
        result_nodes = list(all_nodes.values())
        result_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return result_nodes[:self.default_top_k]
    
    def _collect_ancestors(self,
                          chunk_id: int,
                          query: str,
                          all_nodes: Dict[int, NodeWithScore],
                          visited_ids: Set[int],
                          depth: int):
        """
        Recursively collect ancestor nodes.
        
        Args:
            chunk_id: Current chunk ID
            query: Original query for scoring
            all_nodes: Dictionary to store collected nodes
            visited_ids: Set of already visited chunk IDs
            depth: Current recursion depth
        """
        if depth >= self.max_depth or chunk_id in visited_ids:
            return
        
        visited_ids.add(chunk_id)
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get current chunk and its parent
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata_,
                    c.document_id,
                    c.chunk_index,
                    c.parent_chunk_id,
                    c.chunk_level,
                    d.title as document_title,
                    d.file_path as document_path
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE c.id = %s
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if row:
                # Create node for current chunk
                node = TextNode(
                    text=row['content'],
                    id_=f"postgres_chunk_{row['id']}",
                    metadata={
                        **row['metadata_'],
                        'chunk_id': row['id'],
                        'document_id': row['document_id'],
                        'chunk_index': row['chunk_index'],
                        'chunk_level': row['chunk_level'],
                        'parent_chunk_id': row['parent_chunk_id'],
                        'document_title': row['document_title'],
                        'document_path': row['document_path'],
                        'retrieval_strategy': 'recursive',
                        'hierarchy_depth': depth,
                        'source': 'postgres'
                    }
                )
                
                # Calculate relevance score based on depth
                # Deeper nodes get lower scores
                depth_penalty = 0.9 ** depth
                base_score = self._calculate_relevance_score(row['content'], query)
                score = base_score * depth_penalty
                
                node_with_score = NodeWithScore(node=node, score=score)
                all_nodes[chunk_id] = node_with_score
                
                # Recursively collect parent
                if row['parent_chunk_id']:
                    self._collect_ancestors(
                        row['parent_chunk_id'],
                        query,
                        all_nodes,
                        visited_ids,
                        depth + 1
                    )
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error collecting ancestors: {e}")
    
    def _collect_descendants(self,
                           chunk_id: int,
                           query: str,
                           all_nodes: Dict[int, NodeWithScore],
                           visited_ids: Set[int],
                           depth: int):
        """
        Recursively collect descendant nodes.
        
        Args:
            chunk_id: Current chunk ID
            query: Original query for scoring
            all_nodes: Dictionary to store collected nodes
            visited_ids: Set of already visited chunk IDs
            depth: Current recursion depth
        """
        if depth >= self.max_depth or chunk_id in visited_ids:
            return
        
        visited_ids.add(chunk_id)
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get all children of current chunk
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata_,
                    c.document_id,
                    c.chunk_index,
                    c.chunk_level,
                    d.title as document_title,
                    d.file_path as document_path
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE c.parent_chunk_id = %s
                ORDER BY c.chunk_index
            """, (chunk_id,))
            
            for row in cursor.fetchall():
                child_id = row['id']
                if child_id not in visited_ids:
                    # Create node for child chunk
                    node = TextNode(
                        text=row['content'],
                        id_=f"postgres_chunk_{row['id']}",
                        metadata={
                            **row['metadata_'],
                            'chunk_id': row['id'],
                            'document_id': row['document_id'],
                            'chunk_index': row['chunk_index'],
                            'chunk_level': row['chunk_level'],
                            'parent_chunk_id': chunk_id,
                            'document_title': row['document_title'],
                            'document_path': row['document_path'],
                            'retrieval_strategy': 'recursive_child',
                            'hierarchy_depth': depth,
                            'source': 'postgres'
                        }
                    )
                    
                    # Calculate relevance score
                    depth_penalty = 0.9 ** depth
                    base_score = self._calculate_relevance_score(row['content'], query)
                    score = base_score * depth_penalty * 0.8  # Children get slightly lower score
                    
                    node_with_score = NodeWithScore(node=node, score=score)
                    all_nodes[child_id] = node_with_score
                    
                    # Recursively collect children
                    self._collect_descendants(
                        child_id,
                        query,
                        all_nodes,
                        visited_ids,
                        depth + 1
                    )
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error collecting descendants: {e}")
    
    def _search_parent_nodes(self, query: str) -> List[NodeWithScore]:
        """
        Search for relevant parent nodes (sections/summaries).
        
        Args:
            query: Search query
            
        Returns:
            List of parent nodes that match the query
        """
        try:
            # Get query embedding
            query_embedding = self.vector_retriever._get_query_embedding(query)
            
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search parent nodes (chunk_level < 2)
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata_,
                    c.document_id,
                    c.chunk_index,
                    c.chunk_level,
                    c.embedding <=> %s::vector as distance,
                    1 - (c.embedding <=> %s::vector) as similarity,
                    d.title as document_title,
                    d.file_path as document_path
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE c.chunk_level < 2
                AND 1 - (c.embedding <=> %s::vector) >= %s
                ORDER BY distance
                LIMIT %s
            """, (
                query_embedding,
                query_embedding,
                query_embedding,
                self.config.similarity_threshold * 0.8,  # Lower threshold for parents
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
                        'chunk_level': row['chunk_level'],
                        'document_title': row['document_title'],
                        'document_path': row['document_path'],
                        'retrieval_strategy': 'recursive_parent',
                        'similarity_score': float(row['similarity']),
                        'source': 'postgres'
                    }
                )
                
                node_with_score = NodeWithScore(
                    node=node,
                    score=float(row['similarity'])
                )
                
                nodes.append(node_with_score)
            
            cursor.close()
            conn.close()
            
            return nodes
            
        except Exception as e:
            print(f"Error searching parent nodes: {e}")
            return []
    
    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """
        Calculate simple relevance score between content and query.
        
        Args:
            content: Chunk content
            query: Search query
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple keyword matching for now
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Count matching words
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        
        if not query_words:
            return 0.5
        
        # Calculate score
        score = matches / len(query_words)
        
        # Boost for Thai land deed terms
        thai_terms = ["โฉนด", "นส.3", "นส.4", "ส.ค.1", "ที่ดิน"]
        for term in thai_terms:
            if term in content and term in query:
                score *= 1.2
        
        return min(score, 1.0)
    
    def get_document_hierarchy(self, document_id: int) -> Dict[str, Any]:
        """
        Get the complete hierarchy for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Hierarchical structure of the document
        """
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get all chunks for the document
            cursor.execute(f"""
                SELECT 
                    id,
                    chunk_index,
                    chunk_level,
                    parent_chunk_id,
                    content,
                    metadata_
                FROM {self.config.chunks_table}
                WHERE document_id = %s
                ORDER BY chunk_level, chunk_index
            """, (document_id,))
            
            chunks = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Build hierarchy
            hierarchy = {}
            chunk_map = {chunk['id']: chunk for chunk in chunks}
            
            # Find root chunks (no parent)
            for chunk in chunks:
                if chunk['parent_chunk_id'] is None:
                    hierarchy[chunk['id']] = {
                        'id': chunk['id'],
                        'level': chunk['chunk_level'],
                        'index': chunk['chunk_index'],
                        'content': chunk['content'][:100] + '...',
                        'metadata_': chunk['metadata_'],
                        'children': []
                    }
            
            # Build tree
            for chunk in chunks:
                if chunk['parent_chunk_id'] is not None:
                    parent_id = chunk['parent_chunk_id']
                    if parent_id in hierarchy:
                        hierarchy[parent_id]['children'].append({
                            'id': chunk['id'],
                            'level': chunk['chunk_level'],
                            'index': chunk['chunk_index'],
                            'content': chunk['content'][:100] + '...',
                            'metadata_': chunk['metadata_']
                        })
            
            return hierarchy
            
        except Exception as e:
            print(f"Error getting document hierarchy: {e}")
            return {}