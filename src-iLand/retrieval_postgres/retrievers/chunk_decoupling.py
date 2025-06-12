"""
PostgreSQL Chunk Decoupling Retriever for iLand Data

Implements chunk decoupling strategy that separates fine-grained chunk retrieval
from broader context synthesis. Retrieves relevant chunks and their surrounding context.
"""

import json
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever

from ..config import PostgresRetrievalConfig
from .vector import PostgresVectorRetriever


class PostgresChunkDecouplingRetriever(BaseRetriever):
    """PostgreSQL-based chunk decoupling retriever for precise chunk + context retrieval."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 default_top_k: int = 5,
                 context_window: int = 2,
                 use_bge_embeddings: bool = True,
                 chunk_weight: float = 0.7):
        """
        Initialize PostgreSQL chunk decoupling retriever.
        
        Args:
            config: PostgreSQL configuration
            default_top_k: Default number of nodes to retrieve
            context_window: Number of chunks before/after to include as context
            use_bge_embeddings: Whether to use BGE-M3 embeddings
            chunk_weight: Weight for chunk relevance vs context (0-1)
        """
        super().__init__()
        
        self.config = config or PostgresRetrievalConfig()
        self.default_top_k = default_top_k
        self.context_window = context_window
        self.chunk_weight = chunk_weight
        
        # Use vector retriever for similarity search
        self.vector_retriever = PostgresVectorRetriever(
            config=config,
            default_top_k=default_top_k * 2,  # Get more candidates
            use_bge_embeddings=use_bge_embeddings
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve using chunk decoupling strategy.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of nodes with chunk and context information
        """
        query = query_bundle.query_str
        
        # Step 1: Find relevant chunks
        chunk_nodes = self._find_relevant_chunks(query)
        
        if not chunk_nodes:
            return []
        
        # Step 2: Expand with context
        expanded_nodes = self._expand_with_context(chunk_nodes, query)
        
        # Step 3: Deduplicate and score
        final_nodes = self._deduplicate_and_score(expanded_nodes)
        
        # Step 4: Sort and return top_k
        final_nodes.sort(key=lambda x: x.score, reverse=True)
        return final_nodes[:self.default_top_k]
    
    def _find_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Find the most relevant fine-grained chunks.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Get query embedding
            query_embedding = self.vector_retriever._get_query_embedding(query)
            
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search for relevant chunks
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata,
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
                self.config.similarity_threshold,
                self.default_top_k * 2
            ))
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row['id'],
                    'content': row['content'],
                    'metadata': row['metadata'],
                    'document_id': row['document_id'],
                    'chunk_index': row['chunk_index'],
                    'similarity': float(row['similarity']),
                    'document_title': row['document_title'],
                    'document_path': row['document_path']
                })
            
            cursor.close()
            conn.close()
            
            return chunks
            
        except Exception as e:
            print(f"Error finding relevant chunks: {e}")
            return []
    
    def _expand_with_context(self, 
                           chunks: List[Dict[str, Any]], 
                           query: str) -> List[NodeWithScore]:
        """
        Expand chunks with surrounding context.
        
        Args:
            chunks: List of relevant chunks
            query: Original query
            
        Returns:
            List of nodes with chunk and context
        """
        expanded_nodes = []
        processed_chunks = set()
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            for chunk in chunks:
                chunk_id = chunk['id']
                
                if chunk_id in processed_chunks:
                    continue
                
                # Get surrounding chunks
                start_index = max(0, chunk['chunk_index'] - self.context_window)
                end_index = chunk['chunk_index'] + self.context_window
                
                cursor.execute(f"""
                    SELECT 
                        c.id,
                        c.content,
                        c.metadata,
                        c.chunk_index,
                        c.id = %s as is_target
                    FROM {self.config.chunks_table} c
                    WHERE c.document_id = %s
                    AND c.chunk_index >= %s
                    AND c.chunk_index <= %s
                    ORDER BY c.chunk_index
                """, (chunk_id, chunk['document_id'], start_index, end_index))
                
                context_chunks = cursor.fetchall()
                
                # Build combined content
                combined_content = []
                target_position = 0
                
                for i, ctx_chunk in enumerate(context_chunks):
                    if ctx_chunk['is_target']:
                        target_position = i
                        combined_content.append(f"[RELEVANT CHUNK] {ctx_chunk['content']}")
                    else:
                        combined_content.append(ctx_chunk['content'])
                    processed_chunks.add(ctx_chunk['id'])
                
                # Create node with expanded context
                full_content = "\n\n".join(combined_content)
                
                node = TextNode(
                    text=full_content,
                    id_=f"postgres_chunk_decoupled_{chunk_id}",
                    metadata={
                        **chunk['metadata'],
                        'chunk_id': chunk_id,
                        'document_id': chunk['document_id'],
                        'document_title': chunk['document_title'],
                        'document_path': chunk['document_path'],
                        'chunk_index': chunk['chunk_index'],
                        'target_chunk': chunk['content'],
                        'target_position': target_position,
                        'context_window_size': len(context_chunks),
                        'retrieval_strategy': 'chunk_decoupling',
                        'similarity_score': chunk['similarity'],
                        'source': 'postgres'
                    }
                )
                
                # Calculate score based on chunk relevance and context quality
                chunk_score = chunk['similarity']
                context_score = self._calculate_context_quality(
                    context_chunks, 
                    target_position,
                    query
                )
                
                final_score = (self.chunk_weight * chunk_score + 
                             (1 - self.chunk_weight) * context_score)
                
                node_with_score = NodeWithScore(
                    node=node,
                    score=final_score
                )
                
                expanded_nodes.append(node_with_score)
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error expanding with context: {e}")
        
        return expanded_nodes
    
    def _calculate_context_quality(self,
                                 context_chunks: List[Dict[str, Any]],
                                 target_position: int,
                                 query: str) -> float:
        """
        Calculate quality score for the context.
        
        Args:
            context_chunks: List of context chunks
            target_position: Position of target chunk in context
            query: Original query
            
        Returns:
            Context quality score (0-1)
        """
        if not context_chunks:
            return 0.0
        
        # Base score from context completeness
        completeness_score = len(context_chunks) / (2 * self.context_window + 1)
        
        # Position score (prefer centered target)
        center = len(context_chunks) // 2
        position_score = 1.0 - abs(target_position - center) / len(context_chunks)
        
        # Content coherence (simplified - check for Thai land deed terms)
        thai_terms = ["โฉนด", "นส.3", "นส.4", "ส.ค.1", "ที่ดิน", "อำเภอ", "จังหวัด"]
        
        coherence_score = 0.0
        for chunk in context_chunks:
            content = chunk['content'].lower()
            term_count = sum(1 for term in thai_terms if term in content)
            coherence_score += term_count / len(thai_terms)
        
        coherence_score = coherence_score / len(context_chunks)
        
        # Combine scores
        final_score = (0.4 * completeness_score + 
                      0.3 * position_score + 
                      0.3 * coherence_score)
        
        return min(final_score, 1.0)
    
    def _deduplicate_and_score(self, 
                              nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Deduplicate overlapping contexts and adjust scores.
        
        Args:
            nodes: List of nodes with potential overlaps
            
        Returns:
            Deduplicated list of nodes
        """
        # Group by document
        doc_groups = {}
        for node in nodes:
            doc_id = node.node.metadata.get('document_id')
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(node)
        
        final_nodes = []
        
        for doc_id, doc_nodes in doc_groups.items():
            # Sort by chunk index
            doc_nodes.sort(key=lambda x: x.node.metadata.get('chunk_index', 0))
            
            # Track processed chunk ranges
            processed_ranges = []
            
            for node in doc_nodes:
                chunk_index = node.node.metadata.get('chunk_index', 0)
                window_size = node.node.metadata.get('context_window_size', 1)
                
                start = chunk_index - self.context_window
                end = chunk_index + self.context_window
                
                # Check overlap with processed ranges
                overlap = False
                for proc_start, proc_end in processed_ranges:
                    if not (end < proc_start or start > proc_end):
                        overlap = True
                        break
                
                if not overlap:
                    final_nodes.append(node)
                    processed_ranges.append((start, end))
                else:
                    # Reduce score for overlapping content
                    node.score *= 0.7
                    final_nodes.append(node)
        
        return final_nodes
    
    def retrieve_chunk_only(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve only the relevant chunks without context expansion.
        
        Args:
            query: Search query
            
        Returns:
            List of chunk nodes without expanded context
        """
        chunks = self._find_relevant_chunks(query)
        
        nodes = []
        for chunk in chunks[:self.default_top_k]:
            node = TextNode(
                text=chunk['content'],
                id_=f"postgres_chunk_{chunk['id']}",
                metadata={
                    **chunk['metadata'],
                    'chunk_id': chunk['id'],
                    'document_id': chunk['document_id'],
                    'document_title': chunk['document_title'],
                    'document_path': chunk['document_path'],
                    'chunk_index': chunk['chunk_index'],
                    'retrieval_strategy': 'chunk_only',
                    'similarity_score': chunk['similarity'],
                    'source': 'postgres'
                }
            )
            
            node_with_score = NodeWithScore(
                node=node,
                score=chunk['similarity']
            )
            
            nodes.append(node_with_score)
        
        return nodes
    
    def update_context_window(self, window_size: int):
        """Update the context window size."""
        self.context_window = max(0, window_size)
    
    def update_chunk_weight(self, weight: float):
        """Update the chunk vs context weight."""
        self.chunk_weight = max(0.0, min(1.0, weight))