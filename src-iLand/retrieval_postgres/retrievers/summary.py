"""
PostgreSQL Summary Retriever for iLand Data

Implements document summary-first retrieval for Thai land deed data.
Uses PostgreSQL for vector storage and retrieval with BGE-M3 embeddings.
"""

import time
import logging
from typing import List, Optional, Dict, Any, Set

import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever

from ..config import PostgresRetrievalConfig
from ...docs_embedding_postgres.bge_embedding_processor import BGEEmbeddingProcessor

logger = logging.getLogger(__name__)


class PostgresSummaryRetriever(BaseRetriever):
    """PostgreSQL-based summary-first retriever for Thai land deed data."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 default_top_k: int = 5,
                 similarity_threshold: Optional[float] = None,
                 use_bge_embeddings: bool = True,
                 retrieve_full_documents: bool = True,
                 chunks_per_document: int = 3):
        """
        Initialize PostgreSQL summary retriever.
        
        Args:
            config: PostgreSQL configuration
            default_top_k: Default number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold
            use_bge_embeddings: Whether to use BGE-M3 embeddings (True) or OpenAI (False)
            retrieve_full_documents: Whether to retrieve all chunks from matched documents
            chunks_per_document: Number of top chunks to retrieve per document if not retrieving all
        """
        super().__init__()
        
        self.config = config or PostgresRetrievalConfig()
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold or self.config.similarity_threshold
        self.use_bge_embeddings = use_bge_embeddings
        self.retrieve_full_documents = retrieve_full_documents
        self.chunks_per_document = chunks_per_document
        
        # Initialize embedding processor
        if self.use_bge_embeddings:
            self.embedding_processor = BGEEmbeddingProcessor()
        else:
            raise NotImplementedError("Only BGE embeddings are supported for PostgreSQL retrieval")
        
        # Table names
        self.summaries_table = "iland_summaries"
        self.chunks_table = self.config.chunks_table
        self.documents_table = self.config.documents_table
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text."""
        embeddings = self.embedding_processor.get_text_embeddings([query])
        return embeddings[0]
    
    def _retrieve_summaries(self, 
                           query_embedding: List[float], 
                           top_k: int,
                           conn) -> List[Dict[str, Any]]:
        """
        Retrieve document summaries based on query similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of summaries to retrieve
            conn: Database connection
            
        Returns:
            List of summary results
        """
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Query summaries table
            cursor.execute(f"""
                SELECT 
                    s.id,
                    s.document_id,
                    s.summary_text,
                    s.summary_embedding <=> %s::vector as distance,
                    1 - (s.summary_embedding <=> %s::vector) as similarity,
                    s.metadata,
                    d.title as document_title,
                    d.file_path as document_path,
                    d.metadata as document_metadata
                FROM {self.summaries_table} s
                LEFT JOIN {self.documents_table} d ON s.document_id = d.id
                WHERE 1 - (s.summary_embedding <=> %s::vector) >= %s
                ORDER BY distance
                LIMIT %s
            """, (
                query_embedding,
                query_embedding,
                query_embedding,
                self.similarity_threshold,
                top_k
            ))
            
            results = cursor.fetchall()
            cursor.close()
            
            logger.debug(f"Retrieved {len(results)} summaries")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving summaries: {e}")
            cursor.close()
            return []
    
    def _retrieve_chunks_for_documents(self,
                                     document_ids: List[str],
                                     query_embedding: List[float],
                                     conn) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks for specific documents.
        
        Args:
            document_ids: List of document IDs
            query_embedding: Query vector for ranking chunks
            conn: Database connection
            
        Returns:
            Dictionary mapping document_id to list of chunks
        """
        if not document_ids:
            return {}
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        chunks_by_document = {}
        
        try:
            for doc_id in document_ids:
                if self.retrieve_full_documents:
                    # Retrieve all chunks for the document
                    cursor.execute(f"""
                        SELECT 
                            id,
                            content,
                            metadata,
                            document_id,
                            chunk_index,
                            embedding <=> %s::vector as distance,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM {self.chunks_table}
                        WHERE document_id = %s
                        ORDER BY chunk_index
                    """, (query_embedding, query_embedding, doc_id))
                else:
                    # Retrieve top-k chunks for the document
                    cursor.execute(f"""
                        SELECT 
                            id,
                            content,
                            metadata,
                            document_id,
                            chunk_index,
                            embedding <=> %s::vector as distance,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM {self.chunks_table}
                        WHERE document_id = %s
                        ORDER BY distance
                        LIMIT %s
                    """, (query_embedding, query_embedding, doc_id, self.chunks_per_document))
                
                chunks = cursor.fetchall()
                if chunks:
                    chunks_by_document[doc_id] = chunks
                    logger.debug(f"Retrieved {len(chunks)} chunks for document {doc_id}")
            
            cursor.close()
            return chunks_by_document
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            cursor.close()
            return {}
    
    def _create_nodes_from_results(self,
                                  summaries: List[Dict[str, Any]],
                                  chunks_by_document: Dict[str, List[Dict[str, Any]]]) -> List[NodeWithScore]:
        """
        Create nodes from summary and chunk results.
        
        Args:
            summaries: List of summary results
            chunks_by_document: Dictionary of chunks by document ID
            
        Returns:
            List of NodeWithScore objects
        """
        nodes = []
        
        # First, add summary nodes
        for summary in summaries:
            # Create summary node
            summary_node = TextNode(
                text=summary['summary_text'],
                id_=f"postgres_summary_{summary['id']}",
                metadata={
                    **summary.get('metadata', {}),
                    'node_type': 'summary',
                    'summary_id': summary['id'],
                    'document_id': summary['document_id'],
                    'document_title': summary.get('document_title'),
                    'document_path': summary.get('document_path'),
                    'retrieval_strategy': 'summary',
                    'similarity_score': float(summary['similarity']),
                    'source': 'postgres'
                }
            )
            
            nodes.append(NodeWithScore(
                node=summary_node,
                score=float(summary['similarity'])
            ))
            
            # Add chunks from this document
            doc_id = summary['document_id']
            if doc_id in chunks_by_document:
                for chunk in chunks_by_document[doc_id]:
                    chunk_node = TextNode(
                        text=chunk['content'],
                        id_=f"postgres_summary_chunk_{chunk['id']}",
                        metadata={
                            **chunk.get('metadata', {}),
                            'node_type': 'chunk',
                            'chunk_id': chunk['id'],
                            'document_id': chunk['document_id'],
                            'chunk_index': chunk['chunk_index'],
                            'document_title': summary.get('document_title'),
                            'document_path': summary.get('document_path'),
                            'retrieval_strategy': 'summary',
                            'retrieved_via': 'document_summary',
                            'summary_similarity': float(summary['similarity']),
                            'chunk_similarity': float(chunk['similarity']),
                            'source': 'postgres'
                        }
                    )
                    
                    # Score based on both summary and chunk similarity
                    combined_score = 0.6 * float(summary['similarity']) + 0.4 * float(chunk['similarity'])
                    
                    nodes.append(NodeWithScore(
                        node=chunk_node,
                        score=combined_score
                    ))
        
        # Sort all nodes by score
        nodes.sort(key=lambda x: x.score, reverse=True)
        
        return nodes
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using summary-first approach.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of nodes with scores
        """
        start_time = time.time()
        query = query_bundle.query_str
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            
            # Step 1: Retrieve relevant summaries
            summaries = self._retrieve_summaries(
                query_embedding=query_embedding,
                top_k=self.default_top_k,
                conn=conn
            )
            
            if not summaries:
                logger.warning("No summaries found for query")
                conn.close()
                return []
            
            # Step 2: Get document IDs from summaries
            document_ids = [s['document_id'] for s in summaries]
            
            # Step 3: Retrieve chunks for these documents
            chunks_by_document = self._retrieve_chunks_for_documents(
                document_ids=document_ids,
                query_embedding=query_embedding,
                conn=conn
            )
            
            conn.close()
            
            # Step 4: Create nodes from results
            nodes = self._create_nodes_from_results(
                summaries=summaries,
                chunks_by_document=chunks_by_document
            )
            
            # Step 5: Limit to top_k if needed
            if len(nodes) > self.default_top_k:
                nodes = nodes[:self.default_top_k]
            
            execution_time = time.time() - start_time
            logger.info(f"Summary retrieval completed in {execution_time:.2f}s. "
                       f"Retrieved {len(nodes)} nodes from {len(summaries)} documents.")
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error in summary retrieval: {e}")
            if 'conn' in locals():
                conn.close()
            return []
    
    def retrieve_summaries_only(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve only document summaries without chunks.
        
        Args:
            query: The search query
            top_k: Number of summaries to retrieve
            
        Returns:
            List of summary nodes with scores
        """
        k = top_k or self.default_top_k
        query_embedding = self._get_query_embedding(query)
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            
            summaries = self._retrieve_summaries(
                query_embedding=query_embedding,
                top_k=k,
                conn=conn
            )
            
            conn.close()
            
            # Create nodes from summaries only
            nodes = []
            for summary in summaries:
                summary_node = TextNode(
                    text=summary['summary_text'],
                    id_=f"postgres_summary_{summary['id']}",
                    metadata={
                        **summary.get('metadata', {}),
                        'node_type': 'summary',
                        'summary_id': summary['id'],
                        'document_id': summary['document_id'],
                        'document_title': summary.get('document_title'),
                        'document_path': summary.get('document_path'),
                        'retrieval_strategy': 'summary_only',
                        'similarity_score': float(summary['similarity']),
                        'source': 'postgres'
                    }
                )
                
                nodes.append(NodeWithScore(
                    node=summary_node,
                    score=float(summary['similarity'])
                ))
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error retrieving summaries: {e}")
            return []
    
    def retrieve_with_metadata_filter(self,
                                    query: str,
                                    metadata_filters: Dict[str, Any],
                                    top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve summaries and chunks with metadata filtering.
        
        Args:
            query: The search query
            metadata_filters: Dictionary of metadata filters
            top_k: Number of results to retrieve
            
        Returns:
            List of nodes with scores
        """
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Store original settings
        original_top_k = self.default_top_k
        self.default_top_k = top_k or original_top_k
        
        # TODO: Implement metadata filtering in SQL queries
        # For now, retrieve normally and filter results
        nodes = self._retrieve(query_bundle)
        
        # Filter nodes based on metadata
        filtered_nodes = []
        for node in nodes:
            match = True
            for key, value in metadata_filters.items():
                if key not in node.node.metadata or node.node.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_nodes.append(node)
        
        # Restore original settings
        self.default_top_k = original_top_k
        
        return filtered_nodes[:self.default_top_k]