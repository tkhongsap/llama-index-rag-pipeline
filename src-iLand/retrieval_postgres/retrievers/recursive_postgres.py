"""
Recursive PostgreSQL Retriever for iLand Data

Implements hierarchical retrieval: searches summaries first to identify
relevant documents, then retrieves detailed chunks from those documents.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict

from llama_index.core.schema import NodeWithScore

from ..base_retriever import BasePostgresRetriever
from ..config import PostgresConfig

logger = logging.getLogger(__name__)


class RecursivePostgresRetriever(BasePostgresRetriever):
    """
    Recursive retriever implementing hierarchical search strategy.
    
    Step 1: Search document summaries to find relevant documents
    Step 2: Search detailed chunks within those documents
    Step 3: Combine and rank results
    """
    
    def __init__(
        self,
        config: PostgresConfig,
        summary_threshold: float = 0.3,
        summary_top_k: int = 10,
        chunks_per_document: int = 5
    ):
        """
        Initialize recursive PostgreSQL retriever.
        
        Args:
            config: PostgreSQL configuration
            summary_threshold: Similarity threshold for summary search
            summary_top_k: Number of documents to find via summaries
            chunks_per_document: Max chunks to retrieve per relevant document
        """
        super().__init__(config, "recursive_postgres")
        self.summary_threshold = summary_threshold
        self.summary_top_k = summary_top_k
        self.chunks_per_document = chunks_per_document
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using recursive hierarchical search.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Total number of nodes to retrieve
            filters: Metadata filters
            
        Returns:
            List of NodeWithScore objects from hierarchical search
        """
        start_time = time.time()
        
        top_k = top_k or self.config.default_top_k
        validated_filters = self._validate_filters(filters)
        
        try:
            # Generate query embedding once
            query_embedding = self._get_query_embedding(query)
            
            # Step 1: Search summaries to find relevant documents
            relevant_documents = self._find_relevant_documents(
                query_embedding=query_embedding,
                filters=validated_filters
            )
            
            if not relevant_documents:
                logger.warning("No relevant documents found in summary search")
                return []
            
            # Step 2: Search chunks within relevant documents
            chunk_results = self._search_chunks_in_documents(
                query_embedding=query_embedding,
                relevant_documents=relevant_documents,
                filters=validated_filters
            )
            
            # Step 3: Combine with summary information and rank
            combined_results = self._combine_summary_and_chunk_results(
                query_embedding=query_embedding,
                summary_results=relevant_documents,
                chunk_results=chunk_results
            )
            
            # Step 4: Apply final filtering and ranking
            filtered_results = self._apply_similarity_threshold(combined_results)
            filtered_results = filtered_results[:top_k]
            
            # Convert to nodes
            nodes = self._format_results_as_nodes(filtered_results)
            
            # Log metrics
            execution_time = time.time() - start_time
            self._log_retrieval_metrics(
                query=query,
                num_results=len(nodes),
                execution_time=execution_time,
                additional_info={
                    "relevant_documents": len(relevant_documents),
                    "chunk_results": len(chunk_results),
                    "summary_threshold": self.summary_threshold,
                    "chunks_per_document": self.chunks_per_document
                }
            )
            
            logger.info(f"Recursive retrieval completed: {len(nodes)} nodes from {len(relevant_documents)} documents")
            return nodes
            
        except Exception as e:
            logger.error(f"Recursive retrieval failed: {e}")
            raise
    
    def _find_relevant_documents(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search summaries to find relevant documents.
        
        Args:
            query_embedding: Query embedding vector
            filters: Metadata filters
            
        Returns:
            List of relevant document summaries with scores
        """
        try:
            # Search summaries table
            summary_results = self._vector_similarity_search(
                query_embedding=query_embedding,
                table_name=self.config.summaries_table,
                top_k=self.summary_top_k,
                similarity_threshold=self.summary_threshold,
                metadata_filters=filters,
                select_fields=["id", "deed_id", "summary_text", "metadata"]
            )
            
            logger.debug(f"Found {len(summary_results)} relevant documents from summaries")
            return summary_results
            
        except Exception as e:
            logger.error(f"Failed to search summaries: {e}")
            return []
    
    def _search_chunks_in_documents(
        self,
        query_embedding: List[float],
        relevant_documents: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search chunks within the relevant documents.
        
        Args:
            query_embedding: Query embedding vector
            relevant_documents: Documents identified from summary search
            filters: Metadata filters
            
        Returns:
            List of relevant chunks with scores
        """
        all_chunk_results = []
        
        # Extract node_ids from relevant documents
        relevant_node_ids = [doc.get("node_id") for doc in relevant_documents]
        relevant_node_ids = [node_id for node_id in relevant_node_ids if node_id]
        if not relevant_node_ids:
            return []
        
        try:
            # Create metadata filter for relevant documents
            document_filters = filters.copy() if filters else {}
            document_filters["node_id"] = relevant_node_ids
            
            # Search chunks in relevant documents
            chunk_results = self._vector_similarity_search(
                query_embedding=query_embedding,
                table_name=self.config.chunks_table,
                top_k=len(relevant_node_ids) * self.chunks_per_document,
                similarity_threshold=self.config.similarity_threshold,
                metadata_filters=document_filters
            )
            
            # Group chunks by document and limit per document
            chunks_by_doc = defaultdict(list)
            for chunk in chunk_results:
                node_id = chunk.get("node_id")
                if node_id:
                    chunks_by_doc[node_id].append(chunk)
            
            # Limit chunks per document and maintain diversity
            for node_id, chunks in chunks_by_doc.items():
                # Sort by similarity score
                chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
                
                # Take top chunks for this document
                selected_chunks = chunks[:self.chunks_per_document]
                all_chunk_results.extend(selected_chunks)
            
            logger.debug(f"Found {len(all_chunk_results)} relevant chunks from {len(chunks_by_doc)} documents")
            return all_chunk_results
            
        except Exception as e:
            logger.error(f"Failed to search chunks in documents: {e}")
            return []
    
    def _combine_summary_and_chunk_results(
        self,
        query_embedding: List[float],
        summary_results: List[Dict[str, Any]],
        chunk_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine summary and chunk results with enhanced scoring.
        
        Args:
            query_embedding: Query embedding vector
            summary_results: Summary search results
            chunk_results: Chunk search results
            
        Returns:
            Combined and enhanced results
        """
        # Create summary scores lookup
        summary_scores = {}
        for summary in summary_results:
            deed_id = summary.get("deed_id")
            if deed_id:
                summary_scores[deed_id] = summary.get("similarity_score", 0.0)
        
        # Enhance chunk results with summary information
        enhanced_results = []
        
        for chunk in chunk_results:
            deed_id = chunk.get("deed_id")
            summary_score = summary_scores.get(deed_id, 0.0)
            chunk_score = chunk.get("similarity_score", 0.0)
            
            # Calculate combined score (weighted average)
            combined_score = (0.3 * summary_score) + (0.7 * chunk_score)
            
            # Create enhanced result
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update({
                "similarity_score": combined_score,
                "chunk_similarity": chunk_score,
                "summary_similarity": summary_score,
                "retrieval_method": "recursive_hierarchical"
            })
            
            # Enhance metadata
            if isinstance(enhanced_chunk.get("metadata"), dict):
                enhanced_chunk["metadata"].update({
                    "chunk_similarity": chunk_score,
                    "summary_similarity": summary_score,
                    "combined_score": combined_score,
                    "retrieval_method": "recursive"
                })
            
            enhanced_results.append(enhanced_chunk)
        
        # Sort by combined score
        enhanced_results.sort(
            key=lambda x: x.get("similarity_score", 0.0),
            reverse=True
        )
        
        # Include summary information as separate results for high-scoring summaries
        for summary in summary_results[:3]:  # Top 3 summaries
            if summary.get("similarity_score", 0.0) > 0.8:  # High similarity threshold
                summary_result = {
                    "id": f"summary_{summary.get('id', 'unknown')}",
                    "deed_id": summary.get("deed_id"),
                    "text": summary.get("summary_text", ""),
                    "metadata": summary.get("metadata", {}),
                    "similarity_score": summary.get("similarity_score", 0.0),
                    "chunk_similarity": 0.0,
                    "summary_similarity": summary.get("similarity_score", 0.0),
                    "retrieval_method": "recursive_summary",
                    "result_type": "summary"
                }
                
                # Enhance metadata
                if isinstance(summary_result.get("metadata"), dict):
                    summary_result["metadata"].update({
                        "result_type": "summary",
                        "retrieval_method": "recursive"
                    })
                
                enhanced_results.insert(0, summary_result)  # Add at beginning
        
        return enhanced_results 