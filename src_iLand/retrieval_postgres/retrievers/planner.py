"""
PostgreSQL Query Planner Retriever for iLand Data

Implements multi-step query planning and execution for complex Thai land deed queries.
Uses PostgreSQL for vector storage and retrieval with BGE-M3 embeddings.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.llms.openai import OpenAI

from ..config import PostgresRetrievalConfig
from src_iLand.docs_embedding_postgres.bge_embedding_processor import BGEEmbeddingProcessor
from src_iLand.common.thai_provinces import THAI_PROVINCES

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Query execution plan with sub-queries and reasoning."""
    sub_queries: List[str]
    reasoning: str
    original_query: str


class PostgresPlannerRetriever(BaseRetriever):
    """PostgreSQL-based query planner retriever for complex Thai land deed queries."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 default_top_k: int = 5,
                 similarity_threshold: Optional[float] = None,
                 use_bge_embeddings: bool = True):
        """
        Initialize PostgreSQL planner retriever.
        
        Args:
            config: PostgreSQL configuration
            default_top_k: Default number of nodes to retrieve per sub-query
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
            raise NotImplementedError("Only BGE embeddings are supported for PostgreSQL retrieval")
        
        # Initialize LLM for query planning
        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        
        # Thai location keywords for query decomposition (all 77 provinces)
        self.thai_provinces = THAI_PROVINCES
        
        self.land_deed_types = ["โฉนด", "นส.3", "นส.4", "ส.ค.1", "น.ส.3ก"]
    
    def _create_query_plan(self, query: str) -> QueryPlan:
        """
        Create a query execution plan for Thai land deed queries.
        
        Args:
            query: Original query
            
        Returns:
            QueryPlan with sub-queries and reasoning
        """
        prompt = f"""
You are an expert at analyzing Thai land deed queries and breaking them into sub-queries.

Original Query: "{query}"

Analyze this query and break it into 2-4 focused sub-queries that will help retrieve comprehensive information about Thai land deeds. Consider:

1. Location-based aspects (จังหวัด, อำเภอ, ตำบล, province, district, subdistrict)
2. Land deed types (โฉนด, นส.3, นส.4, ส.ค.1, น.ส.3ก)
3. Property characteristics (area size, boundaries, land use)
4. Legal or procedural aspects (ownership, transfer, registration)
5. Historical or temporal aspects (dates, changes over time)

Respond in this format:
SUB_QUERY_1: [specific focused query]
SUB_QUERY_2: [specific focused query]
SUB_QUERY_3: [specific focused query if needed]
SUB_QUERY_4: [specific focused query if needed]
REASONING: [brief explanation of the decomposition strategy]

Keep sub-queries focused and specific. Use Thai terms when appropriate for land deed context.
Ensure each sub-query targets a different aspect of the original query.
"""
        
        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse the response
            sub_queries = []
            reasoning = "Query decomposition for comprehensive land deed search"
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('SUB_QUERY_'):
                    sub_query = line.split(':', 1)[1].strip()
                    if sub_query:
                        sub_queries.append(sub_query)
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Fallback if parsing fails
            if not sub_queries:
                sub_queries = self._fallback_query_decomposition(query)
                reasoning = "Fallback heuristic-based query decomposition"
            
            logger.info(f"Query plan created with {len(sub_queries)} sub-queries")
            logger.debug(f"Sub-queries: {sub_queries}")
            
            return QueryPlan(
                sub_queries=sub_queries,
                reasoning=reasoning,
                original_query=query
            )
            
        except Exception as e:
            logger.warning(f"LLM query planning failed: {e}. Using fallback.")
            sub_queries = self._fallback_query_decomposition(query)
            return QueryPlan(
                sub_queries=sub_queries,
                reasoning="Fallback heuristic-based query decomposition",
                original_query=query
            )
    
    def _fallback_query_decomposition(self, query: str) -> List[str]:
        """
        Fallback query decomposition using heuristics.
        
        Args:
            query: Original query
            
        Returns:
            List of sub-queries
        """
        sub_queries = [query]  # Start with original
        
        # Add location-specific query if location terms detected
        for province in self.thai_provinces:
            if province in query:
                sub_queries.append(f"ที่ดินใน{province} โฉนดที่ดิน")
                sub_queries.append(f"{province} การถือครองที่ดิน")
                break
        
        # Add land deed type query if not specific
        if not any(deed_type in query for deed_type in self.land_deed_types):
            sub_queries.append("ประเภทโฉนดที่ดิน นส.3 นส.4 ส.ค.1")
        
        # Add procedural query if keywords detected
        procedural_keywords = ["ขั้นตอน", "วิธีการ", "การโอน", "การจดทะเบียน", "procedure", "transfer", "registration"]
        if any(keyword in query.lower() for keyword in procedural_keywords):
            sub_queries.append("ขั้นตอนการโอนที่ดิน การจดทะเบียน")
        
        return sub_queries[:4]  # Limit to 4 sub-queries
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text."""
        embeddings = self.embedding_processor.get_text_embeddings([query])
        return embeddings[0]
    
    def _execute_sub_query(self, 
                          sub_query: str, 
                          top_k: int,
                          conn,
                          existing_ids: set) -> List[Dict[str, Any]]:
        """
        Execute a single sub-query against PostgreSQL.
        
        Args:
            sub_query: The sub-query to execute
            top_k: Number of results to retrieve
            conn: Database connection
            existing_ids: Set of already retrieved chunk IDs
            
        Returns:
            List of result dictionaries
        """
        query_embedding = self._get_query_embedding(sub_query)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Build exclusion clause if we have existing IDs
            exclusion_clause = ""
            params = [query_embedding, query_embedding, query_embedding, self.similarity_threshold]
            
            if existing_ids:
                placeholders = ','.join(['%s'] * len(existing_ids))
                exclusion_clause = f"AND c.id NOT IN ({placeholders})"
                params.extend(list(existing_ids))
            
            params.append(top_k)
            
            # Execute vector similarity search
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
                {exclusion_clause}
                ORDER BY distance
                LIMIT %s
            """, params)
            
            results = cursor.fetchall()
            cursor.close()
            
            # Add sub-query info to results
            for result in results:
                result['sub_query'] = sub_query
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing sub-query '{sub_query}': {e}")
            cursor.close()
            return []
    
    def _deduplicate_and_rank(self, 
                             all_results: List[Dict[str, Any]], 
                             original_query: str,
                             top_k: int) -> List[Dict[str, Any]]:
        """
        Deduplicate and re-rank results based on relevance to original query.
        
        Args:
            all_results: All results from sub-queries
            original_query: Original query for relevance scoring
            top_k: Number of final results to return
            
        Returns:
            Ranked and deduplicated results
        """
        # Group results by chunk ID to handle duplicates
        results_by_id = {}
        
        for result in all_results:
            chunk_id = result['id']
            
            if chunk_id not in results_by_id:
                results_by_id[chunk_id] = result
            else:
                # Keep the result with higher similarity score
                if result['similarity'] > results_by_id[chunk_id]['similarity']:
                    results_by_id[chunk_id] = result
        
        # Get unique results
        unique_results = list(results_by_id.values())
        
        # Re-rank based on relevance to original query
        original_embedding = self._get_query_embedding(original_query)
        
        for result in unique_results:
            # Calculate final score as weighted combination
            sub_query_score = result['similarity']
            
            # If we have the embedding, calculate direct similarity to original query
            # For now, use the sub-query score with a boost if it matched multiple sub-queries
            result['final_score'] = sub_query_score
        
        # Sort by final score
        unique_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return unique_results[:top_k]
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using query planning approach.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of nodes with scores
        """
        start_time = time.time()
        query = query_bundle.query_str
        
        # Step 1: Create query plan
        query_plan = self._create_query_plan(query)
        
        # Step 2: Execute sub-queries
        all_results = []
        existing_ids = set()
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            
            for i, sub_query in enumerate(query_plan.sub_queries):
                logger.debug(f"Executing sub-query {i+1}/{len(query_plan.sub_queries)}: {sub_query}")
                
                # Get more results for early sub-queries to ensure diversity
                sub_k = self.default_top_k * 2 if i == 0 else self.default_top_k
                
                sub_results = self._execute_sub_query(
                    sub_query=sub_query,
                    top_k=sub_k,
                    conn=conn,
                    existing_ids=existing_ids
                )
                
                # Track retrieved IDs
                for result in sub_results:
                    existing_ids.add(result['id'])
                
                all_results.extend(sub_results)
                logger.debug(f"Sub-query returned {len(sub_results)} results")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error during query plan execution: {e}")
            if 'conn' in locals():
                conn.close()
            return []
        
        # Step 3: Deduplicate and rank results
        final_results = self._deduplicate_and_rank(
            all_results=all_results,
            original_query=query,
            top_k=self.default_top_k
        )
        
        # Step 4: Convert to nodes
        nodes = []
        for result in final_results:
            # Create text node
            node = TextNode(
                text=result['content'],
                id_=f"postgres_planner_chunk_{result['id']}",
                metadata={
                    **result.get('metadata_', {}),
                    'chunk_id': result['id'],
                    'document_id': result['document_id'],
                    'chunk_index': result['chunk_index'],
                    'document_title': result.get('document_title'),
                    'document_path': result.get('document_path'),
                    'retrieval_strategy': 'planner',
                    'query_plan_reasoning': query_plan.reasoning,
                    'sub_query': result.get('sub_query'),
                    'similarity_score': float(result['similarity']),
                    'final_score': float(result['final_score']),
                    'source': 'postgres'
                }
            )
            
            # Create node with score
            node_with_score = NodeWithScore(
                node=node,
                score=float(result['final_score'])
            )
            
            nodes.append(node_with_score)
        
        execution_time = time.time() - start_time
        logger.info(f"Query planning retrieval completed in {execution_time:.2f}s. "
                   f"Retrieved {len(nodes)} nodes from {len(all_results)} total results.")
        
        return nodes
    
    def retrieve_with_custom_plan(self,
                                 query: str,
                                 sub_queries: List[str],
                                 top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve using a custom query plan.
        
        Args:
            query: Original query
            sub_queries: List of sub-queries to execute
            top_k: Number of results to retrieve
            
        Returns:
            List of nodes with scores
        """
        # Create custom query plan
        query_plan = QueryPlan(
            sub_queries=sub_queries,
            reasoning="Custom user-provided query plan",
            original_query=query
        )
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Store the plan and retrieve
        self._last_query_plan = query_plan
        return self._retrieve(query_bundle)