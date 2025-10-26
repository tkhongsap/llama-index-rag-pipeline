"""
Planner Retriever Adapter for Sales Promotions

Strategy 7: Query decomposition with LLM for complex multi-part questions.
Best for analytical queries like "Compare average discounts across categories"

Performance Target: < 500ms
"""

import json
from typing import List, Optional, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from .base import BaseRetrieverAdapter


class PlannerRetrieverAdapter(BaseRetrieverAdapter):
    """
    Query planner retriever for complex multi-part queries.
    
    Uses GPT-5 to:
    1. Decompose complex queries into sub-questions
    2. Execute sub-queries (parallel or sequential)
    3. Aggregate results
    4. Synthesize final answer
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        llm: OpenAI,
        max_sub_questions: int = 5,
        enable_parallel: bool = True,
        max_depth: int = 3
    ):
        """
        Initialize planner retriever.
        
        Args:
            vector_index: VectorStoreIndex for sub-query execution
            llm: LLM for query decomposition (GPT-5)
            max_sub_questions: Maximum number of sub-questions
            enable_parallel: Enable parallel sub-query execution
            max_depth: Maximum query decomposition depth
        """
        super().__init__("planner")
        self.vector_index = vector_index
        self.llm = llm
        self.max_sub_questions = max_sub_questions
        self.enable_parallel = enable_parallel
        self.max_depth = max_depth
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes using query planning and decomposition.
        
        Args:
            query: Complex search query (supports Thai text)
            top_k: Number of final results
            
        Returns:
            List of NodeWithScore objects tagged with strategy metadata
        """
        k = top_k if top_k is not None else 5
        
        # Decompose query into sub-questions
        sub_questions = self._decompose_query(query)
        
        # If no decomposition needed (simple query), use direct retrieval
        if not sub_questions or len(sub_questions) == 1:
            retriever = self.vector_index.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(query)
            return self._tag_nodes_with_strategy(nodes)
        
        # Execute sub-queries
        all_results = []
        for sub_q in sub_questions:
            retriever = self.vector_index.as_retriever(similarity_top_k=3)
            sub_results = retriever.retrieve(sub_q)
            all_results.extend(sub_results)
        
        # Deduplicate and rank by score
        seen_ids = set()
        unique_results = []
        
        for node in sorted(all_results, key=lambda x: x.score, reverse=True):
            node_id = node.node.id_
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_results.append(node)
                if len(unique_results) >= k:
                    break
        
        # Tag nodes with strategy info and decomposition metadata
        tagged_nodes = self._tag_nodes_with_strategy(unique_results)
        
        # Add decomposition info
        for node in tagged_nodes:
            node.node.metadata['sub_questions'] = sub_questions
        
        return tagged_nodes
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-questions using GPT-5.
        
        Args:
            query: Original complex query
            
        Returns:
            List of sub-questions
        """
        prompt = f"""
Analyze this sales promotion query and decompose it into simpler sub-questions if needed.

Query: {query}

If the query is complex (contains "compare", "analyze", "calculate", multiple conditions),
break it down into 2-5 simpler sub-questions.

If the query is simple, return it as-is.

Examples:
1. Complex: "Compare average discounts for beer vs water"
   → ["Find all beer promotions", "Find all water promotions", "Calculate average discount for each"]

2. Simple: "Find beer promotions"
   → ["Find beer promotions"]

Return JSON array of sub-questions:
["sub-question 1", "sub-question 2", ...]
"""
        
        try:
            response = self.llm.complete(prompt)
            sub_questions = json.loads(response.text.strip())
            
            # Limit to max_sub_questions
            if len(sub_questions) > self.max_sub_questions:
                sub_questions = sub_questions[:self.max_sub_questions]
            
            return sub_questions
        except Exception as e:
            print(f"  ⚠️  Query decomposition failed: {e}")
            # Fall back to original query
            return [query]
