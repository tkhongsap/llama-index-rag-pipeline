"""
Query Planning Agent Retriever Adapter for iLand Data

Implements multi-step query planning and execution for complex Thai land deed queries.
Adapted from src/agentic_retriever/retrievers/planner.py for iLand data.
"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

from .base import BaseRetrieverAdapter

# Import from updated iLand embedding loading modules
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from load_embedding import iLandIndexReconstructor, EmbeddingConfig
except ImportError as e:
    print(f"Warning: Could not import iLand embedding utilities: {e}")
    iLandIndexReconstructor = None
    EmbeddingConfig = None


class PlannerRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for query planning agent retrieval on iLand data."""
    
    def __init__(self, 
                 index: VectorStoreIndex,
                 api_key: Optional[str] = None,
                 default_top_k: int = 5):
        """
        Initialize query planning retriever adapter for iLand data.
        
        Args:
            index: Vector store index for Thai land deed data
            api_key: OpenAI API key for LLM planning
            default_top_k: Default number of nodes to retrieve
        """
        super().__init__("planner")
        self.index = index
        self.default_top_k = default_top_k
        self.retriever = index.as_retriever(similarity_top_k=default_top_k * 2)
        
        # Setup LLM for query planning
        if api_key:
            self.llm = OpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )
        else:
            self.llm = Settings.llm
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using query planning approach on iLand data.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Step 1: Analyze query and create execution plan
        query_plan = self._create_query_plan(query)
        
        # Step 2: Execute planned sub-queries
        all_nodes = []
        for sub_query in query_plan["sub_queries"]:
            sub_nodes = self.retriever.retrieve(sub_query)
            all_nodes.extend(sub_nodes)
        
        # Step 3: Deduplicate and rank results
        unique_nodes = self._deduplicate_nodes(all_nodes)
        
        # Step 4: Apply final ranking based on original query
        ranked_nodes = self._rank_nodes_by_relevance(unique_nodes, query)
        
        # Step 5: Take top_k results
        final_nodes = ranked_nodes[:k]
        
        # Tag nodes with strategy and plan info
        for node in final_nodes:
            node.node.metadata["query_plan"] = query_plan["reasoning"]
        
        return self._tag_nodes_with_strategy(final_nodes)
    
    def _create_query_plan(self, query: str) -> Dict[str, Any]:
        """
        Create a query execution plan for Thai land deed queries.
        
        Args:
            query: Original query
            
        Returns:
            Query plan with sub-queries and reasoning
        """
        prompt = f"""
You are an expert at analyzing Thai land deed queries and breaking them into sub-queries.

Original Query: "{query}"

Analyze this query and break it into 2-4 focused sub-queries that will help retrieve comprehensive information about Thai land deeds. Consider:

1. Location-based aspects (province, district, subdistrict)
2. Land deed types (โฉนด, นส.3, นส.4, ส.ค.1)
3. Property characteristics 
4. Legal or procedural aspects
5. Historical or temporal aspects

Respond in this format:
SUB_QUERY_1: [specific focused query]
SUB_QUERY_2: [specific focused query]
SUB_QUERY_3: [specific focused query if needed]
REASONING: [brief explanation of the decomposition strategy]

Keep sub-queries focused and specific. Use Thai terms when appropriate for land deed context.
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
                sub_queries = [query]  # Use original query as fallback
            
            return {
                "sub_queries": sub_queries,
                "reasoning": reasoning
            }
            
        except Exception as e:
            # Fallback to simple query decomposition
            return self._fallback_query_decomposition(query)
    
    def _fallback_query_decomposition(self, query: str) -> Dict[str, Any]:
        """
        Fallback query decomposition using heuristics.
        
        Args:
            query: Original query
            
        Returns:
            Simple query plan
        """
        sub_queries = [query]  # Start with original
        
        # Add location-specific query if location terms detected
        thai_provinces = ["กรุงเทพ", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี"]
        for province in thai_provinces:
            if province in query:
                sub_queries.append(f"ที่ดินใน{province}")
                break
        
        # Add land deed type query if not specific
        land_terms = ["โฉนด", "นส.3", "นส.4", "ส.ค.1"]
        if not any(term in query for term in land_terms):
            sub_queries.append("ประเภทโฉนดที่ดิน")
        
        return {
            "sub_queries": sub_queries[:3],  # Limit to 3 sub-queries
            "reasoning": "Heuristic-based query decomposition"
        }
    
    def _deduplicate_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Remove duplicate nodes based on content similarity.
        
        Args:
            nodes: List of nodes to deduplicate
            
        Returns:
            Deduplicated list of nodes
        """
        seen_node_ids = set()
        unique_nodes = []
        
        for node in nodes:
            node_id = getattr(node.node, 'node_id', str(hash(node.node.text)))
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                unique_nodes.append(node)
        
        return unique_nodes
    
    def _rank_nodes_by_relevance(self, nodes: List[NodeWithScore], 
                                original_query: str) -> List[NodeWithScore]:
        """
        Re-rank nodes by relevance to original query.
        
        Args:
            nodes: Nodes to rank
            original_query: Original query for relevance scoring
            
        Returns:
            Ranked list of nodes
        """
        # Simple relevance scoring based on keyword overlap
        query_terms = set(original_query.lower().split())
        
        scored_nodes = []
        for node in nodes:
            text_terms = set(node.node.text.lower().split())
            overlap_score = len(query_terms.intersection(text_terms)) / len(query_terms)
            
            # Combine with original similarity score
            combined_score = 0.7 * node.score + 0.3 * overlap_score
            
            new_node = NodeWithScore(node=node.node, score=combined_score)
            scored_nodes.append(new_node)
        
        # Sort by combined score
        scored_nodes.sort(key=lambda x: x.score, reverse=True)
        return scored_nodes
    
    @classmethod
    def from_iland_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5
    ) -> "PlannerRetrieverAdapter":
        """
        Create adapter from iLand embeddings.
        
        Args:
            embeddings: List of iLand embedding dictionaries
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            
        Returns:
            PlannerRetrieverAdapter instance for iLand data
        """
        if not iLandIndexReconstructor or not EmbeddingConfig:
            raise ImportError("iLand embedding utilities not available")
        
        config = EmbeddingConfig(api_key=api_key)
        reconstructor = iLandIndexReconstructor(config=config)
        index = reconstructor.create_vector_index_from_embeddings(
            embeddings, show_progress=False
        )
        
        return cls(index, api_key, default_top_k) 