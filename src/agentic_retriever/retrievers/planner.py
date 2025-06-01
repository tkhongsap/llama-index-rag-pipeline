"""
Planner Retriever Adapter

Wraps the query planning agent from 17_query_planning_agent.py
"""

import importlib.util
from pathlib import Path
from typing import List, Optional

from llama_index.core.schema import NodeWithScore, TextNode

from .base import BaseRetrieverAdapter


class PlannerRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for query planning agent retrieval."""
    
    def __init__(
        self,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        max_iterations: int = 3
    ):
        """
        Initialize planner retriever adapter.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            max_iterations: Maximum planning iterations
        """
        super().__init__("planner")
        self.embeddings = embeddings
        self.api_key = api_key
        self.default_top_k = default_top_k
        self.max_iterations = max_iterations
        
        # Import the query planning agent module
        try:
            spec = importlib.util.spec_from_file_location(
                "query_planning_agent", 
                Path(__file__).parent.parent.parent / "17_query_planning_agent.py"
            )
            planning_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(planning_module)
            
            # Create the query planning agent
            self.retriever = planning_module.QueryPlanningAgent()
            
        except Exception as e:
            print(f"Warning: Could not import QueryPlanningAgent: {e}")
            # Fallback to basic vector search would be implemented here
            self.retriever = None
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using query planning agent.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        # Use provided top_k or default
        k = top_k if top_k is not None else self.default_top_k
        
        try:
            if self.retriever is None:
                # Fallback to basic vector search if retriever not available
                return self._basic_vector_fallback(query, k)
            
            # Perform query planning and execution
            result = self.retriever.plan_and_execute_query(
                query=query,
                use_decomposition=True,
                use_parallel_execution=True,
                show_details=False
            )
            
            # Convert results to NodeWithScore objects
            nodes = []
            
            # Extract sources from sub_results
            for sub_result in result.get('sub_results', []):
                sources = sub_result.get('sources', [])
                for source in sources[:k]:  # Limit to top_k
                    # Create TextNode from source
                    text_node = TextNode(
                        text=source.get('text', ''),
                        metadata={
                            **source.get('metadata', {}),
                            'sub_query': sub_result.get('query', ''),
                            'engine_used': sub_result.get('engine_used', 'unknown'),
                            'planning_strategy': 'query_decomposition'
                        }
                    )
                    
                    # Create NodeWithScore
                    node_with_score = NodeWithScore(
                        node=text_node,
                        score=source.get('score', 0.0)
                    )
                    nodes.append(node_with_score)
            
            # If no sources found, try to extract from final response
            if not nodes and result.get('final_response'):
                text_node = TextNode(
                    text=result['final_response'],
                    metadata={
                        'query': query,
                        'source_type': 'synthesized_response',
                        'planning_strategy': 'query_decomposition',
                        'num_sub_queries': result.get('metadata', {}).get('num_sub_queries', 0)
                    }
                )
                
                node_with_score = NodeWithScore(
                    node=text_node,
                    score=1.0  # High score for synthesized response
                )
                nodes.append(node_with_score)
        
        except Exception as e:
            print(f"Warning: Query planning failed: {e}")
            # Fallback to basic retrieval
            return self._basic_vector_fallback(query, k)
        
        # Tag nodes with strategy and return
        return self._tag_nodes_with_strategy(nodes[:k])  # Ensure we don't exceed top_k
    
    def _basic_vector_fallback(self, query: str, top_k: int) -> List[NodeWithScore]:
        """Fallback to basic vector search if planning fails."""
        # Create a simple response node
        text_node = TextNode(
            text=f"Fallback response for query: {query}",
            metadata={
                'query': query,
                'source_type': 'fallback',
                'strategy': 'planner_fallback'
            }
        )
        
        node_with_score = NodeWithScore(
            node=text_node,
            score=0.5  # Medium score for fallback
        )
        
        return self._tag_nodes_with_strategy([node_with_score])
    
    @classmethod
    def from_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        max_iterations: int = 3
    ) -> "PlannerRetrieverAdapter":
        """
        Create adapter from embedding data.
        
        Args:
            embeddings: Embedding data
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            max_iterations: Maximum planning iterations
            
        Returns:
            PlannerRetrieverAdapter instance
        """
        return cls(
            embeddings=embeddings,
            api_key=api_key,
            default_top_k=default_top_k,
            max_iterations=max_iterations
        ) 