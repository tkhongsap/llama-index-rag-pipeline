"""
Planner Retriever Adapter

Wraps the query planning agent from 17_query_planning_agent.py
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

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
        
        # Import and create the query planning agent
        from query_planning_agent import QueryPlanningAgent
        self.retriever = QueryPlanningAgent(
            embeddings=embeddings,
            api_key=api_key,
            max_iterations=max_iterations
        )
    
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
        
        # Perform query planning and retrieval
        result = self.retriever.plan_and_retrieve(
            query=query,
            top_k=k
        )
        
        # Convert retrieved chunks to NodeWithScore objects
        nodes = []
        for chunk_info in result.get('retrieved_chunks', []):
            # Create TextNode from chunk info
            text_node = TextNode(
                text=chunk_info.get('text', ''),
                metadata=chunk_info.get('metadata', {})
            )
            
            # Create NodeWithScore
            node_with_score = NodeWithScore(
                node=text_node,
                score=chunk_info.get('score', 0.0)
            )
            nodes.append(node_with_score)
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(nodes)
    
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