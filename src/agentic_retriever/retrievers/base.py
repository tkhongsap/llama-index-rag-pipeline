"""
Base Retriever Adapter

Defines the common interface that all retrieval strategy adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore


class BaseRetrieverAdapter(ABC):
    """Base class for all retrieval strategy adapters."""
    
    def __init__(self, strategy_name: str):
        """Initialize the adapter with strategy name."""
        self.strategy_name = strategy_name
    
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for the given query.
        
        Args:
            query: The search query
            top_k: Number of nodes to retrieve (optional)
            
        Returns:
            List of NodeWithScore objects with strategy metadata
        """
        pass
    
    def _tag_nodes_with_strategy(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Tag retrieved nodes with strategy information."""
        for node in nodes:
            if hasattr(node.node, 'metadata'):
                node.node.metadata['retrieval_strategy'] = self.strategy_name
            else:
                node.node.metadata = {'retrieval_strategy': self.strategy_name}
        return nodes
    
    @property
    def name(self) -> str:
        """Get the strategy name."""
        return self.strategy_name 