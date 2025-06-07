"""
Base Retriever Adapter for iLand Data

Defines the common interface that all iLand retrieval strategy adapters must implement.
This mirrors the src/agentic_retriever/retrievers/base.py but adapted for Thai land deed data.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore


class BaseRetrieverAdapter(ABC):
    """Base class for all iLand retrieval strategy adapters."""
    
    def __init__(self, strategy_name: str):
        """Initialize the adapter with strategy name."""
        self.strategy_name = strategy_name
    
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for the given query.
        
        Args:
            query: The search query (may contain Thai text)
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
                node.node.metadata['data_source'] = 'iland'
            else:
                node.node.metadata = {
                    'retrieval_strategy': self.strategy_name,
                    'data_source': 'iland'
                }
        return nodes
    
    @property
    def name(self) -> str:
        """Get the strategy name."""
        return self.strategy_name 