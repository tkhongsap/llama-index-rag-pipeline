"""
Adapter classes for PostgreSQL retrievers to maintain compatibility with local interfaces.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever

from ..retrieval.retrievers.base import BaseRetrieverAdapter
from .config import PostgresRetrievalConfig


class PostgresRetrieverAdapter(BaseRetrieverAdapter):
    """
    Base adapter for PostgreSQL retrievers to maintain interface compatibility.
    
    This adapter allows PostgreSQL retrievers to work seamlessly with the existing
    router infrastructure while providing PostgreSQL-specific functionality.
    """
    
    def __init__(self, 
                 postgres_retriever: BaseRetriever,
                 config: Optional[PostgresRetrievalConfig] = None,
                 enable_metrics: bool = True):
        """
        Initialize PostgreSQL retriever adapter.
        
        Args:
            postgres_retriever: The underlying PostgreSQL retriever
            config: PostgreSQL configuration
            enable_metrics: Whether to track retrieval metrics
        """
        self.postgres_retriever = postgres_retriever
        self.config = config or PostgresRetrievalConfig()
        self.enable_metrics = enable_metrics
        self._metrics = {
            "total_queries": 0,
            "total_results": 0,
            "errors": 0
        }
    
    def retrieve(self, query: str, **kwargs) -> List[NodeWithScore]:
        """
        Retrieve nodes using the PostgreSQL retriever.
        
        Args:
            query: The query string
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of nodes with scores
        """
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Update metrics
            if self.enable_metrics:
                self._metrics["total_queries"] += 1
            
            # Perform retrieval
            nodes = self.postgres_retriever.retrieve(query_bundle)
            
            # Update metrics
            if self.enable_metrics:
                self._metrics["total_results"] += len(nodes)
            
            # Add PostgreSQL-specific metadata
            for node in nodes:
                if hasattr(node.node, 'metadata'):
                    node.node.metadata.update({
                        "source": "postgres",
                        "retriever_type": self.get_retriever_type()
                    })
            
            return nodes
            
        except Exception as e:
            if self.enable_metrics:
                self._metrics["errors"] += 1
            raise e
    
    def get_retriever_type(self) -> str:
        """Get the type of this retriever."""
        return self.postgres_retriever.__class__.__name__.replace("Postgres", "").replace("Retriever", "").lower()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retrieval metrics."""
        if self.enable_metrics:
            return self._metrics.copy()
        return {}
    
    def reset_metrics(self):
        """Reset retrieval metrics."""
        if self.enable_metrics:
            self._metrics = {
                "total_queries": 0,
                "total_results": 0,
                "errors": 0
            }


class HybridModeAdapter(BaseRetrieverAdapter):
    """
    Adapter that supports hybrid mode - can use both PostgreSQL and local retrievers.
    
    This allows for gradual migration and fallback capabilities.
    """
    
    def __init__(self,
                 postgres_adapter: BaseRetrieverAdapter,
                 local_adapter: Optional[BaseRetrieverAdapter] = None,
                 mode: str = "postgres_first",
                 fallback_on_error: bool = True):
        """
        Initialize hybrid mode adapter.
        
        Args:
            postgres_adapter: PostgreSQL retriever adapter
            local_adapter: Local file-based retriever adapter
            mode: Retrieval mode ("postgres_first", "local_first", "postgres_only", "local_only")
            fallback_on_error: Whether to fallback to other source on error
        """
        self.postgres_adapter = postgres_adapter
        self.local_adapter = local_adapter
        self.mode = mode
        self.fallback_on_error = fallback_on_error
        self._metrics = {
            "postgres_queries": 0,
            "local_queries": 0,
            "fallbacks": 0,
            "errors": 0
        }
    
    def retrieve(self, query: str, **kwargs) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid mode logic.
        
        Args:
            query: The query string
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of nodes with scores
        """
        if self.mode == "postgres_only":
            self._metrics["postgres_queries"] += 1
            return self._retrieve_postgres(query, **kwargs)
        
        elif self.mode == "local_only":
            if not self.local_adapter:
                raise ValueError("Local adapter not configured for local_only mode")
            self._metrics["local_queries"] += 1
            return self._retrieve_local(query, **kwargs)
        
        elif self.mode == "postgres_first":
            try:
                self._metrics["postgres_queries"] += 1
                return self._retrieve_postgres(query, **kwargs)
            except Exception as e:
                if self.fallback_on_error and self.local_adapter:
                    self._metrics["fallbacks"] += 1
                    self._metrics["local_queries"] += 1
                    return self._retrieve_local(query, **kwargs)
                raise e
        
        elif self.mode == "local_first":
            if not self.local_adapter:
                raise ValueError("Local adapter not configured for local_first mode")
            try:
                self._metrics["local_queries"] += 1
                return self._retrieve_local(query, **kwargs)
            except Exception as e:
                if self.fallback_on_error:
                    self._metrics["fallbacks"] += 1
                    self._metrics["postgres_queries"] += 1
                    return self._retrieve_postgres(query, **kwargs)
                raise e
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _retrieve_postgres(self, query: str, **kwargs) -> List[NodeWithScore]:
        """Retrieve from PostgreSQL."""
        try:
            nodes = self.postgres_adapter.retrieve(query, **kwargs)
            # Add hybrid mode metadata
            for node in nodes:
                if hasattr(node.node, 'metadata'):
                    node.node.metadata['hybrid_source'] = 'postgres'
            return nodes
        except Exception as e:
            self._metrics["errors"] += 1
            raise e
    
    def _retrieve_local(self, query: str, **kwargs) -> List[NodeWithScore]:
        """Retrieve from local files."""
        try:
            nodes = self.local_adapter.retrieve(query, **kwargs)
            # Add hybrid mode metadata
            for node in nodes:
                if hasattr(node.node, 'metadata'):
                    node.node.metadata['hybrid_source'] = 'local'
            return nodes
        except Exception as e:
            self._metrics["errors"] += 1
            raise e
    
    def get_retriever_type(self) -> str:
        """Get the type of this retriever."""
        return f"hybrid_{self.mode}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get hybrid mode metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset metrics."""
        self._metrics = {
            "postgres_queries": 0,
            "local_queries": 0,
            "fallbacks": 0,
            "errors": 0
        }
    
    def set_mode(self, mode: str):
        """Change the hybrid mode."""
        valid_modes = ["postgres_first", "local_first", "postgres_only", "local_only"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.mode = mode


def create_postgres_adapter(
    retriever_class: type,
    config: Optional[PostgresRetrievalConfig] = None,
    **retriever_kwargs
) -> PostgresRetrieverAdapter:
    """
    Factory function to create a PostgreSQL retriever adapter.
    
    Args:
        retriever_class: The PostgreSQL retriever class
        config: PostgreSQL configuration
        **retriever_kwargs: Arguments to pass to retriever constructor
        
    Returns:
        PostgresRetrieverAdapter instance
    """
    # Add config to retriever kwargs if not present
    if 'config' not in retriever_kwargs:
        retriever_kwargs['config'] = config or PostgresRetrievalConfig()
    
    # Create retriever instance
    postgres_retriever = retriever_class(**retriever_kwargs)
    
    # Wrap in adapter
    return PostgresRetrieverAdapter(
        postgres_retriever=postgres_retriever,
        config=config
    )


def create_hybrid_adapter(
    postgres_retriever_class: type,
    local_retriever_adapter: Optional[BaseRetrieverAdapter] = None,
    config: Optional[PostgresRetrievalConfig] = None,
    mode: str = "postgres_first",
    **postgres_kwargs
) -> HybridModeAdapter:
    """
    Factory function to create a hybrid mode adapter.
    
    Args:
        postgres_retriever_class: The PostgreSQL retriever class
        local_retriever_adapter: Optional local retriever adapter
        config: PostgreSQL configuration
        mode: Hybrid mode setting
        **postgres_kwargs: Arguments for PostgreSQL retriever
        
    Returns:
        HybridModeAdapter instance
    """
    # Create PostgreSQL adapter
    postgres_adapter = create_postgres_adapter(
        postgres_retriever_class,
        config,
        **postgres_kwargs
    )
    
    # Create hybrid adapter
    return HybridModeAdapter(
        postgres_adapter=postgres_adapter,
        local_adapter=local_retriever_adapter,
        mode=mode
    )