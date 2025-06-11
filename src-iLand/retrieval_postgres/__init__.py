"""
PostgreSQL Retrieval Package for iLand Thai Land Deed Data

This package implements PostgreSQL-based retrieval strategies that query directly
from the database instead of loading embeddings from files. It provides:

- Four implemented PostgreSQL-based retrieval strategies
- Router retriever with intelligent strategy selection
- Direct pgVector similarity search
- CLI interface for testing and usage
- Performance logging and caching

Main Components:
- BasePostgresRetriever: Base class for all PostgreSQL retrievers
- PostgresRouterRetriever: Main router with strategy selection
- Four retrieval strategies: basic, sentence_window, recursive, metadata_filter
- PostgresQueryEngine: Query engine wrapper
- CLI: Command-line interface

Database Tables Used:
- iland_chunks: Document chunks with embeddings
- iland_summaries: Document summaries with embeddings  
- iland_indexnodes: Index nodes with embeddings
- iland_combined: All embeddings combined

TODO: Implement remaining strategies:
- AutoMergePostgresRetriever: Automatic chunk merging
- EnsemblePostgresRetriever: Multiple strategy combination  
- AgenticPostgresRetriever: LLM-guided retrieval
"""

from .config import PostgresConfig
from .base_retriever import BasePostgresRetriever
from .router import PostgresRouterRetriever
from .query_engines import PostgresQueryEngine
from .retrievers import (
    BasicPostgresRetriever,
    SentenceWindowPostgresRetriever,
    RecursivePostgresRetriever,
    MetadataFilterPostgresRetriever
)

# Version information
__version__ = "1.0.0"
__author__ = "iLand Team"

# Public API
__all__ = [
    # Configuration
    "PostgresConfig",
    
    # Base classes
    "BasePostgresRetriever",
    "PostgresRouterRetriever", 
    "PostgresQueryEngine",
    
    # Implemented retrieval strategies
    "BasicPostgresRetriever",
    "SentenceWindowPostgresRetriever",
    "RecursivePostgresRetriever",
    "MetadataFilterPostgresRetriever"
] 