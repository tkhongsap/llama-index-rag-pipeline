"""
PostgreSQL Retrieval Package for iLand Thai Land Deed Data

This package implements PostgreSQL-based retrieval strategies that query directly
from the database instead of loading embeddings from files. It provides:

- Seven PostgreSQL-based retrieval strategies
- Router retriever with intelligent strategy selection
- Direct pgVector similarity search
- CLI interface for testing and usage
- Performance logging and caching

Main Components:
- BasePostgresRetriever: Base class for all PostgreSQL retrievers
- PostgresRouterRetriever: Main router with strategy selection
- Seven retrieval strategies: basic, window, recursive, auto_merge, metadata, ensemble, agentic
- PostgresQueryEngine: Query engine wrapper
- CLI: Command-line interface

Database Tables Used:
- iland_chunks: Document chunks with embeddings
- iland_summaries: Document summaries with embeddings  
- iland_indexnodes: Index nodes with embeddings
- iland_combined: All embeddings combined
"""

from .config import PostgresConfig
from .base_retriever import BasePostgresRetriever
from .router import PostgresRouterRetriever
from .query_engines import PostgresQueryEngine
from .retrievers import (
    BasicPostgresRetriever,
    SentenceWindowPostgresRetriever,
    RecursivePostgresRetriever,
    AutoMergePostgresRetriever,
    MetadataFilterPostgresRetriever,
    EnsemblePostgresRetriever,
    AgenticPostgresRetriever
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
    
    # Retrieval strategies
    "BasicPostgresRetriever",
    "SentenceWindowPostgresRetriever",
    "RecursivePostgresRetriever", 
    "AutoMergePostgresRetriever",
    "MetadataFilterPostgresRetriever",
    "EnsemblePostgresRetriever",
    "AgenticPostgresRetriever"
] 