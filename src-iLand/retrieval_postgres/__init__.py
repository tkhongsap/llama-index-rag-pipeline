"""PostgreSQL-based retrieval module for iLand RAG pipeline.

This module provides PostgreSQL implementations of all retrieval strategies
with complete parity to the local file-based retrieval system.
"""

try:
    from .index_classifier import PostgresIndexClassifier
    from .router import PostgresRouterRetriever
    from .retrievers import (
        PostgresVectorRetriever,
        PostgresHybridRetriever,
        PostgresRecursiveRetriever,
        PostgresChunkDecouplingRetriever,
        PostgresPlannerRetriever,
        PostgresMetadataRetriever,
        PostgresSummaryRetriever
    )
    from .config import PostgresRetrievalConfig
    from .adapters import PostgresRetrieverAdapter, HybridModeAdapter, create_postgres_adapter
    
    __all__ = [
        "PostgresIndexClassifier",
        "PostgresRouterRetriever",
        "PostgresVectorRetriever",
        "PostgresHybridRetriever",
        "PostgresRecursiveRetriever",
        "PostgresChunkDecouplingRetriever",
        "PostgresPlannerRetriever",
        "PostgresMetadataRetriever",
        "PostgresSummaryRetriever",
        "PostgresRetrievalConfig",
        "PostgresRetrieverAdapter",
        "HybridModeAdapter",
        "create_postgres_adapter"
    ]
except ImportError as e:
    print(f"Warning: Some PostgreSQL retrieval components could not be imported: {e}")
    __all__ = []