"""
iLand Document Embedding Module

Production-ready embedding extraction for Thai land deed documents.
Follows LlamaIndex best practices for production RAG applications.
"""

from .metadata_extractor import iLandMetadataExtractor
from .document_loader import iLandDocumentLoader
from .embedding_processor import EmbeddingProcessor
from .file_storage import EmbeddingStorage
from .batch_embedding import (
    iLandBatchEmbeddingPipeline,
    iLandHierarchicalRetriever,
    iLandProductionQueryEngine,
    create_iland_production_query_engine
)
# Import PostgreSQL embedding generator
from .postgres_embedding import PostgresEmbeddingGenerator

__all__ = [
    'iLandMetadataExtractor',
    'iLandDocumentLoader',
    'EmbeddingProcessor',
    'EmbeddingStorage',
    'iLandBatchEmbeddingPipeline',
    'iLandHierarchicalRetriever',
    'iLandProductionQueryEngine',
    'create_iland_production_query_engine',
    'PostgresEmbeddingGenerator'  # Add PostgreSQL embedding generator to exports
] 