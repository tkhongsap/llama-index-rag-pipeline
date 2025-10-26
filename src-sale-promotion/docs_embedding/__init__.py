"""
Sales Promotion Document Embedding Module

Production-ready embedding extraction for Thai sales promotion data.
Follows LlamaIndex best practices for production RAG applications.

Following PRD: 02_PRD_PHASE2_EMBEDDING_INDEXING.md
"""

from .document_loader import PromotionDocumentLoader
from .embedding_processor import EmbeddingProcessor
from .file_storage import EmbeddingStorage
from .batch_embedding import SalesPromotionBatchEmbeddingPipeline

__all__ = [
    'PromotionDocumentLoader',
    'EmbeddingProcessor',
    'EmbeddingStorage',
    'SalesPromotionBatchEmbeddingPipeline'
]
