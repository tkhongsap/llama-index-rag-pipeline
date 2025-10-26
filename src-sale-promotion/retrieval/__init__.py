"""
Sales Promotion Retrieval Module

Production-ready retrieval system with 7 specialized strategies for Thai sales promotion queries.
Follows src-imap/retrieval/ architecture with two-stage routing.
"""

from .config import CONFIG, get_config, get_strategy_config, is_strategy_enabled
from .loader import EmbeddingLoader, load_and_build_index
from .fast_metadata_index import FastMetadataIndex
from .retrievers import (
    BaseRetrieverAdapter,
    VectorRetrieverAdapter,
    SummaryRetrieverAdapter,
    MetadataRetrieverAdapter,
    RecursiveRetrieverAdapter,
    HybridRetrieverAdapter,
    ChunkDecouplingRetrieverAdapter,
    PlannerRetrieverAdapter,
)
from .index_classifier import IndexClassifier
from .strategy_selector import StrategySelector
from .cache import QueryCache
from .router import PromotionRouter
from .query_engine import PromotionQueryEngine

__all__ = [
    "CONFIG",
    "get_config",
    "get_strategy_config",
    "is_strategy_enabled",
    "EmbeddingLoader",
    "load_and_build_index",
    "FastMetadataIndex",
    "BaseRetrieverAdapter",
    "VectorRetrieverAdapter",
    "SummaryRetrieverAdapter",
    "MetadataRetrieverAdapter",
    "RecursiveRetrieverAdapter",
    "HybridRetrieverAdapter",
    "ChunkDecouplingRetrieverAdapter",
    "PlannerRetrieverAdapter",
    "IndexClassifier",
    "StrategySelector",
    "QueryCache",
    "PromotionRouter",
    "PromotionQueryEngine",
]
