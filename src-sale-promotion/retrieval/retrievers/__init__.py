"""
Retrieval Strategy Adapters for Sales Promotion Data

This module contains adapters for all seven retrieval strategies:
- vector: Basic vector similarity search
- summary: Document summary-first retrieval
- recursive: Recursive retrieval with hierarchical relationships
- metadata: Metadata-filtered retrieval for sales promotion attributes
- hybrid: Hybrid search combining vector and keyword for Thai content
- chunk_decoupling: Chunk decoupling strategy (adapted for atomic docs)
- planner: Query planning agent for complex analytical queries
"""

from .base import BaseRetrieverAdapter
from .vector import VectorRetrieverAdapter
from .summary import SummaryRetrieverAdapter
from .metadata import MetadataRetrieverAdapter
from .recursive import RecursiveRetrieverAdapter
from .hybrid import HybridRetrieverAdapter
from .chunk_decoupling import ChunkDecouplingRetrieverAdapter
from .planner import PlannerRetrieverAdapter

__all__ = [
    "BaseRetrieverAdapter",
    "VectorRetrieverAdapter",
    "SummaryRetrieverAdapter",
    "MetadataRetrieverAdapter",
    "RecursiveRetrieverAdapter",
    "HybridRetrieverAdapter",
    "ChunkDecouplingRetrieverAdapter",
    "PlannerRetrieverAdapter",
]
