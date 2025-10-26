"""
Retrieval Strategies

Seven retrieval strategies for sales promotion documents.
"""

from .vector import VectorRetrieverAdapter
from .hybrid import HybridRetrieverAdapter
from .metadata import MetadataRetrieverAdapter
from .recursive import RecursiveRetrieverAdapter
from .summary import SummaryRetrieverAdapter
from .chunk_decoupling import ChunkDecouplingRetrieverAdapter
from .planner import PlannerRetrieverAdapter

__all__ = [
    "VectorRetrieverAdapter",
    "HybridRetrieverAdapter",
    "MetadataRetrieverAdapter",
    "RecursiveRetrieverAdapter",
    "SummaryRetrieverAdapter",
    "ChunkDecouplingRetrieverAdapter",
    "PlannerRetrieverAdapter",
]
