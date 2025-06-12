"""
Retrieval Strategy Adapters for iLand Data

This module contains adapters for all seven retrieval strategies adapted for Thai land deed data:
- vector: Basic vector similarity search
- summary: Document summary-first retrieval  
- recursive: Recursive retrieval with parent-child relationships
- metadata: Metadata-filtered retrieval for Thai land deed attributes
- chunk_decoupling: Chunk decoupling strategy
- hybrid: Hybrid search combining vector and keyword for Thai content
- planner: Query planning agent for complex land deed queries
"""

from .vector import VectorRetrieverAdapter
from .summary import SummaryRetrieverAdapter
from .recursive import RecursiveRetrieverAdapter
from .metadata import MetadataRetrieverAdapter
from .chunk_decoupling import ChunkDecouplingRetrieverAdapter
from .hybrid import HybridRetrieverAdapter
from .planner import PlannerRetrieverAdapter

__all__ = [
    "VectorRetrieverAdapter",
    "SummaryRetrieverAdapter", 
    "RecursiveRetrieverAdapter",
    "MetadataRetrieverAdapter",
    "ChunkDecouplingRetrieverAdapter",
    "HybridRetrieverAdapter",
    "PlannerRetrieverAdapter"
] 