"""
Retrieval Package for iLand Thai Land Deed Data

This package implements an agentic retrieval workflow that mirrors the src/agentic_retriever 
system but adapted for Thai land deed data. It provides:

- Index classification for iLand indices
- Seven retrieval strategy adapters  
- Router retriever with LLM strategy selection
- CLI interface for testing and usage
- Performance logging and statistics

Main Components:
- iLandIndexClassifier: Routes queries to appropriate iLand indices
- iLandRouterRetriever: Main router with strategy selection
- Seven retrieval adapters: vector, summary, recursive, metadata, chunk_decoupling, hybrid, planner
- CLI: Command-line interface for testing and usage
"""

from .index_classifier import iLandIndexClassifier, create_default_iland_classifier
from .router import iLandRouterRetriever
from .retrievers import (
    VectorRetrieverAdapter,
    SummaryRetrieverAdapter, 
    RecursiveRetrieverAdapter,
    MetadataRetrieverAdapter,
    ChunkDecouplingRetrieverAdapter,
    HybridRetrieverAdapter,
    PlannerRetrieverAdapter
)

# Version information
__version__ = "1.0.0"
__author__ = "iLand Team"

# Public API
__all__ = [
    # Main classes
    "iLandIndexClassifier",
    "iLandRouterRetriever",
    
    # Retrieval adapters
    "VectorRetrieverAdapter",
    "SummaryRetrieverAdapter", 
    "RecursiveRetrieverAdapter",
    "MetadataRetrieverAdapter",
    "ChunkDecouplingRetrieverAdapter",
    "HybridRetrieverAdapter",
    "PlannerRetrieverAdapter",
    
    # Factory functions
    "create_default_iland_classifier"
] 