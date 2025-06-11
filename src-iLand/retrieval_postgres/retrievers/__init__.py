"""
PostgreSQL-based retrieval strategies for iLand data

This module contains retrieval strategies that query directly
from PostgreSQL/pgVector instead of loading embeddings from files.

Currently implemented strategies:
- BasicPostgresRetriever: Direct vector similarity search
- SentenceWindowPostgresRetriever: Context expansion with surrounding chunks  
- RecursivePostgresRetriever: Hierarchical summary→chunk search
- MetadataFilterPostgresRetriever: Thai geographic/legal metadata filtering

TODO: Implement remaining strategies:
- AutoMergePostgresRetriever: Automatic chunk merging
- EnsemblePostgresRetriever: Multiple strategy combination
- AgenticPostgresRetriever: LLM-guided retrieval
"""

from .basic_postgres import BasicPostgresRetriever
from .sentence_window_postgres import SentenceWindowPostgresRetriever
from .recursive_postgres import RecursivePostgresRetriever
from .auto_merge_postgres import AutoMergePostgresRetriever
from .metadata_filter_postgres import MetadataFilterPostgresRetriever
from .ensemble_postgres import EnsemblePostgresRetriever
from .agentic_postgres import AgenticPostgresRetriever

__all__ = [
    "BasicPostgresRetriever",
    "SentenceWindowPostgresRetriever", 
    "RecursivePostgresRetriever",
    "AutoMergePostgresRetriever",
    "MetadataFilterPostgresRetriever",
    "EnsemblePostgresRetriever",
    "AgenticPostgresRetriever"
] 