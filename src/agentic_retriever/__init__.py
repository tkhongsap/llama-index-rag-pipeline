"""
Agentic Retrieval Layer - v1.3

This package provides an intelligent retrieval layer that:
1. Chooses the right index (finance docs, slides, etc.) for every question
2. Chooses the right retrieval strategy (vector, summary-first, recursive, etc.)

All components are designed to be stateless and scalable.
"""

from .router import RouterRetriever
from .index_classifier import IndexClassifier
from .log_utils import log_retrieval_call

__version__ = "1.3.0"
__all__ = [
    "RouterRetriever",
    "IndexClassifier", 
    "log_retrieval_call"
] 