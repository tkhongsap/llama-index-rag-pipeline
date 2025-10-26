"""
Retrieval Module

Implements agentic retrieval with 7 strategies and two-stage routing
for sales promotion documents.
"""

from .router import SalesPromotionRouterRetriever
from .index_classifier import SalesPromotionIndexClassifier

__all__ = ["SalesPromotionRouterRetriever", "SalesPromotionIndexClassifier"]
