"""
Data Processing Module

Handles CSV/document conversion to structured markdown format
with sales promotion-specific metadata extraction.
"""

from .models import SalesPromotionDocument
from .document_processor import SalesPromotionProcessor

__all__ = ["SalesPromotionDocument", "SalesPromotionProcessor"]
