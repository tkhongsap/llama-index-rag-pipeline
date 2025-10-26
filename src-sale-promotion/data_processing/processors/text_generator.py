"""
Text generator for creating natural language promotion descriptions.

Generates human-readable text (50-200 tokens) for embedding.
"""

import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Generates natural language text descriptions for promotions.
    
    This text will be embedded in Phase 2 for semantic search.
    """
    
    def __init__(self):
        """Initialize text generator"""
        pass
    
    def generate(self, promotion_data: dict) -> str:
        """
        Generate natural language promotion description.
        
        Args:
            promotion_data: Dictionary with all promotion fields
        
        Returns:
            Natural language text (50-200 tokens)
        
        Note:
            This method is a convenience wrapper. The actual text generation
            logic is implemented in PromotionDocument._generate_text() method
            in models.py for better encapsulation.
        """
        # This is handled by PromotionDocument._generate_text()
        # This class exists for extensibility if custom text templates are needed
        logger.debug("Text generation delegated to PromotionDocument._generate_text()")
        return ""

