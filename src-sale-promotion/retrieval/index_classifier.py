"""
Index Classifier for Sales Promotion Retrieval

Stage 1 of two-stage routing: Determines which index to search.
For Phase 3: Single index (sales_promotions), but designed for future multi-index expansion.
"""

from typing import Dict, Any
from llama_index.llms.openai import OpenAI


class IndexClassifier:
    """
    Classifies queries to determine which index to search.
    
    Phase 3: Single index (sales_promotions)
    Future expansion: products, customers, performance_metrics
    """
    
    def __init__(self, llm: OpenAI = None):
        """
        Initialize index classifier.
        
        Args:
            llm: LLM for future multi-index classification (GPT-5)
        """
        self.llm = llm
        self.available_indices = ["sales_promotions"]
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query to determine target index.
        
        Args:
            query: User query
            
        Returns:
            Classification result with selected index and confidence
        """
        # Phase 3: Always return sales_promotions (single index)
        return {
            "selected_index": "sales_promotions",
            "confidence": 1.0,
            "reasoning": "Single index available in Phase 3",
            "query": query
        }

