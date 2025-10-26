"""
Strategy Selector for Sales Promotion Retrieval

Stage 2 of two-stage routing: Selects optimal retrieval strategy based on query analysis.
Uses GPT-5 to analyze query characteristics and select from 7 available strategies.
"""

import json
from typing import Dict, Any, List
from llama_index.llms.openai import OpenAI


class StrategySelector:
    """
    Selects optimal retrieval strategy using GPT-5 query analysis.
    
    Decision criteria:
    - Contains specific filters → metadata
    - Multi-part question → planner
    - Thai keywords + semantic → hybrid
    - Document overview → summary
    - Deep analysis → recursive
    - Exact answer → chunk_decoupling
    - Default → vector
    """
    
    def __init__(self, llm: OpenAI):
        """
        Initialize strategy selector.
        
        Args:
            llm: LLM for query analysis (GPT-5)
        """
        self.llm = llm
        self.strategies = [
            "vector", "summary", "recursive",
            "metadata", "hybrid", "chunk_decoupling", "planner"
        ]
    
    def select_strategy(
        self,
        query: str,
        index: str = "sales_promotions"
    ) -> Dict[str, Any]:
        """
        Analyze query and select best retrieval strategy using GPT-5.
        
        Args:
            query: User query
            index: Target index (from Stage 1)
            
        Returns:
            Strategy selection result with confidence and reasoning
        """
        prompt = f"""
Analyze this sales promotion query and select the best retrieval strategy.

Query: {query}

Available strategies:
1. **vector**: General semantic search (fast, ~100ms)
   - Use for: Simple similarity queries, "Find beer promotions", "Show water deals"
   
2. **summary**: Document-level retrieval (moderate, ~200ms)
   - Use for: Overview queries, "Which promotions...", "List all..."
   
3. **recursive**: Multi-level hierarchical (thorough, ~300ms)
   - Use for: Deep analysis, "Explain how...", "Compare X across Y"
   
4. **metadata**: Filtered/analytical queries (very fast, ~50ms)
   - Use for: Specific filters (customer_type, discount > X, is_active, etc.)
   - Keywords: "active", "for เอเย่นต์", "with discount > 25%", specific customer/product
   
5. **hybrid**: Thai keywords + semantic (moderate, ~150ms)
   - Use for: Thai product names + similarity, "น้ำดื่มช้าง PET promotions"
   
6. **chunk_decoupling**: Precise answers (moderate, ~200ms)
   - Use for: "What is...", "Exact...", specific detail extraction
   
7. **planner**: Complex multi-part queries (comprehensive, ~500ms)
   - Use for: "Compare", "Analyze", "Calculate", multi-step questions

Selection criteria (priority order):
1. If contains specific metadata filters (customer_type, product_category, discount conditions) → **metadata**
2. If multi-part question (compare/analyze/calculate multiple things) → **planner**
3. If Thai product keywords + semantic intent → **hybrid**
4. If document overview ("which", "list all", "show all") → **summary**
5. If deep analysis needed ("explain how", "deep dive") → **recursive**
6. If asking for exact/specific info ("what is", "exact period") → **chunk_decoupling**
7. Default for general semantic queries → **vector**

Return ONLY valid JSON:
{{
    "strategy": "vector|summary|recursive|metadata|hybrid|chunk_decoupling|planner",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation why this strategy was chosen"
}}

Your response (JSON only):
"""
        
        try:
            response = self.llm.complete(prompt)
            result = json.loads(response.text.strip())
            
            # Validate strategy is in available list
            if result.get("strategy") not in self.strategies:
                print(f"  ⚠️  Invalid strategy '{result.get('strategy')}', defaulting to vector")
                result["strategy"] = "vector"
                result["confidence"] = 0.5
            
            return result
        
        except Exception as e:
            print(f"  ⚠️  Strategy selection failed: {e}, defaulting to vector")
            return {
                "strategy": "vector",
                "confidence": 0.5,
                "reasoning": f"Fallback due to error: {str(e)}"
            }

