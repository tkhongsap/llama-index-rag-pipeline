"""
Production Query Engine for Sales Promotions

Complete query-to-answer pipeline with GPT-5 response synthesis.
Integrates routing, retrieval, and response generation.
"""

import time
import json
from typing import Dict, Any, List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from .router import PromotionRouter


class PromotionQueryEngine:
    """
    Production-ready query engine for sales promotions.
    
    Features:
    - Two-stage routing (Index → Strategy)
    - 7 retrieval strategies
    - GPT-5 response synthesis
    - Query caching
    - Performance tracking
    - Thai language support
    """
    
    def __init__(
        self,
        router: PromotionRouter,
        llm: OpenAI,
        response_mode: str = "compact",
        verbose: bool = False
    ):
        """
        Initialize query engine.
        
        Args:
            router: Two-stage router
            llm: LLM for response synthesis (GPT-5)
            response_mode: Response synthesis mode
            verbose: Enable verbose output
        """
        self.router = router
        self.llm = llm
        self.response_mode = response_mode
        self.verbose = verbose
    
    def query(
        self,
        query_str: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        strategy_override: Optional[str] = None,
        return_nodes: bool = False
    ) -> Dict[str, Any]:
        """
        Execute query and return formatted response.
        
        Args:
            query_str: User query (supports Thai)
            top_k: Number of results to retrieve
            filters: Optional metadata filters
            strategy_override: Manually specify strategy
            return_nodes: Include retrieved nodes in response
            
        Returns:
            Dictionary with response, metadata, and optional nodes
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"QUERY: {query_str}")
            print(f"{'='*80}")
        
        # Route and retrieve
        nodes = self.router.route(query_str, top_k, filters, strategy_override)
        
        retrieval_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n📊 Retrieved {len(nodes)} nodes in {retrieval_time:.3f}s")
            if nodes:
                print(f"   Strategy: {nodes[0].node.metadata.get('selected_strategy', 'unknown')}")
        
        # Synthesize response
        response_text = self._synthesize_response(query_str, nodes)
        
        synthesis_time = time.time() - start_time - retrieval_time
        total_time = time.time() - start_time
        
        # Build result dictionary
        result = {
            "query": query_str,
            "response": response_text,
            "metadata": {
                "num_results": len(nodes),
                "selected_strategy": nodes[0].node.metadata.get("selected_strategy") if nodes else None,
                "strategy_confidence": nodes[0].node.metadata.get("strategy_confidence") if nodes else None,
                "cache_hit": nodes[0].node.metadata.get("cache_hit", False) if nodes else False,
                "latency": {
                    "retrieval_seconds": round(retrieval_time, 3),
                    "synthesis_seconds": round(synthesis_time, 3),
                    "total_seconds": round(total_time, 3)
                },
                "top_scores": [round(node.score, 3) for node in nodes[:3]]
            }
        }
        
        # Optionally include nodes
        if return_nodes:
            result["nodes"] = nodes
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RESPONSE")
            print(f"{'='*80}")
            print(response_text)
            print(f"\n⏱️  Total time: {total_time:.3f}s")
        
        return result
    
    def _synthesize_response(
        self,
        query: str,
        nodes: List[NodeWithScore]
    ) -> str:
        """
        Generate natural language response using GPT-5.
        
        Args:
            query: User query
            nodes: Retrieved nodes
            
        Returns:
            Natural language response
        """
        if not nodes:
            return "ไม่พบโปรโมชันที่ตรงกับเงื่อนไข (No promotions found matching your criteria.)"
        
        # Build context from retrieved nodes
        context_parts = []
        for i, node in enumerate(nodes[:5], 1):
            metadata = node.node.metadata
            
            # Extract key info
            doc_id = metadata.get('document_id', 'N/A')
            customer_type = metadata.get('customer_type', 'N/A')
            product_name = metadata.get('product_name', 'N/A')
            discount = metadata.get('discount_percentage', 'N/A')
            is_active = metadata.get('is_active', False)
            
            context_parts.append(
                f"Promotion {i} (ID: {doc_id}):\n"
                f"Customer Type: {customer_type}\n"
                f"Product: {product_name}\n"
                f"Discount: {discount}%\n"
                f"Active: {is_active}\n"
                f"Details: {node.node.text[:200]}...\n"
            )
        
        context = "\n".join(context_parts)
        
        # Determine response language (Thai if query contains Thai, else English)
        has_thai = any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in query)
        language = "Thai" if has_thai else "English"
        
        # Generate response
        prompt = f"""
Based on the following sales promotion information, answer the user's query clearly and concisely.

Query: {query}

Promotion Information:
{context}

Instructions:
- Answer in {language}
- Be specific and include key details (discount percentages, product names, customer types, dates)
- If comparing or analyzing, provide clear comparison
- If asked for list, format as numbered list
- Include relevant promotion IDs for reference
- Keep response concise but informative

Your response:
"""
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.router.get_cache_stats()
    
    def clear_cache(self):
        """Clear query cache."""
        self.router.clear_cache()

