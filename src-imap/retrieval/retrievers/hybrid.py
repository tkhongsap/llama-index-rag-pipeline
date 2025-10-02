"""
Hybrid Retriever Adapter for iLand Data

Implements hybrid search combining vector similarity and keyword matching for Thai land deed data.
Adapted from src/agentic_retriever/retrievers/hybrid.py for iLand data.
"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from .base import BaseRetrieverAdapter

# Import from updated iLand embedding loading modules
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from load_embedding import iLandIndexReconstructor, EmbeddingConfig
except ImportError as e:
    print(f"Warning: Could not import iLand embedding utilities: {e}")
    iLandIndexReconstructor = None
    EmbeddingConfig = None


class HybridRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for hybrid search retrieval on iLand data."""
    
    def __init__(self, 
                 vector_index: VectorStoreIndex,
                 default_top_k: int = 5,
                 alpha: float = 0.7):
        """
        Initialize hybrid retriever adapter for iLand data.
        
        Args:
            vector_index: Vector store index for semantic search
            default_top_k: Default number of nodes to retrieve
            alpha: Weight for vector search vs keyword search (0.0-1.0)
        """
        super().__init__("hybrid")
        self.vector_index = vector_index
        self.default_top_k = default_top_k
        self.alpha = alpha
        
        # Create vector retriever
        self.vector_retriever = vector_index.as_retriever(similarity_top_k=default_top_k * 2)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid search approach on iLand data.
        
        Args:
            query: The search query (may contain Thai text)
            top_k: Number of nodes to retrieve (uses default if None)
            
        Returns:
            List of NodeWithScore objects tagged with strategy
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Step 1: Get semantic search results
        vector_nodes = self.vector_retriever.retrieve(query)
        
        # Step 2: Perform keyword-based scoring for Thai content
        keyword_scored_nodes = self._apply_thai_keyword_scoring(vector_nodes, query)
        
        # Step 3: Combine scores using alpha weighting
        hybrid_nodes = self._combine_scores(vector_nodes, keyword_scored_nodes, self.alpha)
        
        # Step 4: Sort by combined score and take top_k
        hybrid_nodes.sort(key=lambda x: x.score, reverse=True)
        final_nodes = hybrid_nodes[:k]
        
        # Tag nodes with strategy
        return self._tag_nodes_with_strategy(final_nodes)
    
    def _apply_thai_keyword_scoring(self, nodes: List[NodeWithScore], 
                                   query: str) -> List[NodeWithScore]:
        """
        Apply keyword-based scoring for Thai text content.
        
        Args:
            nodes: Vector search results
            query: Search query
            
        Returns:
            Nodes with keyword-based scores
        """
        # Extract Thai keywords from query
        thai_keywords = self._extract_thai_keywords(query)
        
        keyword_nodes = []
        for node in nodes:
            text = node.node.text.lower()
            keyword_score = 0.0
            
            # Count keyword matches (simple approach)
            for keyword in thai_keywords:
                if keyword in text:
                    # Weight by keyword frequency
                    keyword_score += text.count(keyword) * 0.1
            
            # Normalize keyword score
            if thai_keywords:
                keyword_score = min(keyword_score / len(thai_keywords), 1.0)
            
            # Create new node with keyword score
            new_node = NodeWithScore(node=node.node, score=keyword_score)
            keyword_nodes.append(new_node)
        
        return keyword_nodes
    
    def _extract_thai_keywords(self, query: str) -> List[str]:
        """
        Extract Thai keywords from query.
        
        Args:
            query: Search query
            
        Returns:
            List of Thai keywords
        """
        # Simple tokenization for Thai text
        # In production, consider using pythainlp for proper Thai tokenization
        keywords = []
        
        # Split by spaces and common punctuation
        terms = query.lower().replace(',', ' ').replace('.', ' ').split()
        
        # Filter for Thai terms (contains Thai characters)
        for term in terms:
            if any('\u0e00' <= char <= '\u0e7f' for char in term):  # Thai Unicode range
                keywords.append(term)
            elif len(term) > 2:  # Include longer English terms
                keywords.append(term)
        
        # Add important Thai land deed terms
        land_terms = ["ที่ดิน", "โฉนด", "นส3", "นส4", "ส.ค.1", "อำเภอ", "ตำบล", "จังหวัด"]
        for term in land_terms:
            if term in query.lower():
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    def _combine_scores(self, vector_nodes: List[NodeWithScore], 
                       keyword_nodes: List[NodeWithScore], 
                       alpha: float) -> List[NodeWithScore]:
        """
        Combine vector and keyword scores using alpha weighting.
        
        Args:
            vector_nodes: Nodes with vector similarity scores
            keyword_nodes: Nodes with keyword scores
            alpha: Weight for vector scores (1-alpha for keyword scores)
            
        Returns:
            Nodes with combined scores
        """
        combined_nodes = []
        
        for i, vector_node in enumerate(vector_nodes):
            if i < len(keyword_nodes):
                keyword_score = keyword_nodes[i].score
            else:
                keyword_score = 0.0
            
            # Combine scores: alpha * vector_score + (1-alpha) * keyword_score
            combined_score = alpha * vector_node.score + (1 - alpha) * keyword_score
            
            # Create new node with combined score
            combined_node = NodeWithScore(node=vector_node.node, score=combined_score)
            combined_nodes.append(combined_node)
        
        return combined_nodes
    
    @classmethod
    def from_iland_embeddings(
        cls,
        embeddings: List[dict],
        api_key: Optional[str] = None,
        default_top_k: int = 5,
        alpha: float = 0.7
    ) -> "HybridRetrieverAdapter":
        """
        Create adapter from iLand embeddings.
        
        Args:
            embeddings: List of iLand embedding dictionaries
            api_key: OpenAI API key
            default_top_k: Default number of nodes to retrieve
            alpha: Weight for vector vs keyword search
            
        Returns:
            HybridRetrieverAdapter instance for iLand data
        """
        if not iLandIndexReconstructor or not EmbeddingConfig:
            raise ImportError("iLand embedding utilities not available")
        
        config = EmbeddingConfig(api_key=api_key)
        reconstructor = iLandIndexReconstructor(config=config)
        vector_index = reconstructor.create_vector_index_from_embeddings(
            embeddings, show_progress=False
        )
        
        return cls(vector_index, default_top_k, alpha) 