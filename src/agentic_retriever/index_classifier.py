"""
Index Classifier

Chooses the right index (finance docs, slides, etc.) for every question.
Supports both LLM-based classification and embedding-based fallback.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
import numpy as np

# Load environment variables
load_dotenv(override=True)


class IndexClassifier:
    """Classifies queries to determine the most appropriate index to use."""
    
    def __init__(
        self,
        available_indices: Dict[str, str],
        api_key: Optional[str] = None,
        mode: str = "llm"
    ):
        """
        Initialize the index classifier.
        
        Args:
            available_indices: Dict mapping index names to descriptions
            api_key: OpenAI API key
            mode: Classification mode - "llm" or "embedding"
        """
        self.available_indices = available_indices
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.mode = mode.lower()
        
        if self.mode not in ["llm", "embedding"]:
            raise ValueError("Mode must be 'llm' or 'embedding'")
        
        self._setup_models()
        
        if self.mode == "embedding":
            self._setup_embedding_classifier()
    
    def _setup_models(self):
        """Setup LLM and embedding models."""
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=self.api_key
        )
    
    def _setup_embedding_classifier(self):
        """Setup embedding-based classifier with index descriptions."""
        # Create embeddings for each index description
        self.index_embeddings = {}
        
        for index_name, description in self.available_indices.items():
            embedding = Settings.embed_model.get_text_embedding(description)
            self.index_embeddings[index_name] = np.array(embedding)
    
    def classify_query_llm(self, query: str) -> Dict[str, Any]:
        """
        Classify query using LLM.
        
        Args:
            query: The user query
            
        Returns:
            Dict with selected index and confidence
        """
        # Create prompt for index classification
        index_descriptions = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in self.available_indices.items()
        ])
        
        prompt = f"""
You are an expert at routing queries to the most appropriate data source.

Available indices:
{index_descriptions}

Query: "{query}"

Based on the query, which index would be most appropriate? 
Respond with ONLY the index name (exactly as listed above).
If unsure, choose the most general index.
"""
        
        try:
            response = Settings.llm.complete(prompt)
            selected_index = response.text.strip()
            
            # Validate the response
            if selected_index in self.available_indices:
                return {
                    "selected_index": selected_index,
                    "confidence": 0.9,  # High confidence for LLM
                    "method": "llm",
                    "reasoning": f"LLM selected {selected_index} for query"
                }
            else:
                # Fallback to first available index if LLM response is invalid
                fallback_index = list(self.available_indices.keys())[0]
                return {
                    "selected_index": fallback_index,
                    "confidence": 0.3,  # Low confidence for fallback
                    "method": "llm_fallback",
                    "reasoning": f"LLM response '{selected_index}' invalid, using fallback"
                }
                
        except Exception as e:
            # Fallback to embedding method on LLM error
            return self.classify_query_embedding(query)
    
    def classify_query_embedding(self, query: str) -> Dict[str, Any]:
        """
        Classify query using embedding similarity.
        
        Args:
            query: The user query
            
        Returns:
            Dict with selected index and confidence
        """
        try:
            # Get query embedding
            query_embedding = np.array(Settings.embed_model.get_text_embedding(query))
            
            # Calculate similarities with each index
            similarities = {}
            for index_name, index_embedding in self.index_embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, index_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(index_embedding)
                )
                similarities[index_name] = similarity
            
            # Select index with highest similarity
            selected_index = max(similarities, key=similarities.get)
            confidence = similarities[selected_index]
            
            return {
                "selected_index": selected_index,
                "confidence": float(confidence),
                "method": "embedding",
                "reasoning": f"Embedding similarity: {confidence:.3f}",
                "all_similarities": {k: float(v) for k, v in similarities.items()}
            }
            
        except Exception as e:
            # Final fallback to first available index
            fallback_index = list(self.available_indices.keys())[0]
            return {
                "selected_index": fallback_index,
                "confidence": 0.1,  # Very low confidence
                "method": "error_fallback",
                "reasoning": f"Error in embedding classification: {str(e)}"
            }
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query using the configured mode.
        
        Args:
            query: The user query
            
        Returns:
            Dict with selected index and metadata
        """
        if self.mode == "llm":
            return self.classify_query_llm(query)
        else:
            return self.classify_query_embedding(query)
    
    def add_index(self, name: str, description: str):
        """Add a new index to the classifier."""
        self.available_indices[name] = description
        
        if self.mode == "embedding":
            # Update embedding classifier
            embedding = Settings.embed_model.get_text_embedding(description)
            self.index_embeddings[name] = np.array(embedding)
    
    def remove_index(self, name: str):
        """Remove an index from the classifier."""
        if name in self.available_indices:
            del self.available_indices[name]
            
            if self.mode == "embedding" and name in self.index_embeddings:
                del self.index_embeddings[name]
    
    def get_available_indices(self) -> Dict[str, str]:
        """Get all available indices and their descriptions."""
        return self.available_indices.copy()
    
    def set_mode(self, mode: str):
        """Change the classification mode."""
        if mode.lower() not in ["llm", "embedding"]:
            raise ValueError("Mode must be 'llm' or 'embedding'")
        
        self.mode = mode.lower()
        
        if self.mode == "embedding" and not hasattr(self, 'index_embeddings'):
            self._setup_embedding_classifier()


# Default index configurations
DEFAULT_INDICES = {
    "finance_docs": "Financial documents, earnings reports, revenue data, financial statements, and business metrics",
    "general_docs": "General documents, presentations, reports, and miscellaneous content",
    "technical_docs": "Technical documentation, API references, code documentation, and technical specifications"
}


def create_default_classifier(
    api_key: Optional[str] = None,
    mode: Optional[str] = None
) -> IndexClassifier:
    """
    Create a classifier with default index configurations.
    
    Args:
        api_key: OpenAI API key
        mode: Classification mode (defaults to env var CLASSIFIER_MODE or "llm")
        
    Returns:
        IndexClassifier instance
    """
    # Get mode from environment variable or default to "llm"
    if mode is None:
        mode = os.getenv("CLASSIFIER_MODE", "llm")
    
    return IndexClassifier(
        available_indices=DEFAULT_INDICES,
        api_key=api_key,
        mode=mode
    ) 