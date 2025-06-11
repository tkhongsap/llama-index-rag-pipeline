"""
Vector operations utilities for PostgreSQL retrieval

Handles embedding generation, similarity calculations, and vector operations.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from openai import OpenAI

logger = logging.getLogger(__name__)


class VectorOperations:
    """Handles vector operations for PostgreSQL retrieval."""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        openai_api_key: Optional[str] = None
    ):
        """Initialize vector operations."""
        self.embedding_model_name = embedding_model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize embedding model
        self._embedding_model = None
        self._openai_client = None
        
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        try:
            if self.embedding_model_name.startswith("text-embedding"):
                # OpenAI embedding model
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key required for OpenAI embedding models")
                self._openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info(f"Loaded OpenAI embedding model: {self.embedding_model_name}")
            else:
                # Sentence Transformers model (e.g., BGE)
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.embedding_model_name}: {e}")
            raise
    
    def get_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Generate embedding for given text.
        
        Args:
            text: Input text
            normalize: Whether to normalize the embedding
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            if self._openai_client:
                # Use OpenAI API
                response = self._openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=text
                )
                embedding = response.data[0].embedding
            else:
                # Use SentenceTransformers
                embedding = self._embedding_model.encode([text])[0]
                embedding = embedding.tolist()
            
            # Normalize if requested
            if normalize and embedding:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = [x / norm for x in embedding]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    def get_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        try:
            if self._openai_client:
                # Process in batches for OpenAI API
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    response = self._openai_client.embeddings.create(
                        model=self.embedding_model_name,
                        input=batch_texts
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
            else:
                # Use SentenceTransformers batch processing
                batch_embeddings = self._embedding_model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=len(texts) > 100
                )
                embeddings = [emb.tolist() for emb in batch_embeddings]
            
            # Normalize if requested
            if normalize:
                normalized_embeddings = []
                for embedding in embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        normalized_embeddings.append([x / norm for x in embedding])
                    else:
                        normalized_embeddings.append(embedding)
                embeddings = normalized_embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def rerank_by_similarity(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        embedding_key: str = "embedding",
        score_key: str = "similarity_score",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates by similarity to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidates: List of candidate dictionaries
            embedding_key: Key for embedding in candidate dict
            score_key: Key to store similarity score
            top_k: Number of top candidates to return
            
        Returns:
            Reranked candidates with similarity scores
        """
        try:
            # Calculate similarities
            for candidate in candidates:
                if embedding_key in candidate:
                    similarity = self.cosine_similarity(
                        query_embedding,
                        candidate[embedding_key]
                    )
                    candidate[score_key] = similarity
                else:
                    candidate[score_key] = 0.0
            
            # Sort by similarity (descending)
            ranked_candidates = sorted(
                candidates,
                key=lambda x: x.get(score_key, 0.0),
                reverse=True
            )
            
            # Return top_k if specified
            if top_k is not None:
                ranked_candidates = ranked_candidates[:top_k]
            
            return ranked_candidates
            
        except Exception as e:
            logger.error(f"Failed to rerank candidates: {e}")
            return candidates
    
    def filter_by_similarity_threshold(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float,
        score_key: str = "similarity_score"
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates by similarity threshold.
        
        Args:
            candidates: List of candidate dictionaries
            threshold: Minimum similarity threshold
            score_key: Key for similarity score in candidate dict
            
        Returns:
            Filtered candidates above threshold
        """
        return [
            candidate for candidate in candidates
            if candidate.get(score_key, 0.0) >= threshold
        ]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        try:
            if self._openai_client:
                # OpenAI embedding dimensions (model-specific)
                if "text-embedding-3-small" in self.embedding_model_name:
                    return 1536
                elif "text-embedding-3-large" in self.embedding_model_name:
                    return 3072
                elif "text-embedding-ada-002" in self.embedding_model_name:
                    return 1536
                else:
                    return 1536  # Default
            else:
                # SentenceTransformers dimension
                return self._embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            return 1024  # Default BGE dimension 