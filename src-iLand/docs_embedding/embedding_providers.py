"""
Multi-model embedding provider system for iLand RAG Pipeline.
Supports BGE-M3 as default with OpenAI as fallback, following PRD specifications.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from dataclasses import dataclass

# Third-party imports (optional for graceful fallback)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    BGE_M3_AVAILABLE = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModel = None
    SentenceTransformer = None
    BGE_M3_AVAILABLE = False

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAIEmbedding = None
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetrics:
    """Track embedding performance metrics."""
    provider_usage: Dict[str, int]
    embedding_times: Dict[str, List[float]]
    fallback_count: int
    error_count: Dict[str, int]
    
    def __init__(self):
        self.provider_usage = defaultdict(int)
        self.embedding_times = defaultdict(list)
        self.fallback_count = 0
        self.error_count = defaultdict(int)
    
    def log_embedding(self, provider: str, batch_size: int, duration: float):
        """Log embedding operation metrics."""
        self.provider_usage[provider] += batch_size
        self.embedding_times[provider].append(duration)
    
    def log_fallback(self):
        """Log fallback usage."""
        self.fallback_count += 1
    
    def log_error(self, provider: str):
        """Log provider errors."""
        self.error_count[provider] += 1
    
    def generate_report(self) -> Dict:
        """Generate performance metrics report."""
        report = {
            "provider_usage": dict(self.provider_usage),
            "fallback_count": self.fallback_count,
            "error_count": dict(self.error_count),
            "average_times": {}
        }
        
        for provider, times in self.embedding_times.items():
            if times:
                report["average_times"][provider] = sum(times) / len(times)
        
        return report


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def initialize(self, config: Dict) -> None:
        """Initialize the embedding provider with configuration."""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model capabilities and metadata."""
        pass
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming embeddings."""
        return False


class BGE_M3Provider(EmbeddingProvider):
    """BGE-M3 embedding provider for local processing."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self.model_name = "BAAI/bge-m3"
        self.max_length = 8192
        self.embedding_dim = 1024
    
    def initialize(self, config: Dict) -> None:
        """Initialize BGE-M3 model with configuration."""
        if not BGE_M3_AVAILABLE:
            raise RuntimeError(
                "BGE-M3 dependencies not available. Install with: "
                "pip install transformers sentence-transformers torch FlagEmbedding"
            )
        
        self.config = config
        
        try:
            # Determine device
            if config.get("device") == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.get("device", "cpu")
            
            logger.info(f"🔄 Initializing BGE-M3 on device: {self.device}")
            
            # Load model using SentenceTransformer for easier handling
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=config.get("trust_remote_code", True)
            )
            
            # Set max sequence length
            self.model.max_seq_length = config.get("max_length", self.max_length)
            
            # Verify embedding dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"✅ BGE-M3 initialized successfully")
            logger.info(f"   • Device: {self.device}")
            logger.info(f"   • Embedding dim: {self.embedding_dim}")
            logger.info(f"   • Max length: {self.model.max_seq_length}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize BGE-M3: {str(e)}")
            raise RuntimeError(f"BGE-M3 initialization failed: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using BGE-M3."""
        if not self.model:
            raise RuntimeError("BGE-M3 model not initialized")
        
        try:
            start_time = time.time()
            
            # Handle batch processing
            batch_size = self.config.get("batch_size", 32)
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings
                embeddings = self.model.encode(
                    batch_texts,
                    normalize_embeddings=self.config.get("normalize", True),
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                
                # Convert to list format
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
                
                batch_embeddings = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) 
                                  for emb in embeddings]
                all_embeddings.extend(batch_embeddings)
            
            duration = time.time() - start_time
            logger.info(f"✅ BGE-M3 embedded {len(texts)} documents in {duration:.2f}s")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ BGE-M3 embedding failed: {str(e)}")
            raise RuntimeError(f"BGE-M3 embedding error: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using BGE-M3."""
        if not self.model:
            raise RuntimeError("BGE-M3 model not initialized")
        
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.config.get("normalize", True),
                convert_to_tensor=False
            )
            
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
        except Exception as e:
            logger.error(f"❌ BGE-M3 query embedding failed: {str(e)}")
            raise RuntimeError(f"BGE-M3 query embedding error: {str(e)}")
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of BGE-M3 embeddings."""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return BGE-M3 model information."""
        return {
            "provider": "BGE_M3",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "device": self.device,
            "local_model": True,
            "multilingual": True,
            "supports_thai": True
        }


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider for cloud processing."""
    
    def __init__(self):
        self.embed_model = None
        self.config = None
        self.model_name = "text-embedding-3-small"
        self.embedding_dim = 1536
    
    def initialize(self, config: Dict) -> None:
        """Initialize OpenAI embedding model with configuration."""
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI dependencies not available. Install with: "
                "pip install llama-index-embeddings-openai"
            )
        
        self.config = config
        
        try:
            # Get API key from environment
            api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
            if not api_key:
                raise RuntimeError(f"OpenAI API key not found in environment variable: {config.get('api_key_env', 'OPENAI_API_KEY')}")
            
            # Initialize OpenAI embedding model
            self.model_name = config.get("model_name", "text-embedding-3-small")
            self.embed_model = OpenAIEmbedding(
                model=self.model_name,
                api_key=api_key
            )
            
            # Set embedding dimension based on model
            if "text-embedding-3-small" in self.model_name:
                self.embedding_dim = 1536
            elif "text-embedding-3-large" in self.model_name:
                self.embedding_dim = 3072
            else:
                self.embedding_dim = 1536  # Default
            
            logger.info(f"✅ OpenAI provider initialized")
            logger.info(f"   • Model: {self.model_name}")
            logger.info(f"   • Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI provider: {str(e)}")
            raise RuntimeError(f"OpenAI initialization failed: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using OpenAI."""
        if not self.embed_model:
            raise RuntimeError("OpenAI model not initialized")
        
        try:
            start_time = time.time()
            
            # Handle batch processing
            batch_size = self.config.get("batch_size", 20)
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Add retry logic
                retry_attempts = self.config.get("retry_attempts", 3)
                for attempt in range(retry_attempts):
                    try:
                        batch_embeddings = [
                            self.embed_model.get_text_embedding(text) 
                            for text in batch_texts
                        ]
                        all_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        if attempt == retry_attempts - 1:
                            raise e
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            duration = time.time() - start_time
            logger.info(f"✅ OpenAI embedded {len(texts)} documents in {duration:.2f}s")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ OpenAI embedding failed: {str(e)}")
            raise RuntimeError(f"OpenAI embedding error: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using OpenAI."""
        if not self.embed_model:
            raise RuntimeError("OpenAI model not initialized")
        
        try:
            return self.embed_model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"❌ OpenAI query embedding failed: {str(e)}")
            raise RuntimeError(f"OpenAI query embedding error: {str(e)}")
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of OpenAI embeddings."""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return OpenAI model information."""
        return {
            "provider": "OPENAI",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_length": 8191,
            "device": "cloud",
            "local_model": False,
            "multilingual": True,
            "supports_thai": True
        }


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    @staticmethod
    def create_provider(provider_type: str, config: Dict) -> EmbeddingProvider:
        """Create appropriate embedding provider instance."""
        provider_type = provider_type.upper()
        
        if provider_type == "BGE_M3":
            return BGE_M3Provider()
        elif provider_type == "OPENAI":
            return OpenAIProvider()
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")


class EmbeddingManager:
    """Manages multiple embedding providers with fallback support."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.primary_provider = None
        self.fallback_providers = []
        self.metrics = EmbeddingMetrics()
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize primary and fallback providers."""
        default_provider = self.config.get("default_provider", "BGE_M3")
        fallback_order = self.config.get("fallback_order", ["BGE_M3", "OPENAI"])
        fallback_enabled = self.config.get("fallback_enabled", True)
        
        # Initialize primary provider
        try:
            primary_config = self.config["providers"][default_provider]
            self.primary_provider = EmbeddingProviderFactory.create_provider(
                default_provider, primary_config
            )
            self.primary_provider.initialize(primary_config)
            logger.info(f"✅ Primary provider initialized: {default_provider}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize primary provider {default_provider}: {str(e)}")
            if not fallback_enabled:
                raise e
        
        # Initialize fallback providers
        if fallback_enabled:
            for provider_name in fallback_order:
                if provider_name != default_provider:
                    try:
                        provider_config = self.config["providers"][provider_name]
                        provider = EmbeddingProviderFactory.create_provider(
                            provider_name, provider_config
                        )
                        provider.initialize(provider_config)
                        self.fallback_providers.append((provider_name, provider))
                        logger.info(f"✅ Fallback provider initialized: {provider_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to initialize fallback provider {provider_name}: {str(e)}")
    
    def embed_documents_with_fallback(self, texts: List[str]) -> List[List[float]]:
        """Attempt embedding with primary provider, fall back if needed."""
        # Try primary provider first
        if self.primary_provider:
            try:
                start_time = time.time()
                embeddings = self.primary_provider.embed_documents(texts)
                duration = time.time() - start_time
                
                provider_name = self.primary_provider.get_model_info()["provider"]
                self.metrics.log_embedding(provider_name, len(texts), duration)
                
                return embeddings
            except Exception as e:
                logger.warning(f"⚠️ Primary provider failed: {str(e)}")
                self.metrics.log_error("primary")
        
        # Try fallback providers
        for provider_name, provider in self.fallback_providers:
            try:
                logger.info(f"🔄 Trying fallback provider: {provider_name}")
                start_time = time.time()
                embeddings = provider.embed_documents(texts)
                duration = time.time() - start_time
                
                self.metrics.log_embedding(provider_name, len(texts), duration)
                self.metrics.log_fallback()
                
                logger.info(f"✅ Fallback successful with {provider_name}")
                return embeddings
            except Exception as e:
                logger.warning(f"⚠️ Fallback provider {provider_name} failed: {str(e)}")
                self.metrics.log_error(provider_name)
        
        raise RuntimeError("All embedding providers failed")
    
    def embed_query_with_fallback(self, text: str) -> List[float]:
        """Attempt query embedding with primary provider, fall back if needed."""
        # Try primary provider first
        if self.primary_provider:
            try:
                return self.primary_provider.embed_query(text)
            except Exception as e:
                logger.warning(f"⚠️ Primary provider query failed: {str(e)}")
                self.metrics.log_error("primary")
        
        # Try fallback providers
        for provider_name, provider in self.fallback_providers:
            try:
                logger.info(f"🔄 Trying fallback provider for query: {provider_name}")
                result = provider.embed_query(text)
                self.metrics.log_fallback()
                logger.info(f"✅ Query fallback successful with {provider_name}")
                return result
            except Exception as e:
                logger.warning(f"⚠️ Fallback provider {provider_name} query failed: {str(e)}")
                self.metrics.log_error(provider_name)
        
        raise RuntimeError("All embedding providers failed for query")
    
    def get_active_provider_info(self) -> Dict[str, Any]:
        """Get information about the currently active provider."""
        if self.primary_provider:
            return self.primary_provider.get_model_info()
        elif self.fallback_providers:
            return self.fallback_providers[0][1].get_model_info()
        else:
            return {"provider": "NONE", "status": "No providers available"}
    
    def get_metrics_report(self) -> Dict:
        """Get embedding performance metrics."""
        return self.metrics.generate_report()