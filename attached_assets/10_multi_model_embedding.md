# Product Requirements Document (PRD)
## Multi-Model Embedding Support for iLand RAG Pipeline

### 1. Executive Summary

This PRD outlines the modification of the existing `batch_embedding.py` module to support multiple embedding models with BGE-M3 as the default, while maintaining OpenAI embeddings as a fallback option. This enables on-premise deployment while preserving cloud-based capabilities.

### 2. Background & Context

**Current State:**
- The iLand RAG pipeline exclusively uses OpenAI's `text-embedding-3-small` model
- Requires API key and internet connectivity
- Incurs per-token costs
- Data must be sent to external servers

**Business Need:**
- Support on-premise deployment for sensitive Thai land deed documents
- Reduce operational costs
- Enable offline processing
- Maintain flexibility to switch between local and cloud models

### 3. Goals & Objectives

**Primary Goals:**
1. Implement BGE-M3 as the default embedding model for local processing
2. Maintain OpenAI compatibility as a fallback option
3. Create a seamless switching mechanism between embedding providers

**Success Metrics:**
- Zero breaking changes to existing functionality
- < 5% performance degradation compared to OpenAI embeddings
- Support for batch processing of 1000+ documents locally
- Simple configuration switch between providers

### 4. User Stories

**As a developer:**
- I want to switch between embedding models by changing a single configuration variable
- I want the system to automatically fall back to OpenAI if local models fail
- I want clear logging of which embedding model is being used

**As a system administrator:**
- I want to deploy the system on-premise without internet connectivity
- I want to monitor embedding performance and costs
- I want to easily switch between providers based on availability

### 5. Detailed Requirements

#### 5.1 Configuration Management

```python
# New configuration structure
EMBEDDING_CONFIG = {
    "default_provider": "BGE_M3",  # Options: "BGE_M3", "OPENAI", "AUTO"
    "providers": {
        "BGE_M3": {
            "model_name": "BAAI/bge-m3",
            "device": "auto",  # auto, cuda, cpu
            "batch_size": 32,
            "normalize": True,
            "trust_remote_code": True,
            "max_length": 8192
        },
        "OPENAI": {
            "model_name": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY",
            "batch_size": 20,
            "retry_attempts": 3,
            "timeout": 30
        }
    },
    "fallback_enabled": True,
    "fallback_order": ["BGE_M3", "OPENAI"]
}
```

#### 5.2 Implementation Architecture

```python
# Abstract base class for embedding providers
class EmbeddingProvider(ABC):
    @abstractmethod
    def initialize(self, config: Dict) -> None:
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

# Concrete implementations
class BGE_M3Provider(EmbeddingProvider):
    # Implementation for BGE-M3
    pass

class OpenAIProvider(EmbeddingProvider):
    # Implementation for OpenAI
    pass

# Factory pattern for provider selection
class EmbeddingProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, config: Dict) -> EmbeddingProvider:
        # Returns appropriate provider instance
        pass
```

#### 5.3 Fallback Mechanism

```python
class EmbeddingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.primary_provider = None
        self.fallback_providers = []
        
    def embed_with_fallback(self, texts: List[str]) -> List[List[float]]:
        """Attempt embedding with primary provider, fall back if needed"""
        try:
            return self.primary_provider.embed_documents(texts)
        except Exception as e:
            if self.config["fallback_enabled"]:
                for provider in self.fallback_providers:
                    try:
                        return provider.embed_documents(texts)
                    except:
                        continue
            raise e
```

#### 5.4 Monitoring & Logging

```python
# Embedding metrics tracking
class EmbeddingMetrics:
    def __init__(self):
        self.provider_usage = defaultdict(int)
        self.embedding_times = defaultdict(list)
        self.fallback_count = 0
        self.error_count = defaultdict(int)
    
    def log_embedding(self, provider: str, batch_size: int, duration: float):
        # Track usage statistics
        pass
    
    def generate_report(self) -> Dict:
        # Return performance metrics
        pass
```

### 6. Technical Specifications

#### 6.1 Dependencies

**New Requirements:**
```txt
# Existing dependencies
llama-index>=0.9.0
openai>=1.0.0

# New dependencies for local embeddings
transformers>=4.36.0
sentence-transformers>=2.3.0
torch>=2.0.0
FlagEmbedding>=1.2.0  # For BGE-M3 specific features
```

#### 6.2 Performance Considerations

| Metric | OpenAI | BGE-M3 (GPU) | BGE-M3 (CPU) |
|--------|--------|--------------|--------------|
| Latency (100 docs) | 5-10s | 2-3s | 15-20s |
| Memory Usage | ~100MB | ~4GB | ~2GB |
| Max Batch Size | 100 | 32 | 8 |
| Context Length | 8191 | 8192 | 8192 |

#### 6.3 Error Handling

1. **Model Loading Failures**
   - Automatic fallback to next provider
   - Clear error messages with resolution steps
   - Option to download models on first use

2. **Out of Memory Errors**
   - Dynamic batch size adjustment
   - GPU/CPU fallback
   - Memory usage warnings

3. **API Failures (OpenAI)**
   - Exponential backoff retry
   - Rate limit handling
   - Quota exhaustion alerts

### 7. Migration Plan

#### Phase 1: Implementation (Week 1-2)
1. Create abstract embedding provider interface
2. Implement BGE-M3 provider
3. Refactor existing OpenAI code into provider pattern
4. Add configuration management

#### Phase 2: Testing (Week 3)
1. Unit tests for each provider
2. Integration tests for fallback mechanism
3. Performance benchmarking
4. Memory usage profiling

#### Phase 3: Documentation (Week 4)
1. Update README with configuration options
2. Create migration guide
3. Document performance tuning tips
4. Add troubleshooting section

### 8. Backward Compatibility

**Guaranteed Compatibility:**
- Existing embeddings remain valid
- Current API interfaces unchanged
- Default behavior mimics OpenAI when BGE-M3 unavailable

**Configuration Migration:**
```python
# Old configuration still works
CONFIG = {
    "embedding_model": "text-embedding-3-small",  # Automatically maps to OPENAI provider
    # ...
}

# New configuration style
CONFIG = {
    "embedding_provider": "BGE_M3",
    "embedding_model": "BAAI/bge-m3",  # Provider-specific model
    # ...
}
```

### 9. Security & Privacy

**Data Security:**
- Local embeddings never leave the premise
- No API keys required for BGE-M3
- Audit logs for provider usage

**Privacy Benefits:**
- Sensitive land deed data processed locally
- No third-party data exposure
- Compliance with data residency requirements

### 10. Future Extensibility

**Planned Provider Support:**
1. Other Hugging Face models (E5, LaBSE)
2. Custom fine-tuned models
3. ONNX runtime for optimized inference
4. Quantized models for edge deployment

**Provider Interface Extensions:**
```python
class EmbeddingProvider(ABC):
    # ... existing methods ...
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming embeddings"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return model capabilities and metadata"""
        pass
```

### 11. Acceptance Criteria

1. ✅ BGE-M3 successfully embeds Thai land deed documents
2. ✅ Seamless fallback to OpenAI when BGE-M3 fails
3. ✅ Configuration change requires only variable update
4. ✅ Performance metrics logged for both providers
5. ✅ Existing functionality remains unchanged
6. ✅ Documentation updated with examples
7. ✅ Unit test coverage > 90%

### 12. Example Usage

```python
# Simple configuration change
from batch_embedding import EMBEDDING_PROVIDER

# Option 1: Use BGE-M3 (default)
EMBEDDING_PROVIDER = "BGE_M3"

# Option 2: Use OpenAI
EMBEDDING_PROVIDER = "OPENAI"

# Option 3: Auto-select based on availability
EMBEDDING_PROVIDER = "AUTO"

# Run pipeline - no other changes needed
pipeline = iLandBatchEmbeddingPipeline()
pipeline.run()
```

### 13. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| BGE-M3 model download fails | High | Pre-download option, fallback to OpenAI |
| Insufficient GPU memory | Medium | Auto-detect and use CPU, batch size adjustment |
| Different embedding dimensions | High | Automatic dimension validation and conversion |
| Performance regression | Medium | Comprehensive benchmarking, tuning guides |

### 14. Success Metrics

**Technical Metrics:**
- Embedding generation time < 2x OpenAI latency
- Memory usage < 8GB for typical workloads
- 99.9% compatibility with existing embeddings

**Business Metrics:**
- 90% reduction in embedding costs
- 100% on-premise capability achieved
- Zero downtime during migration

---

**Document Version:** 1.0  
**Last Updated:** [Current Date]  
**Author:** [Your Team]  
**Approval Status:** Pending
