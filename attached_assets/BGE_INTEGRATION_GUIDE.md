# BGE Embedding Integration Guide

## 🎯 Overview

This guide covers the integration of **BGE (BAAI General Embedding)** models with the iLand docs_embedding module. BGE models are state-of-the-art embedding models from BAAI that can serve as alternatives to OpenAI embeddings for Thai land deed document processing.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install BGE-specific packages only
pip install llama-index-embeddings-huggingface sentence-transformers transformers torch
```

### 2. Basic BGE Usage

```python
from src-iLand.docs_embedding.bge_embedding_processor import create_bge_embedding_processor

# Create BGE processor
processor = create_bge_embedding_processor("bge-small-en-v1.5")

# Generate embedding
text = "โฉนดที่ดิน เลขที่ 12345 จังหวัด กรุงเทพมหานคร"
embedding = processor.embed_model.get_text_embedding(text)
print(f"Embedding dimension: {len(embedding)}")  # 384 for bge-small-en-v1.5
```

### 3. Run BGE Batch Processing

```bash
cd src-iLand
python -m docs_embedding.batch_embedding_bge
```

## 🔧 Configuration

### Available BGE Models

| Model | Dimension | Max Length | Description |
|-------|-----------|------------|-------------|
| `bge-small-en-v1.5` | 384 | 512 | Lightweight, fast BGE model |
| `bge-base-en-v1.5` | 768 | 512 | Balanced BGE model |
| `bge-large-en-v1.5` | 1024 | 512 | High-quality BGE model |
| `bge-m3` | 1024 | 8192 | Multilingual BGE model (supports Thai) |

### Configuration Options

Edit `CONFIG` in `batch_embedding_bge.py`:

```python
CONFIG = {
    "embedding": {
        "provider": "bge",  # or "openai"
        
        "bge": {
            "model_name": "bge-small-en-v1.5",  # Choose from available models
            "cache_folder": "./cache/bge_models",
        },
        
        "openai": {
            "model_name": "text-embedding-3-small",
            "api_key": None  # Uses OPENAI_API_KEY env var
        }
    },
    
    "enable_comparison": True,  # Compare BGE vs OpenAI
}
```

## 📊 BGE vs OpenAI Comparison

### Performance Characteristics

| Aspect | BGE Models | OpenAI Models |
|--------|------------|---------------|
| **Cost** | Free (local) | Pay-per-use API |
| **Privacy** | Complete (local) | Data sent to API |
| **Speed** | Fast (after download) | Network dependent |
| **Thai Support** | Good (especially bge-m3) | Excellent |
| **Model Size** | 100MB - 1GB | N/A (API) |
| **Offline Use** | Yes | No |

### Quality Comparison

Run comparison analysis:

```python
from docs_embedding.batch_embedding_bge import iLandBGEBatchEmbeddingPipeline

# Configure for comparison
config = CONFIG.copy()
config["enable_comparison"] = True

pipeline = iLandBGEBatchEmbeddingPipeline(config)
pipeline.run_comparison_analysis([
    "โฉนดที่ดิน เลขที่ 12345 จังหวัด กรุงเทพมหานคร",
    "Land deed document with property information",
    "ข้อมูลที่ดิน ขนาด 2 ไร่ 3 งาน 45 ตารางวา"
])
```

## 🧪 Testing

### 1. Structure Test (No packages required)

```bash
python test_bge_minimal.py
```

### 2. Full Integration Test (Requires packages)

```bash
# Install packages first
pip install -r requirements.txt

# Run comprehensive test
python test_bge_embedding.py
```

### 3. Manual Testing

```python
# Test BGE embedding
from docs_embedding.bge_embedding_processor import create_bge_embedding_processor

processor = create_bge_embedding_processor("bge-small-en-v1.5")
embedding = processor.embed_model.get_text_embedding("Test text")
print(f"BGE embedding: {len(embedding)} dimensions")

# Test OpenAI embedding (requires API key)
from docs_embedding.bge_embedding_processor import create_openai_embedding_processor

processor = create_openai_embedding_processor("text-embedding-3-small")
embedding = processor.embed_model.get_text_embedding("Test text")
print(f"OpenAI embedding: {len(embedding)} dimensions")
```

## 🔄 Migration from OpenAI to BGE

### Step 1: Update Configuration

```python
# Change from OpenAI
CONFIG = {
    "embedding": {"provider": "openai", ...}
}

# To BGE
CONFIG = {
    "embedding": {"provider": "bge", ...}
}
```

### Step 2: Handle Dimension Differences

| Model | Dimension | Impact |
|-------|-----------|---------|
| OpenAI text-embedding-3-small | 1536 | Reference |
| BGE bge-small-en-v1.5 | 384 | 4x smaller |
| BGE bge-base-en-v1.5 | 768 | 2x smaller |
| BGE bge-large-en-v1.5 | 1024 | 1.5x smaller |

### Step 3: Test Performance

```bash
# Run with BGE
python -m docs_embedding.batch_embedding_bge

# Compare output directories
ls data/embedding/
```

## 📁 File Structure

```
src-iLand/docs_embedding/
├── bge_embedding_processor.py      # Enhanced embedding processor
├── batch_embedding_bge.py          # BGE-enhanced batch processing
├── batch_embedding.py              # Original OpenAI-only version
├── embedding_processor.py          # Original processor
└── ...

test_bge_embedding.py               # Comprehensive BGE test
test_bge_minimal.py                 # Structure test (no packages)
BGE_INTEGRATION_GUIDE.md            # This guide
requirements.txt                    # Updated with BGE dependencies
```

## 🎯 Use Cases

### When to Use BGE

- **Privacy-sensitive projects**: Keep embeddings completely local
- **Cost optimization**: Avoid API costs for large-scale processing
- **Offline environments**: No internet connectivity required
- **Experimentation**: Try different embedding models quickly

### When to Use OpenAI

- **Proven quality**: Established performance for production
- **Thai language**: Excellent Thai language support
- **No infrastructure**: Don't want to manage model downloads
- **Latest features**: Access to newest embedding models

## 🔧 Advanced Configuration

### Model Download Management

```python
# Configure custom cache location
CONFIG = {
    "embedding": {
        "bge": {
            "cache_folder": "/path/to/your/cache",
        }
    }
}
```

### Memory Optimization

```python
# Use smaller models for resource-constrained environments
CONFIG = {
    "embedding": {
        "bge": {
            "model_name": "bge-small-en-v1.5",  # Only 384 dimensions
        }
    }
}
```

### GPU Acceleration

```bash
# Install with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# BGE models will automatically use GPU if available
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: No module named 'sentence_transformers'**
```bash
pip install sentence-transformers
```

**2. Model Download Fails**
```bash
# Check internet connection and try different cache folder
export TRANSFORMERS_CACHE=/tmp/transformers_cache
```

**3. CUDA Out of Memory**
```python
# Use smaller model or CPU-only
CONFIG = {
    "embedding": {
        "bge": {
            "model_name": "bge-small-en-v1.5",  # Smaller model
        }
    }
}
```

**4. Slow First Run**
- BGE models download on first use (~100MB-1GB)
- Subsequent runs are fast
- Use `cache_folder` to persist downloads

### Performance Tips

1. **Model Selection**: Start with `bge-small-en-v1.5` for speed
2. **Batch Processing**: Use larger batch sizes for BGE (no API limits)
3. **GPU Usage**: BGE models benefit from GPU acceleration
4. **Cache Management**: Keep model cache to avoid re-downloading

## 📊 Expected Results

### Typical Performance

- **BGE Small**: ~100 embeddings/second (GPU), ~20/second (CPU)
- **BGE Base**: ~50 embeddings/second (GPU), ~10/second (CPU)
- **OpenAI**: ~5-10 embeddings/second (API dependent)

### Quality Metrics

- **BGE-M3**: Best for multilingual (Thai + English)
- **BGE-Large**: Best quality for English
- **BGE-Small**: Best speed/quality balance

## 🚀 Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run structure test**: `python test_bge_minimal.py`
3. **Run full test**: `python test_bge_embedding.py`
4. **Configure for your use case**: Edit `CONFIG` in `batch_embedding_bge.py`
5. **Process your documents**: `python -m docs_embedding.batch_embedding_bge`
6. **Compare results**: Enable comparison analysis
7. **Optimize performance**: Choose the right model for your needs

## 📚 Resources

- [BGE Paper](https://arxiv.org/abs/2309.07597)
- [BAAI Models](https://huggingface.co/BAAI)
- [LlamaIndex Embeddings](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html)
- [Sentence Transformers](https://www.sbert.net/)

---

**Ready to test BGE embeddings with your Thai land deed documents!** 🎉