# Load Embedding Module

Loads previously generated embeddings and reconstructs LlamaIndex indices for retrieval.

## Purpose

- **Input**: Embedding files from `docs_embedding/` module
- **Output**: LlamaIndex VectorStoreIndex ready for queries
- **Process**: Load embeddings → reconstruct nodes → build index

## Usage

### Load Latest Embeddings
```python
from src_sale_promotion.load_embedding.load_embedding import load_latest_embeddings

index = load_latest_embeddings()
query_engine = index.as_query_engine()
response = query_engine.query("Show me summer campaign ROI")
```

### Load Specific Embedding File
```python
from src_sale_promotion.load_embedding.load_embedding import load_embeddings

index = load_embeddings("data/embedding/sale-promotion/embeddings_20241025_143022.pkl")
```

### Command Line
```bash
python -m src_sale_promotion.load_embedding.load_embedding --list    # List available embeddings
python -m src_sale_promotion.load_embedding.load_embedding --latest  # Load latest
```

## Features

- ✅ Automatic detection of latest embeddings
- ✅ Support for multiple embedding formats (PKL, JSON, NPY)
- ✅ Index reconstruction with metadata
- ✅ Validation and integrity checks
- ✅ Fast loading with caching
- ✅ Memory-efficient streaming for large indices

## Files Loaded

The module can load from any of these formats:
```
embeddings_TIMESTAMP.pkl   # Preferred: Full nodes with embeddings
embeddings_TIMESTAMP.json  # Metadata + embedding preview
embeddings_TIMESTAMP.npy   # Numpy arrays (fast but no metadata)
```

## Index Reconstruction

The loader:
1. Loads embedding vectors
2. Reconstructs LlamaIndex `TextNode` objects
3. Preserves all sales promotion metadata
4. Builds `VectorStoreIndex`
5. Validates integrity

## Metadata Access

Once loaded, query results include full metadata:
```python
index = load_latest_embeddings()
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("high ROI campaigns")

for node in nodes:
    print(f"Campaign: {node.metadata['campaign_name']}")
    print(f"ROI: {node.metadata['roi']}%")
    print(f"Revenue: ${node.metadata['revenue']}")
    print(f"Score: {node.score}")
```

## Performance

- **Load Time**: ~1-2 seconds for 10k documents
- **Memory**: ~2x embedding file size during load
- **Index Build**: ~100-200ms for 10k documents

## Validation

The loader performs automatic validation:
- ✓ Embedding dimensions match
- ✓ Metadata fields present
- ✓ No corrupted vectors
- ✓ Node count matches expected

## Error Handling

### No Embeddings Found
```
Error: No embeddings found in data/embedding/sale-promotion/
Solution: Run the embedding generation first:
  python -m src_sale_promotion.docs_embedding.batch_embedding
```

### Corrupted File
```
Error: Failed to load embeddings_20241025.pkl
Solution: Delete corrupted file and regenerate embeddings
```

### Version Mismatch
```
Warning: Embeddings created with different LlamaIndex version
Solution: Consider regenerating for compatibility
```

## Testing

```bash
python -m pytest src-sale-promotion/test/test_load_embedding.py -v
```

## Advanced Usage

### Custom Index Configuration
```python
from llama_index.core import VectorStoreIndex, StorageContext
from src_sale_promotion.load_embedding import load_nodes

nodes = load_nodes("embeddings_20241025.pkl")

# Custom index settings
index = VectorStoreIndex(
    nodes,
    show_progress=True,
    service_context=custom_service_context
)
```

### Partial Loading
```python
# Load only specific campaigns
nodes = load_nodes(
    "embeddings_20241025.pkl",
    filter_fn=lambda node: node.metadata.get("campaign_type") == "email"
)
```

## Reference

Based on implementation from:
- `src/load_embeddings.py` - Main pipeline
- `src-imap/load_embedding/` - Modular structure
