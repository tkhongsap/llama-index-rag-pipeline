# Document Embedding Module

Generates vector embeddings for sales promotion documents using OpenAI's embedding models with batch processing and rate limiting.

## Purpose

- **Input**: Processed markdown documents from `data_processing/`
- **Output**: Vector embeddings + metadata stored in multiple formats
- **Process**: Document loading → chunking → embedding generation → storage

## Features

- ✅ Batch processing with configurable batch size
- ✅ API rate limiting and retry logic
- ✅ Memory-efficient streaming for large datasets
- ✅ Multiple output formats (JSON, PKL, NPY)
- ✅ Progress tracking and statistics
- ✅ Resume from interruption
- ✅ Sales-specific metadata preservation

## Usage

### Basic Usage
```bash
python -m src_sale_promotion.docs_embedding.batch_embedding
```

### With Custom Settings
```bash
# Set environment variables
export BATCH_SIZE=20
export MAX_WORKERS=6
export CHUNK_SIZE=1024

python -m src_sale_promotion.docs_embedding.batch_embedding
```

### Programmatic Usage
```python
from src_sale_promotion.docs_embedding.batch_embedding import process_documents

results = process_documents(
    input_dir="data/processed/sale-promotion",
    output_dir="data/embedding/sale-promotion",
    batch_size=10,
    chunk_size=1024
)

print(f"Generated {results['total_embeddings']} embeddings")
```

## Configuration

Environment variables in `.env`:
```bash
# Embedding settings
EMBED_MODEL=text-embedding-3-small  # or text-embedding-3-large
CHUNK_SIZE=1024
CHUNK_OVERLAP=50
BATCH_SIZE=10

# API settings
MAX_WORKERS=4
REQUEST_DELAY=0.1  # seconds between requests
MAX_RETRIES=3
```

## Output Format

### Files Generated
```
data/embedding/sale-promotion/
├── embeddings_20241025_143022.pkl    # Full embeddings with nodes
├── embeddings_20241025_143022.json   # Metadata + preview
├── embeddings_20241025_143022.npy    # Numpy arrays for fast loading
└── stats_20241025_143022.json        # Processing statistics
```

### Metadata Preserved
Each embedded chunk includes:
```json
{
  "campaign_id": "CAMP_001",
  "campaign_type": "seasonal_sale",
  "product_category": "electronics",
  "start_date": "2024-06-01",
  "revenue": 67345,
  "roi": 187,
  "chunk_index": 0,
  "total_chunks": 3
}
```

## Performance

- **Speed**: ~50-100 embeddings/second (depends on model and API limits)
- **Cost**: ~$0.00002/1k tokens (text-embedding-3-small)
- **Memory**: ~100MB per 1000 documents

### Optimization Tips
1. **Use text-embedding-3-small** for cost efficiency (good quality)
2. **Increase BATCH_SIZE** if you have API quota headroom
3. **Adjust CHUNK_SIZE** based on document length
4. **Enable caching** for repeated processing

## Monitoring

The batch processor provides real-time feedback:
```
Processing batch 1/10 [10 documents]
├─ Loaded: 10 documents
├─ Chunked: 45 chunks
├─ Embedded: 45/45 (100%)
└─ Saved: ✓

Progress: [=====>          ] 10% | ETA: 2m 15s
```

## Error Handling

### Rate Limit Errors
- Automatic retry with exponential backoff
- Configurable max retries
- Rate limit tracking

### Memory Errors
- Reduce `BATCH_SIZE`
- Enable streaming mode
- Process in smaller batches

### API Errors
- Logs all errors with context
- Saves partial results
- Resume from last successful batch

## Testing

```bash
# Unit tests
python -m pytest src-sale-promotion/test/test_embedding.py -v

# Integration test with sample data
python -m src_sale_promotion.docs_embedding.batch_embedding --test
```

## Advanced Features

### Custom Metadata Extraction
```python
def extract_campaign_metadata(doc):
    return {
        "campaign_type": doc.campaign.campaign_type,
        "product_categories": [p.category for p in doc.products],
        "roi_bucket": categorize_roi(doc.performance.roi),
        "season": extract_season(doc.campaign.start_date)
    }
```

### Chunk Strategy
- **Default**: Sentence-aware chunking with overlap
- **Sales-specific**: Keep campaign sections together
- **Product-aware**: Don't split product descriptions

## Reference

Based on the architecture from:
- `src/09_enhanced_batch_embeddings.py` - Main pipeline implementation
- `src-imap/docs_embedding/` - Modular structure
