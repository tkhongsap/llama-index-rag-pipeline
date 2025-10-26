# Sales Promotion Embedding Module

Production-ready embedding generation for Thai sales promotion data following Phase 2 PRD specifications.

## Overview

This module transforms structured sales promotion documents from Phase 1 (JSONL format) into production-ready vector embeddings optimized for RAG (Retrieval-Augmented Generation) applications.

**Architecture Pattern:** DocumentSummaryIndex → IndexNodes → Two Embedding Types (IndexNode + Summary)

## Features

- ✅ **DocumentSummaryIndex** - Production-grade hierarchical retrieval structure
- ✅ **LLM Summaries** - GPT-5 generated summaries for each promotion
- ✅ **High-Quality Embeddings** - text-embedding-3-large (3072 dimensions)
- ✅ **IndexNodes** - Document-level navigation nodes
- ✅ **Dual Embeddings** - IndexNode + Summary embeddings (142 total for 71 promotions)
- ✅ **Metadata Preservation** - All 25 Phase 1 fields + 2 Phase 2 fields maintained
- ✅ **Multi-Format Output** - JSON (metadata), PKL (objects), NPY (vectors)
- ✅ **Batch Processing** - Configurable batching with rate limiting
- ✅ **Thai UTF-8 Support** - Full support for Thai language text

## Module Structure

```
docs_embedding/
├── __init__.py                  # Package exports
├── document_loader.py           # JSONL → LlamaIndex Documents
├── embedding_processor.py       # DocumentSummaryIndex builder
├── file_storage.py              # Multi-format output manager
├── batch_embedding.py           # Main orchestrator
└── README.md                    # This file
```

## Quick Start

### 1. Prerequisites

Ensure Phase 1 is complete and you have:
- `data/processed/documents/documents.jsonl` (71 promotions)
- OpenAI API key in `.env` file

```bash
# Check Phase 1 output exists
ls -lh src-sale-promotion/data/processed/documents/documents.jsonl

# Verify .env has OPENAI_API_KEY
cat .env | grep OPENAI_API_KEY
```

### 2. Run the Pipeline

```bash
cd src-sale-promotion/docs_embedding
python batch_embedding.py
```

### 3. Expected Output

```
embeddings_sales_YYYYMMDD_HHMMSS/
├── batch_1/
│   ├── indexnodes/
│   │   ├── batch_1_indexnodes_metadata.json
│   │   ├── batch_1_indexnodes_full.pkl
│   │   └── batch_1_indexnodes_vectors.npy     # Shape: (20, 3072)
│   ├── summaries/
│   │   ├── batch_1_summaries_metadata.json
│   │   ├── batch_1_summaries_full.pkl
│   │   └── batch_1_summaries_vectors.npy
│   ├── combined/
│   │   ├── batch_1_all_metadata.json
│   │   ├── batch_1_all_full.pkl
│   │   └── batch_1_all_vectors.npy
│   └── batch_1_statistics.json
├── batch_2/ [same structure]
├── batch_3/ [same structure]
├── batch_4/ [same structure]
└── combined_statistics.json
```

## Configuration

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (with defaults)
CHUNK_SIZE=2048                          # Larger than doc size (no chunking)
CHUNK_OVERLAP=0                          # No overlap for atomic docs
EMBEDDING_MODEL=text-embedding-3-large   # OpenAI embedding model (3072 dims)
LLM_MODEL=gpt-5                          # For summary generation
BATCH_SIZE=20                            # Promotions per batch
DELAY_BETWEEN_BATCHES=3                  # Seconds between batches
MAX_RETRIES=3                            # API retry attempts
```

### Configuration Dictionary

```python
CONFIG = {
    "input_file": Path("../data/processed/documents/documents.jsonl"),
    "output_dir": Path("../data/embedding"),
    "chunk_size": 2048,              # > doc size (no chunking)
    "chunk_overlap": 0,
    "embedding_model": "text-embedding-3-large",  # 3072 dimensions
    "llm_model": "gpt-5",                         # OpenAI's latest
    "batch_size": 20,
    "delay_between_batches": 3,
    "max_retries": 3
}
```

## Usage Examples

### Basic Usage

```python
from docs_embedding import SalesPromotionBatchEmbeddingPipeline

# Run with default configuration
pipeline = SalesPromotionBatchEmbeddingPipeline()
pipeline.run()
```

### Custom Configuration

```python
from pathlib import Path
from docs_embedding import SalesPromotionBatchEmbeddingPipeline

# Custom configuration
custom_config = {
    "input_file": Path("data/processed/documents/documents.jsonl"),
    "output_dir": Path("data/embedding/custom"),
    "batch_size": 10,  # Smaller batches
    "delay_between_batches": 5,  # More conservative rate limiting
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small"
}

pipeline = SalesPromotionBatchEmbeddingPipeline(config=custom_config)
pipeline.run()
```

### Load Embeddings for Phase 3

```python
import pickle
import numpy as np

# Load PKL format (complete objects with metadata)
with open("batch_1/indexnodes/batch_1_indexnodes_full.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Access embedding data
for emb in embeddings:
    print(f"ID: {emb['node_id']}")
    print(f"Text: {emb['text'][:100]}...")
    print(f"Vector dim: {emb['embedding_dim']}")
    print(f"Metadata fields: {len(emb['metadata'])}")

# Load NPY format (vectors only for numerical operations)
vectors = np.load("batch_1/indexnodes/batch_1_indexnodes_vectors.npy")
print(f"Vector array shape: {vectors.shape}")  # (20, 3072)
```

## Module Documentation

### PromotionDocumentLoader

Loads sales promotion documents from JSONL file.

**Key Methods:**
- `load_documents_from_jsonl(jsonl_path)` - Load all 71 promotions
- `validate_jsonl_file(jsonl_path)` - Pre-flight validation

**Example:**
```python
from docs_embedding import PromotionDocumentLoader

loader = PromotionDocumentLoader()
documents = loader.load_documents_from_jsonl(Path("documents.jsonl"))
print(f"Loaded {len(documents)} promotions")
```

### EmbeddingProcessor

Builds DocumentSummaryIndex and extracts embeddings.

**Key Methods:**
- `build_document_summary_index(documents, llm, embed_model)` - Build index
- `extract_indexnode_embeddings(index_nodes, embed_model, batch_num)` - Extract IndexNode embeddings
- `extract_summary_embeddings(doc_summary_index, embed_model, batch_num)` - Extract Summary embeddings

**Example:**
```python
from docs_embedding import EmbeddingProcessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

processor = EmbeddingProcessor()
llm = OpenAI(model="gpt-5")
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

doc_summary_index = processor.build_document_summary_index(
    documents, llm, embed_model
)
```

### EmbeddingStorage

Manages multi-format embedding storage.

**Key Methods:**
- `save_batch_embeddings(output_dir, batch_num, indexnode_embs, summary_embs)` - Save batch
- `save_combined_statistics(output_dir, all_batches_data)` - Save overall stats

**Example:**
```python
from docs_embedding import EmbeddingStorage

storage = EmbeddingStorage()
storage.save_batch_embeddings(
    output_dir=Path("embeddings_sales_20251026"),
    batch_number=1,
    indexnode_embeddings=indexnode_embs,
    summary_embeddings=summary_embs
)
```

## Processing Flow

```
1. Load documents from JSONL (document_loader)
   ↓
2. Group into batches (default: 20 docs/batch)
   71 docs → 4 batches (20+20+20+11)
   ↓
3. FOR EACH BATCH:
   a. Build DocumentSummaryIndex (embedding_processor)
   b. Create IndexNodes (one per promotion)
   c. Extract IndexNode embeddings (71 total)
   d. Extract Summary embeddings (71 total)
   e. Save batch outputs (file_storage)
   f. Rate limit delay (3 seconds)
   ↓
4. Generate combined statistics across all batches
```

## Success Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| **Documents Processed** | 71/71 (100%) | Check combined_statistics.json |
| **Embeddings Generated** | 142 (71 IndexNode + 71 Summary) | Check grand_totals in statistics |
| **Embedding Dimensions** | 3072 | Check embedding_dim field |
| **Metadata Preservation** | 27 fields (25 original + 2 added) | Check metadata_fields count |
| **Thai UTF-8 Integrity** | 100% preserved | Inspect JSON output files |
| **Processing Time** | < 5 minutes | Monitor pipeline execution |
| **Output Formats** | 3/3 (JSON, PKL, NPY) | Check batch directories |

## Output Formats

### 1. JSON Format (Human-Readable)
- **Purpose:** Metadata with embedding preview
- **File:** `*_metadata.json`
- **Contents:** All metadata, text, first 5 embedding dimensions
- **Use Case:** Inspection, debugging, documentation

### 2. PKL Format (Python Objects)
- **Purpose:** Complete embeddings for Python loading
- **File:** `*_full.pkl`
- **Contents:** Full 3072-dim vectors, metadata, text
- **Use Case:** Phase 3 retrieval loading

### 3. NPY Format (Numpy Vectors)
- **Purpose:** Vector arrays for numerical operations
- **File:** `*_vectors.npy`
- **Contents:** Embedding vectors only (no metadata)
- **Shape:** (N, 3072) where N = number of embeddings
- **Use Case:** Vector similarity computations

## Troubleshooting

### Issue: "JSONL file not found"
**Solution:**
```bash
# Verify Phase 1 completed
ls -lh src-sale-promotion/data/processed/documents/documents.jsonl

# If missing, run Phase 1 first
cd src-sale-promotion/data_processing
python main.py
```

### Issue: "OpenAI API key not found"
**Solution:**
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY

# Add if missing
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

### Issue: "Rate limit exceeded"
**Solution:**
```bash
# Increase delay between batches
export DELAY_BETWEEN_BATCHES=5

# Or reduce batch size
export BATCH_SIZE=10

# Re-run pipeline
python batch_embedding.py
```

### Issue: "Thai text corrupted"
**Solution:**
Ensure JSON files use UTF-8 encoding with `ensure_ascii=False`:
```python
json.dump(data, f, ensure_ascii=False, indent=2)
```

## Phase 3 Compatibility

The embeddings are designed for Phase 3 retrieval strategies:

```python
# Load embeddings for retrieval
import pickle

with open("batch_1/indexnodes/batch_1_indexnodes_full.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Access for retrieval
for emb in embeddings:
    vector = emb['embedding_vector']  # 3072-dim vector
    metadata = emb['metadata']        # All 27 fields
    text = emb['text']                # IndexNode text
```

**Supported Retrieval Strategies:**
- Vector similarity search
- Metadata filtering (customer_type, product_category, etc.)
- Hybrid search (vector + keyword)
- Hierarchical retrieval (IndexNode → Summary)

## Performance

**Expected Metrics:**
- **Processing Time:** ~3-5 minutes for 71 promotions
- **API Calls:** ~213 calls (71 summaries + 142 embeddings)
- **Storage:** ~20-30 MB total
- **Memory:** < 2 GB peak usage

## References

**PRD:** `prd/02_PRD_PHASE2_EMBEDDING_INDEXING.md` (3109 lines)  
**Primary Reference:** `src-imap/docs_embedding/` (proven patterns)  
**Phase 1 Output:** `data/processed/documents/documents.jsonl`

## Next Steps

After Phase 2 completion:

1. **Validate Outputs** - Run acceptance tests
2. **Review Statistics** - Check `combined_statistics.json`
3. **Phase 3 Planning** - Draft retrieval strategy PRD
4. **Handoff** - Prepare embeddings for Phase 3 integration

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-26  
**Status:** Production-Ready
