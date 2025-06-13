# PostgreSQL RAG Pipeline for iLand Documents

This directory contains a comprehensive PostgreSQL-based RAG pipeline that systematically processes Thai land deed documents and stores embeddings with rich metadata in PostgreSQL with pgVector support.

## Architecture Overview

The PostgreSQL pipeline follows the same sophisticated processing patterns as the local pipeline but stores everything in PostgreSQL for scalable retrieval:

```
CSV/Excel → Data Processing → Enhanced PostgreSQL → Embedding Generation → Vector Storage
     ↓              ↓                    ↓                     ↓                    ↓
Raw Data    Metadata Extraction    Structured Storage    Multi-Model Embeddings   Fast Search
```

## Key Features

### 🏗️ Enhanced Data Processing (`data_processing_postgres/`)
- **Rich Metadata Extraction**: Province, district, land use, deed type categorization
- **Enhanced PostgreSQL Schema**: Structured storage with JSONB metadata and indexing
- **Automatic Categorization**: Land use, deed type, and area size categorization
- **Status Tracking**: Processing and embedding status tracking
- **Upsert Support**: Conflict resolution for duplicate documents

### 🧠 Advanced Embedding Pipeline (`docs_embedding_postgres/`)
- **Multi-Model Support**: BGE-M3 (local) + OpenAI (API) with automatic fallback
- **Section-Based Chunking**: Specialized parsing for Thai land deed structure
- **Hierarchical Storage**: Separate tables for chunks, summaries, and index nodes
- **Systematic Organization**: Vector embeddings with full metadata preservation
- **Production-Ready**: Batch processing, error handling, and progress tracking

### 📊 Comprehensive Database Schema
- **Source Table** (`iland_md_data`): Enhanced with metadata columns and status tracking
- **Chunks Table** (`iland_chunks`): Text chunks with embeddings and metadata
- **Summaries Table** (`iland_summaries`): Document summaries with embeddings
- **Index Nodes Table** (`iland_indexnodes`): Hierarchical index nodes for recursive retrieval
- **Combined Table** (`iland_combined`): Unified search across all content types

## Quick Start

### 1. Environment Setup

```bash
# Required environment variables
export DB_NAME="iland-vector-dev"
export DB_USER="vector_user_dev"
export DB_PASSWORD="your_password"
export DB_HOST="10.4.102.11"
export DB_PORT="5432"
export OPENAI_API_KEY="your_openai_key"
```

### 2. Data Processing Pipeline

```bash
# Process CSV/Excel data into PostgreSQL with enhanced metadata
cd src-iLand/data_processing_postgres
python main.py --max-rows 1000 --batch-size 100 --db-batch-size 50

# With custom input file
python main.py --input-file custom_data.xlsx --max-rows 500
```

### 3. Embedding Generation Pipeline

```bash
# Generate embeddings with multi-model support
cd src-iLand/docs_embedding_postgres
python enhanced_postgres_embedding.py --limit 100

# With specific configuration
python enhanced_postgres_embedding.py \
    --limit 500 \
    --chunk-size 512 \
    --batch-size 20 \
    --embed-model text-embedding-3-small \
    --status-filter pending
```

### 4. Test the Complete Pipeline

```bash
# Test end-to-end pipeline
cd src-iLand
python test_postgres_pipeline.py --test-full-pipeline

# Test individual components
python test_postgres_pipeline.py --test-data-processing
python test_postgres_pipeline.py --test-embedding
python test_postgres_pipeline.py --verify-database
```

## Database Schema Details

### Enhanced Source Table (`iland_md_data`)

```sql
CREATE TABLE iland_md_data (
    id SERIAL PRIMARY KEY,
    deed_id TEXT NOT NULL UNIQUE,
    md_string TEXT NOT NULL,
    raw_metadata JSONB,
    extracted_metadata JSONB,
    province TEXT,
    district TEXT,
    land_use_category TEXT,
    deed_type_category TEXT,
    area_category TEXT,
    processing_status TEXT DEFAULT 'pending',
    processing_timestamp TIMESTAMP,
    embedding_status TEXT DEFAULT 'pending',
    embedding_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Vector Storage Tables

```sql
-- Chunks with embeddings
CREATE TABLE iland_chunks (
    id SERIAL PRIMARY KEY,
    deed_id TEXT NOT NULL,
    chunk_index INTEGER,
    text TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    embedding_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document summaries with embeddings
CREATE TABLE iland_summaries (
    id SERIAL PRIMARY KEY,
    deed_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    embedding_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index nodes for hierarchical retrieval
CREATE TABLE iland_indexnodes (
    id SERIAL PRIMARY KEY,
    deed_id TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    embedding_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Combined table for unified search
CREATE TABLE iland_combined (
    id SERIAL PRIMARY KEY,
    deed_id TEXT NOT NULL,
    type TEXT NOT NULL, -- 'chunk', 'summary', 'indexnode'
    text TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    embedding_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Advanced Features

### Multi-Model Embedding Support

The pipeline supports multiple embedding models with automatic fallback:

```python
# Configuration example
embedding_config = {
    "default_provider": "BGE_M3",
    "providers": {
        "BGE_M3": {
            "model_name": "BAAI/bge-m3",
            "device": "auto",
            "batch_size": 32
        },
        "OPENAI": {
            "model_name": "text-embedding-3-small",
            "batch_size": 20
        }
    },
    "fallback_enabled": True,
    "fallback_order": ["BGE_M3", "OPENAI"]
}
```

### Section-Based Chunking

Specialized parsing for Thai land deed documents:

```python
# Automatic section detection
sections = [
    "หนังสือสำคัญ",  # Certificate header
    "รายละเอียดที่ดิน",  # Land details
    "ขอบเขตที่ดิน",  # Land boundaries  
    "เจ้าของ",  # Ownership
    "ภาระผูกพัน"  # Obligations
]
```

### Metadata Categorization

Automatic categorization of key fields:

```python
# Land use categories
land_use_categories = ["agricultural", "residential", "commercial", "industrial", "conservation", "other"]

# Deed type categories  
deed_type_categories = ["chanote", "nor_sor_3", "sor_kor", "other"]

# Area categories (based on rai)
area_categories = ["small", "medium", "large", "very_large"]
```

## Performance and Scaling

### Indexing Strategy

```sql
-- Vector similarity search indexes
CREATE INDEX idx_chunks_embedding ON iland_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_summaries_embedding ON iland_summaries USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_indexnodes_embedding ON iland_indexnodes USING ivfflat (embedding vector_cosine_ops);

-- Metadata search indexes
CREATE INDEX idx_metadata_province ON iland_md_data (province);
CREATE INDEX idx_metadata_land_use ON iland_md_data (land_use_category);
CREATE INDEX idx_metadata_deed_type ON iland_md_data (deed_type_category);

-- JSONB indexes for flexible querying
CREATE INDEX idx_raw_metadata ON iland_md_data USING GIN (raw_metadata);
CREATE INDEX idx_extracted_metadata ON iland_md_data USING GIN (extracted_metadata);
```

### Batch Processing

```python
# Optimized batch sizes
PROCESSING_BATCH_SIZE = 500  # Documents per processing batch
DB_BATCH_SIZE = 100         # Documents per database insertion batch
EMBEDDING_BATCH_SIZE = 20   # Embeddings per API batch
```

## Usage Examples

### Basic Data Processing

```python
from data_processing_postgres.iland_converter import iLandCSVConverter

# Initialize converter
converter = iLandCSVConverter("input_data.xlsx", "output_dir")

# Setup configuration
config = converter.setup_configuration("iland_deeds", auto_generate=True)

# Process and store in PostgreSQL
documents = converter.process_csv_to_documents(batch_size=500)
inserted_count = converter.save_documents_to_database(documents)
```

### Enhanced Embedding Generation

```python
from docs_embedding_postgres.enhanced_postgres_embedding import EnhancedPostgresEmbeddingPipeline

# Initialize pipeline
pipeline = EnhancedPostgresEmbeddingPipeline(
    enable_section_chunking=True,
    enable_multi_model=True,
    chunk_size=512
)

# Run pipeline
result = pipeline.run_pipeline(limit=1000, status_filter="pending")
```

### Vector Search Queries

```sql
-- Find similar documents by content
SELECT deed_id, text, metadata, 
       1 - (embedding <=> %s) AS similarity
FROM iland_combined
WHERE type = 'chunk'
ORDER BY embedding <=> %s
LIMIT 10;

-- Combine vector search with metadata filtering
SELECT deed_id, text, metadata,
       1 - (embedding <=> %s) AS similarity  
FROM iland_chunks c
JOIN iland_md_data m ON c.deed_id = m.deed_id
WHERE m.province = 'Bangkok' 
  AND m.land_use_category = 'commercial'
ORDER BY c.embedding <=> %s
LIMIT 5;
```

## Monitoring and Maintenance

### Status Tracking

```sql
-- Check processing status
SELECT processing_status, COUNT(*) 
FROM iland_md_data 
GROUP BY processing_status;

-- Check embedding status
SELECT embedding_status, COUNT(*) 
FROM iland_md_data 
GROUP BY embedding_status;

-- Check embedding distribution
SELECT embedding_model, type, COUNT(*)
FROM iland_combined
GROUP BY embedding_model, type;
```

### Performance Monitoring

```sql
-- Table sizes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename LIKE 'iland_%';

-- Index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE tablename LIKE 'iland_%';
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check connection
   psql -h 10.4.102.11 -p 5432 -U vector_user_dev -d iland-vector-dev
   ```

2. **Vector Extension Missing**
   ```sql
   -- Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Memory Issues with Large Batches**
   ```python
   # Reduce batch sizes
   chunk_size=256
   batch_size=10
   ```

4. **API Rate Limits**
   ```python
   # Add delays between API calls
   time.sleep(1)
   ```

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with small datasets first
python enhanced_postgres_embedding.py --limit 5 --chunk-size 256
```

## Integration with Retrieval

The PostgreSQL pipeline creates a systematic storage structure that can be used with:

- **LlamaIndex VectorStoreIndex**: Direct integration with PGVectorStore
- **Custom Retrieval**: SQL queries with vector similarity
- **Hybrid Search**: Combining vector similarity with metadata filtering
- **Hierarchical Retrieval**: Using summaries → chunks → nodes pattern

## Future Enhancements

- [ ] **Real-time Processing**: Streaming updates from source systems
- [ ] **Advanced Indexing**: LSH and other approximate similarity methods
- [ ] **Multi-language Support**: Additional language models
- [ ] **Distributed Processing**: Multi-worker embedding generation
- [ ] **Query Optimization**: Automatic index selection and query planning

---

For detailed API documentation and advanced usage, see the individual module READMEs in each subdirectory.