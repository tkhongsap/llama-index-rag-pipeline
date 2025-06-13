# PostgreSQL BGE-M3 Pipeline for iLand RAG

## Overview

This implementation provides a complete PostgreSQL-based RAG pipeline using BGE-M3 embeddings with section-based chunking. It replaces OpenAI embeddings with local BGE-M3 processing and reduces chunks from ~169 to 6-10 per document.

## Key Features

- **100% Local Processing**: Uses BGE-M3 model - no external API calls
- **Section-Based Chunking**: 6-10 semantic chunks per document (vs 169 with sentence splitting)
- **1024-Dimensional Embeddings**: BGE-M3 with Thai language support
- **PostgreSQL Storage**: Efficient storage with proper indexing
- **Rich Metadata**: 30+ fields preserved throughout the pipeline

## Architecture

```
CSV Data → Documents → PostgreSQL (iland_md_data)
                ↓
        Section Chunking
                ↓
        Chunks → PostgreSQL (iland_chunks)
                ↓
        BGE-M3 Embeddings
                ↓
        Update Chunks with Embeddings
```

## Installation

```bash
# Install BGE-M3 dependencies
pip install FlagEmbedding torch transformers

# Install other requirements
pip install psycopg2-binary pandas numpy python-dotenv
```

## Database Schema

### iland_md_data Table
- Stores complete documents with metadata
- Enhanced with processing status tracking

### iland_chunks Table
```sql
CREATE TABLE iland_chunks (
    id SERIAL PRIMARY KEY,
    deed_id VARCHAR(255) NOT NULL,
    parent_doc_id INTEGER REFERENCES iland_md_data(id),
    text TEXT NOT NULL,
    embedding_vector REAL[],
    section_type VARCHAR(50) NOT NULL,
    section_name VARCHAR(100),
    chunk_type VARCHAR(20) NOT NULL,
    is_primary_chunk BOOLEAN DEFAULT FALSE,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    embedding_model VARCHAR(50) DEFAULT 'BAAI/bge-m3',
    embedding_dim INTEGER DEFAULT 1024,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Usage

### Complete Pipeline (Recommended)

```bash
# Process sample data (10 documents)
python run_postgres_bge_pipeline.py --max-rows 10

# Process all data
python run_postgres_bge_pipeline.py

# Use GPU for faster processing
python run_postgres_bge_pipeline.py --device cuda
```

### Step-by-Step Processing

#### Step 1: Process CSV and Create Chunks
```bash
cd data_processing_postgres
python main.py --enable-chunking --max-rows 10
```

#### Step 2: Generate BGE-M3 Embeddings
```bash
cd ../docs_embedding_postgres
python postgres_embedding_bge.py --batch-limit 100
```

#### Step 3: Verify Results
```bash
python postgres_embedding_bge.py --verify-only
```

## Configuration

Create a `.env` file with your database credentials:

```env
# Database Configuration
DB_NAME=iland-vector-dev
DB_USER=vector_user_dev
DB_PASSWORD=your_password
DB_HOST=10.4.102.11
DB_PORT=5432

# BGE-M3 Configuration
BGE_MODEL=bge-m3
BGE_CACHE_FOLDER=./models
CHUNK_SIZE=512
CHUNK_OVERLAP=50
BATCH_SIZE=32
```

## Performance

### Chunking Efficiency
- **Before**: ~169 chunks per document (sentence splitting)
- **After**: 6-10 chunks per document (section-based)
- **Reduction**: ~94% fewer chunks

### Processing Speed
- **Documents**: ~100-200 docs/minute
- **Embeddings**: ~50-100 chunks/second (GPU)
- **Storage**: ~90% reduction in database records

### Memory Usage
- **BGE-M3 Model**: ~4GB (GPU) / ~2GB (CPU)
- **Peak Usage**: ~8GB with batch processing

## Section Types

The section parser identifies these semantic sections:
- `key_info`: Primary searchable information
- `deed_info`: Deed details and registration
- `location`: Geographic location information
- `area_measurements`: Land size and measurements
- `geolocation`: Coordinates and map data
- `classification`: Land use and categories
- `dates`: Important dates
- `financial`: Financial information

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python postgres_embedding_bge.py --batch-size 16

# Use CPU instead of GPU
python postgres_embedding_bge.py --device cpu
```

### Connection Issues
```bash
# Check database connection
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1"

# Verify chunks table exists
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "\d iland_chunks"
```

### Slow Processing
```bash
# Enable GPU
python postgres_embedding_bge.py --device cuda

# Increase batch size (if memory allows)
python postgres_embedding_bge.py --batch-size 64
```

## Verification

### Check Chunk Distribution
```sql
SELECT 
    deed_id,
    COUNT(*) as chunk_count,
    STRING_AGG(DISTINCT section_type, ', ') as sections
FROM iland_chunks
GROUP BY deed_id
ORDER BY chunk_count DESC
LIMIT 10;
```

### Check Embedding Coverage
```sql
SELECT 
    COUNT(*) as total_chunks,
    COUNT(embedding_vector) as with_embeddings,
    COUNT(embedding_vector) * 100.0 / COUNT(*) as coverage_percent
FROM iland_chunks;
```

### Sample Embeddings
```sql
SELECT 
    deed_id,
    section_type,
    array_length(embedding_vector, 1) as embedding_dim,
    embedding_model
FROM iland_chunks
WHERE embedding_vector IS NOT NULL
LIMIT 5;
```

## Migration from OpenAI

If you have existing OpenAI embeddings:

1. Backup existing data
2. Clear old embeddings
3. Run the BGE-M3 pipeline

```bash
# Backup
pg_dump -t iland_chunks > backup_chunks.sql

# Clear embeddings (keep chunks)
psql -c "UPDATE iland_chunks SET embedding_vector = NULL"

# Generate new BGE-M3 embeddings
python postgres_embedding_bge.py
```

## Benefits

1. **Security**: No data leaves your infrastructure
2. **Cost**: No API fees - one-time model download
3. **Performance**: Faster with local GPU
4. **Quality**: BGE-M3 excellent for Thai language
5. **Efficiency**: 94% fewer chunks to store/search

## Next Steps

After embeddings are generated, you can:
1. Use the retrieval module for semantic search
2. Build RAG applications with local LLMs
3. Create vector indexes for faster search
4. Export embeddings for other uses

## Support

For issues or questions:
- Check logs in the console output
- Verify database connectivity
- Ensure BGE-M3 model is downloaded
- Check available GPU memory