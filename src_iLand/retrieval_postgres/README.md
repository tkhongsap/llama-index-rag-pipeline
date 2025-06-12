# PostgreSQL Retrieval System for iLand RAG Pipeline

A comprehensive PostgreSQL-based retrieval system that maintains complete parity with the local file-based implementation while leveraging PostgreSQL's advanced features.

## 🎯 Overview

The PostgreSQL retrieval system provides:
- **Complete Feature Parity**: All 7 retrieval strategies from the local implementation
- **Two-Stage Routing**: Index classification → Strategy selection (identical to local)
- **BGE-M3 Embeddings**: Native support for 1024-dimensional embeddings with pgVector
- **Hybrid Search**: Combines vector similarity with PostgreSQL full-text search
- **Thai Language Support**: Optimized for Thai land deed terminology
- **Advanced Analytics**: Query logging, performance metrics, and usage statistics

## 🏗️ Architecture

```
PostgresRouterRetriever (Main Entry Point)
    ├── PostgresIndexClassifier (Stage 1: Route to Index)
    │   ├── LLM-based classification
    │   ├── Embedding-based classification
    │   └── PostgreSQL caching & logging
    │
    └── Strategy Selection (Stage 2: Choose Retrieval Method)
        ├── PostgresVectorRetriever (Semantic similarity)
        ├── PostgresHybridRetriever (Vector + Full-text)
        ├── PostgresRecursiveRetriever (Hierarchical)
        ├── PostgresChunkDecouplingRetriever (Chunk-level)
        ├── PostgresPlannerRetriever (Multi-step)
        ├── PostgresMetadataRetriever (Metadata-first)
        └── PostgresSummaryRetriever (Summary-based)
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install psycopg2-binary asyncpg pandas numpy tqdm

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=iland_rag
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password
export OPENAI_API_KEY=your_openai_key_here
```

### 2. Test Connection

```bash
cd src-iLand
python -m retrieval_postgres.cli --test-connection
```

### 3. Basic Query

```bash
python -m retrieval_postgres.cli --query "โฉนดที่ดินจังหวัดชัยนาท"
```

## 💻 CLI Commands

### System Management
```bash
# Test database connection
python -m retrieval_postgres.cli --test-connection

# Initialize system (check tables, indexes)
python -m retrieval_postgres.cli --init-system

# Show database status
python -m retrieval_postgres.cli --db-status

# Check database indices
python -m retrieval_postgres.cli --check-indices
```

### Query Operations
```bash
# Basic query with auto strategy selection
python -m retrieval_postgres.cli --query "โฉนดที่ดินในจังหวัดชัยนาท"

# Query with specific strategy
python -m retrieval_postgres.cli --query "การโอนที่ดิน" --strategy vector

# Query with filters
python -m retrieval_postgres.cli --query "ที่ดิน" --province "ชัยนาท" --top-k 10

# Interactive mode
python -m retrieval_postgres.cli --interactive
```

### Strategy Testing
```bash
# Test all strategies
python -m retrieval_postgres.cli --test-all-strategies

# Compare specific strategies
python -m retrieval_postgres.cli --query "ที่ดินในอำเภอเมือง" --compare-strategies vector hybrid metadata

# Run benchmarks
python -m retrieval_postgres.cli --benchmark
```

## 🔧 Available Strategies

1. **auto** - Automatic strategy selection using LLM
2. **vector** - Similarity search using BGE-M3 embeddings
3. **hybrid** - Combines vector similarity with keyword search
4. **recursive** - Multi-level retrieval with chunk relationships
5. **chunk_decoupling** - Retrieves chunks and their parent context
6. **planner** - Query planning with multi-step retrieval
7. **metadata** - Metadata-first filtering with semantic search
8. **summary** - Document summary-based retrieval

## 🏛️ Core Components

### Router System (`router.py`)
- **PostgresRouterRetriever**: Inherits from `iLandRouterRetriever` for exact parity
- Two-stage routing with comprehensive logging
- Performance metrics tracking (p50, p95, p99 latencies)
- Query analytics and caching

### Index Classifier (`index_classifier.py`)
- **PostgresIndexClassifier**: Inherits from `iLandIndexClassifier`
- PostgreSQL caching for classification results
- Automatic index metadata management
- LLM and embedding-based classification

### Configuration (`config.py`)
- **PostgresRetrievalConfig**: Centralized configuration management
- Database connection settings and performance parameters
- Table names and schema configuration

### Adapters (`adapters.py`)
- Interface compatibility layer
- Seamless integration with existing router infrastructure
- Maintains exact API contracts

### Database Utilities (`utils/`)
- **db_connection.py**: Connection pooling and management
- **vector_ops.py**: pgVector operations and optimizations
- **metadata_utils.py**: JSONB indexing and metadata operations

## 🔍 Retrieval Strategies

### Vector Retriever (`retrievers/vector.py`)
```python
# pgVector similarity search with BGE-M3 embeddings
cursor.execute(f"""
    SELECT c.id, c.content, c.metadata, 
           1 - (c.embedding <=> %s::vector) as similarity
    FROM {self.config.chunks_table} c
    WHERE 1 - (c.embedding <=> %s::vector) >= %s
    ORDER BY c.embedding <=> %s::vector LIMIT %s
""")
```

### Hybrid Retriever (`retrievers/hybrid.py`)
- Combines pgVector similarity with PostgreSQL full-text search
- Thai-aware keyword extraction with weighted land deed terms
- Configurable score weighting between vector and text search

### Metadata Retriever (`retrievers/metadata.py`)
- JSONB-based metadata filtering
- Province/district mapping for Thai locations
- Combined metadata + semantic search

## 🇹🇭 Thai Language Support

The system includes specialized handling for Thai language queries:

- **Province mapping**: Automatically maps Thai province names
- **Land deed terminology**: Enhanced weighting for terms like โฉนด, นส.3, ส.ค.1
- **Unicode normalization**: Proper handling of Thai characters
- **Mixed language**: Support for Thai-English mixed queries

### Sample Thai Queries
```bash
python -m retrieval_postgres.cli --query "โฉนดที่ดินจังหวัดชัยนาท"
python -m retrieval_postgres.cli --query "นส.3 อำเภอเมือง การโอนที่ดิน"
python -m retrieval_postgres.cli --query "ขั้นตอนการจดทะเบียนที่ดิน"
```

## 📊 Analytics and Monitoring

### Query Analytics
```bash
# Show query analytics
python -m retrieval_postgres.cli --analytics

# Show popular queries
python -m retrieval_postgres.cli --popular-queries

# Show metadata statistics
python -m retrieval_postgres.cli --metadata-stats
```

### Performance Metrics
- Query execution times (p50, p95, p99)
- Strategy selection frequency
- Database query performance
- Cache hit rates

## 🔧 Configuration

### Environment Variables
```bash
# Required
export OPENAI_API_KEY=your_openai_key_here

# Database (optional, defaults provided)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=iland_rag
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password

# Optional
export BGE_EMBEDDING_URL=http://localhost:8080/embeddings
```

### Database Schema
The system expects these tables:
- `iland_chunks`: Document chunks with embeddings
- `iland_documents`: Document metadata and summaries
- `query_logs`: Query analytics and performance data

## 🧪 Testing and Development

### Setup for Development
```bash
# Run setup script
python setup_postgres_cli.py

# Test all components
python -m retrieval_postgres.cli --test-all-strategies

# Run benchmarks
python -m retrieval_postgres.cli --benchmark
```

### Output Formats
```bash
# Table format (default)
python -m retrieval_postgres.cli --query "ที่ดิน" --output-format table

# JSON format
python -m retrieval_postgres.cli --query "ที่ดิน" --output-format json

# Export results
python -m retrieval_postgres.cli --query "ที่ดิน" --export results.json
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection failed**: Check PostgreSQL is running and credentials are correct
2. **Missing tables**: Ensure embedding pipeline has been run to create required tables
3. **No results**: Try different strategies or lower similarity threshold
4. **Performance issues**: Check `--analytics` and database indices with `--check-indices`

### Error Messages
- **"psycopg2 not available"**: Install with `pip install psycopg2-binary`
- **"OPENAI_API_KEY not set"**: Set environment variable for LLM features
- **"Connection refused"**: Check PostgreSQL service is running
- **"Table does not exist"**: Ensure embedding pipeline has been run

## 📁 File Structure

```
retrieval_postgres/
├── __init__.py                    # Module initialization
├── config.py                     # Configuration management
├── router.py                     # Main routing logic
├── index_classifier.py           # Index classification
├── adapters.py                   # Interface compatibility
├── cli.py                        # CLI entry point
├── cli_handlers_postgres.py      # CLI implementation
├── setup_postgres_cli.py         # Setup script
├── example_usage.py              # Usage examples
├── retrievers/                   # Retrieval strategies
│   ├── __init__.py
│   ├── vector.py                 # Vector similarity
│   ├── hybrid.py                 # Hybrid search
│   ├── metadata.py               # Metadata filtering
│   ├── recursive.py              # Hierarchical retrieval
│   ├── chunk_decoupling.py       # Chunk-level retrieval
│   ├── planner.py                # Query planning
│   └── summary.py                # Summary-based retrieval
└── utils/                        # Database utilities
    ├── __init__.py
    ├── db_connection.py           # Connection management
    ├── vector_ops.py              # pgVector operations
    └── metadata_utils.py          # Metadata operations
```

## 🎯 Key Features

- **100% Parity**: Maintains identical behavior to local retrieval
- **Performance**: Optimized PostgreSQL queries with proper indexing
- **Scalability**: Supports large document collections
- **Analytics**: Comprehensive query logging and performance tracking
- **Thai Support**: Specialized handling for Thai land deed terminology
- **CLI**: Full-featured command-line interface for testing and operations

For detailed API documentation and advanced usage, see the inline docstrings in each module.