# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive RAG (Retrieval-Augmented Generation) pipeline built with LlamaIndex and OpenAI for transforming CSV/document data into intelligent vector embeddings with advanced retrieval strategies. The project includes two main implementations:

1. **Main Pipeline (`src/`)**: General-purpose RAG system with 7+ retrieval strategies
2. **iLand Pipeline (`src-iLand/`)**: Specialized Thai land deed document processing with agentic retrieval workflow

## Architecture

### Main Pipeline (`src/`)
- **Processing Flow**: CSV → Documents → Markdown → Embeddings → Retrieval
- **Core Components**:
  - Document processing and CSV conversion (`02_prep_doc_for_embedding.py`)
  - Batch embedding generation (`09_enhanced_batch_embeddings.py`)
  - 7 retrieval strategies (scripts `10-17`)
  - Agentic retriever with LLM-based routing (`src/agentic_retriever/`)

### iLand Pipeline (`src-iLand/`)
Modular architecture with clear separation of concerns:

```
src-iLand/
├── data_processing/       # CSV → Markdown conversion with Thai province support
├── docs_embedding/        # Embedding generation and batch processing
├── load_embedding/        # Embedding loading and index reconstruction
└── retrieval/            # Agentic retrieval with 7 strategies
    ├── retrievers/       # Individual retrieval strategies
    ├── router.py         # Two-stage routing (index + strategy)
    ├── index_classifier.py # Routes queries to appropriate indices
    ├── fast_metadata_index.py # Sub-50ms metadata filtering
    ├── cli.py            # Command-line interface
    └── streamlit_cli.py  # WhatsApp-style chat UI
```

**Two-Stage Routing Process**:
1. **Index Classification**: Determines which iLand index to search
2. **Strategy Selection**: LLM chooses best retrieval strategy for query

**Fast Metadata Indexing**:
- Sub-50ms response time for metadata filtering on 50k+ documents
- Inverted indices for categorical fields (province, deed_type)
- B-tree indices for numeric fields (area, coordinates)
- Auto-initializes in MetadataRetrieverAdapter during CLI startup

### Retrieval Strategies (Both Pipelines)
1. **Vector** (`retrievers/vector.py`): Basic semantic similarity search
2. **Summary** (`retrievers/summary.py`): Document summary-first retrieval
3. **Recursive** (`retrievers/recursive.py`): Hierarchical multi-level retrieval
4. **Metadata** (`retrievers/metadata.py`): Fast filtering by attributes (Thai geographic/legal metadata)
5. **Chunk Decoupling** (`retrievers/chunk_decoupling.py`): Separates embedding from content storage
6. **Hybrid** (`retrievers/hybrid.py`): Combines vector + keyword search (Thai support)
7. **Planner** (`retrievers/planner.py`): Multi-step query planning agent

## Common Commands

### Main Pipeline
```bash
# Document processing
python src/02_prep_doc_for_embedding.py

# Generate embeddings
python src/09_enhanced_batch_embeddings.py

# Test retrievers
python src/10_basic_query_engine.py          # Vector search
python src/16_hybrid_search.py               # Hybrid search
python src/demo_complete_pipeline.py         # All strategies

# Agentic retriever CLI
python -m src.agentic_retriever
```

### iLand Pipeline

**Data Processing**:
```bash
python -m src_iLand.data_processing.run_data_processing_standalone
```

**Embedding Generation**:
```bash
# Standard batch embedding
python -m src_iLand.docs_embedding.batch_embedding

# Load embeddings
python -m src_iLand.load_embedding.load_embedding
```

**Retrieval**:
```bash
# CLI - Interactive mode
python -m src_iLand.retrieval.cli --load-embeddings latest --interactive

# CLI - Single query
python -m src_iLand.retrieval.cli --load-embeddings latest --query "โฉนดที่ดินในกรุงเทพ"

# CLI - Test multiple queries
python -m src_iLand.retrieval.cli --load-embeddings latest --test-queries "ที่ดินในสมุทรปราการ" "นส.3 คืออะไร"

# CLI - Batch summary
python -m src_iLand.retrieval.cli --batch-summary

# Streamlit UI (WhatsApp-style chat)
streamlit run src-iLand/retrieval/streamlit_cli.py
```

### Testing
```bash
# Main pipeline tests
python -m pytest tests/ -v

# iLand unit tests
python -m pytest src-iLand/test/unit_test_*.py -v

# End-to-end test
python src-iLand/test/end_to_end_modules_test.py
```

## Environment Setup

Required environment variables in `.env`:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
CHUNK_SIZE=1024
CHUNK_OVERLAP=50
EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
BATCH_SIZE=10
MAX_WORKERS=4
REQUEST_DELAY=0.1
MAX_RETRIES=3

# iLand retrieval options
CLASSIFIER_MODE=llm  # or "embedding"
```

## Development Guidelines

### Code Organization
- Prefer editing existing files over creating new ones
- Follow modular architecture: data processing → embedding → retrieval
- Keep files under 200-300 lines; refactor if exceeding
- Avoid duplication - check for existing similar functionality first

### Pattern Consistency
- Iterate on existing patterns before introducing new approaches
- Don't drastically change proven patterns without exhausting alternatives
- Focus on relevant code areas only; don't touch unrelated code
- Use simple solutions and avoid unnecessary complexity

### File Structure
- `src/`: General-purpose RAG pipeline with numbered scripts (`02_`, `09_`, `10-17_`)
- `src-iLand/`: Thai land deed pipeline with modular packages
- `tests/`: Main pipeline tests
- `src-iLand/test/`: iLand-specific unit tests
- `data/`: Input docs and generated embeddings

### Thai Language Support (iLand)
- Unicode range: \u0e00-\u0e7f
- Built-in Thai province support (`common/thai_provinces.py`)
- Land deed types: โฉนด, นส.3, นส.4, ส.ค.1
- Geographic metadata: จังหวัด, อำเภอ, ตำบล

### Performance Optimization
- Use batch processing for large datasets (default: 10 files per batch)
- Enable fast metadata filtering for geographic/attribute queries (sub-50ms)
- Monitor API rate limits and implement delays
- Adjust `MAX_WORKERS` based on CPU cores
- Choose appropriate retrieval strategy:
  - **Vector**: Fast, simple queries
  - **Hybrid**: Best overall for mixed content
  - **Metadata**: Structured queries with filtering
  - **Planner**: Complex multi-step questions

### Testing Strategy
- Write thorough tests for all major functionality
- Use pytest for unit and integration tests
- Test with realistic Thai language data for iLand pipeline
- Validate embedding quality and retrieval accuracy

### Server Management
- Always kill existing related servers before starting new ones
- Start up a new server after making changes for testing
- Handle different environments: dev, test, prod

### Configuration Management
- Never overwrite `.env` files without confirming first
- Use YAML for dataset-specific configurations
- Keep configurations in version control (except secrets)
- Avoid stubbing/fake data in dev or prod environments

## Key Implementation Details

### iLand Router Strategy Selection
The `iLandRouterRetriever` uses LLM-based strategy selection with fallback heuristics:
- Analyzes query complexity and Thai content
- Considers land deed terminology and geographic references
- Provides detailed reasoning for strategy selection
- Supports round-robin testing mode

### Fast Metadata Indexing
- Auto-initializes when `enable_fast_filtering=True` in MetadataRetrieverAdapter
- Builds inverted and B-tree indices from LlamaIndex nodes
- Filters up to 90% of documents before vector search
- <10% memory overhead vs vector store

### Embedding Storage Format
Both pipelines store embeddings in multiple formats:
- **JSON**: Metadata with embedding previews
- **PKL**: Complete Python objects with full embeddings
- **NPY**: Numpy arrays for vector operations
- **Statistics**: Analysis and quality metrics

## Troubleshooting

### Common Issues

**API Key Errors**: Verify `OPENAI_API_KEY` is set in environment

**Memory Issues**: Reduce `BATCH_SIZE`, process smaller datasets, monitor system memory

**Import Errors**: Run from project root, verify dependencies installed

**Missing Embeddings**: Generate embeddings first before retrieval
- Main: `python src/09_enhanced_batch_embeddings.py`
- iLand: `python -m src_iLand.docs_embedding.batch_embedding`

**Empty Retrieval Results**: Check embedding quality, try different strategies, verify Thai text encoding (iLand)

**Slow Performance**: Use vector strategy for simple queries, enable fast metadata filtering, reduce top_k

**Fast Indexing Issues**: Verify `enable_fast_filtering=True`, check embeddings have metadata, review logs

### Debug Mode
Most scripts support verbose output via `--verbose` flag or debug logging:
```python
router.enable_debug_logging(True)
```
