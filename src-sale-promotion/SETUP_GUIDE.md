# Sales Promotion Pipeline - Setup Guide

Quick start guide to get the sales promotion RAG pipeline up and running.

## Prerequisites

1. **Python 3.8+** installed
2. **OpenAI API key** (for embeddings and LLM)
3. **Dependencies** from main project requirements.txt

## Installation Steps

### 1. Environment Setup

```bash
# Create/update .env file in project root
cat >> .env << EOF

# Sales Promotion Pipeline Settings
OPENAI_API_KEY=your_openai_api_key_here

# Optional configurations
CHUNK_SIZE=1024
CHUNK_OVERLAP=50
EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
BATCH_SIZE=10
MAX_WORKERS=4
EOF
```

### 2. Verify Installation

```bash
# Check directory structure
ls -la src-sale-promotion/

# Expected output:
# common/
# data_processing/
# docs_embedding/
# load_embedding/
# retrieval/
# test/
# README.md
```

### 3. Prepare Your Data

Create input directory and add your CSV files:

```bash
# Create input directory
mkdir -p data/input_docs/sale-promotion

# Copy your sales/promotion CSV files
cp your_campaigns.csv data/input_docs/sale-promotion/
```

**Expected CSV Format:**

```csv
campaign_id,campaign_name,campaign_type,product_name,category,price,discount_percentage,clicks,conversions,revenue
CAMP_001,Summer Sale 2024,seasonal_sale,Wireless Headphones,electronics,89.99,25,15420,892,67345
CAMP_002,Email Blast,email,Yoga Mat,sports,34.99,15,8234,456,14256
```

## Pipeline Workflow

### Phase 1: Data Processing

Convert CSV to structured documents:

```bash
python -m src_sale_promotion.data_processing.run_data_processing_standalone

# Current status: Structure created, implementation in progress
# Reference: src-imap/data_processing/ for implementation patterns
```

### Phase 2: Generate Embeddings

Create vector embeddings:

```bash
python -m src_sale_promotion.docs_embedding.batch_embedding

# Current status: Module structure ready, core implementation needed
# Reference: src/09_enhanced_batch_embeddings.py
```

### Phase 3: Query & Retrieve

Interactive retrieval:

```bash
# CLI mode
python -m src_sale_promotion.retrieval.cli --interactive

# Single query
python -m src_sale_promotion.retrieval.cli --query "Show summer campaigns with ROI > 150%"

# Streamlit UI
streamlit run src-sale-promotion/retrieval/streamlit_cli.py

# Current status: CLI structure created, implementation in progress
# Reference: src-imap/retrieval/cli.py
```

## Implementation Roadmap

### ✅ Phase 1: Structure & Documentation (COMPLETED)
- [x] Directory structure
- [x] Module organization
- [x] Comprehensive documentation
- [x] Data models
- [x] Retriever strategy stubs

### 🔄 Phase 2: Core Implementation (IN PROGRESS)
- [ ] Data processing pipeline
- [ ] CSV analyzer and converter
- [ ] Embedding batch processor
- [ ] Embedding loader
- [ ] Basic retrieval strategies

### 📋 Phase 3: Advanced Features (PLANNED)
- [ ] Fast metadata indexing
- [ ] Router with LLM-based strategy selection
- [ ] All 7 retrieval strategies
- [ ] CLI with interactive mode
- [ ] Streamlit web interface

### 📋 Phase 4: Testing & Optimization (PLANNED)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarking
- [ ] Documentation updates

## Development Workflow

### Adding Custom Features

1. **Custom Metadata Fields**
   - Edit `common/constants.py` to add new field categories
   - Update `data_processing/models.py` data models
   - Modify metadata extraction logic

2. **New Retrieval Strategy**
   - Create new file in `retrieval/retrievers/`
   - Inherit from `BaseRetrieverAdapter`
   - Implement `_retrieve()` method
   - Add to router configuration

3. **Custom Query Processing**
   - Extend `index_classifier.py` for new indices
   - Update router logic in `router.py`
   - Add strategy selection rules

### Testing Your Changes

```bash
# Run specific tests
python -m pytest src-sale-promotion/test/test_your_module.py -v

# Run all tests
python -m pytest src-sale-promotion/test/ -v

# With coverage
python -m pytest src-sale-promotion/test/ --cov=src_sale_promotion
```

## Quick Reference

### Key Files

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Main documentation | ✅ Complete |
| `data_processing/models.py` | Data models | ✅ Complete |
| `data_processing/run_data_processing_standalone.py` | Data processing CLI | ✅ Complete |
| `docs_embedding/batch_embedding.py` | Embedding generation | ✅ Complete |
| `load_embedding/load_embedding.py` | Load embeddings | 📋 To implement |
| `retrieval/retrievers/vector.py` | Vector retriever | ✅ Complete |
| `retrieval/retrievers/hybrid.py` | Hybrid retriever | ⚠️ Stub |
| `retrieval/router.py` | Main router | 📋 To implement |
| `retrieval/cli.py` | CLI interface | ✅ Complete |

### Common Commands

```bash
# Data processing
python -m src_sale_promotion.data_processing.run_data_processing_standalone
# Use packaged mock data
python -m src_sale_promotion.data_processing.run_data_processing_standalone --sample

# Embedding generation
python -m src_sale_promotion.docs_embedding.batch_embedding

# Retrieval
python -m src_sale_promotion.retrieval.cli --interactive
python -m src_sale_promotion.retrieval.cli --query "Your query"
# Override embedding directory or list available drops
python -m src_sale_promotion.retrieval.cli --embedding-path path/to/embeddings --query "โปรไหนลดแรง"
python -m src_sale_promotion.retrieval.cli --list-embeddings

# Testing
python -m pytest src-sale-promotion/test/ -v

# Documentation
ls src-sale-promotion/*/README.md  # View all READMEs
```

## Getting Help

1. **Main Documentation**: `src-sale-promotion/README.md`
2. **Module Documentation**: Check each module's `README.md`
3. **Reference Implementations**:
   - `src/` - Main pipeline
   - `src-imap/` - Similar modular structure
   - `src-iLand/` - Thai document processing
4. **Project Guidelines**: `/CLAUDE.md`

## Next Steps

1. **Implement Core Processors**
   - Start with `data_processing/document_processor.py`
   - Reference `src-imap/data_processing/document_processor.py`

2. **Add Embedding Generation**
   - Implement `docs_embedding/batch_embedding.py`
   - Reference `src/09_enhanced_batch_embeddings.py`

3. **Build Retrieval System**
   - Implement retriever strategies
   - Add router logic
   - Create CLI interface

4. **Test & Iterate**
   - Write tests for each module
   - Test with real sales data
   - Optimize performance

## Support

For questions or issues:
- Review documentation in each module's README
- Check reference implementations in `src-imap/`
- Refer to main project `CLAUDE.md` for development guidelines

---

**Status**: Structure complete, ready for implementation!

**Next Action**: Implement data processing pipeline following `src-imap/data_processing/` patterns.
