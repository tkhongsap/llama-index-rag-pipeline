# LlamaIndex RAG Pipeline - Source Code Documentation

This directory contains the core processing scripts for the LlamaIndex RAG pipeline that converts CSV data into intelligent vector embeddings.

## 🏗️ Pipeline Scripts Overview

### Core Processing Pipeline (Run in Sequential Order)

| # | Script | Purpose | Input | Output |
|---|--------|---------|-------|--------|
| 02 | `02_csv_to_documents.py` | Convert CSV to structured documents | CSV files | JSONL documents |
| 03 | `03_document_to_markdown.py` | Transform documents to markdown | JSONL documents | Markdown files |
| 04 | `04_build_summary_index.py` | Create document summary index | Markdown files | Summary index |
| 05 | `05_build_recursive_retriever.py` | Build recursive retrieval system | Summary index | Recursive retriever |
| 06 | `06_analyze_index_structure.py` | Analyze and inspect index structure | Built indexes | Analysis reports |
| 07 | `07_extract_embeddings.py` | Extract embeddings for inspection | Built indexes | Embedding files |
| 08 | `08_batch_process_embeddings.py` | **Consolidated pipeline (scripts 04-07) with batch processing** | Markdown files | Batch embeddings |

### Supporting Files

- `csv_converter_cli.py` - Command-line interface for CSV conversion
- `01_simple_converter.py` - Basic converter (for reference)

## 📋 Detailed Script Documentation

### 02_csv_to_documents.py - CSV Data Parser

**Purpose:** Flexible, configuration-driven CSV to document converter

**Key Features:**
- Auto-detects CSV structure and column types
- Maps fields to standardized metadata categories (identifier, demographic, education, career, etc.)
- Generates reusable YAML configuration files
- Handles large datasets efficiently with batching
- Creates structured JSONL documents ready for LLM processing

**Field Categories:**
- `identifier` - ID, No, Code fields
- `demographic` - Age, Province, Region, Location
- `education` - Degree, Major, Institute, School
- `career` - Position, Company, Industry, Experience
- `compensation` - Salary, Bonus, Currency
- `assessment` - Test, Score, Skill, Evaluation

**Usage:**
```bash
python 02_csv_to_documents.py
```

**Configuration Example:**
```yaml
name: my_dataset
field_mappings:
- csv_column: "Age"
  metadata_key: "age"
  field_type: "demographic"
  data_type: "numeric"
  required: true
```

### 03_document_to_markdown.py - Document Formatter

**Purpose:** Transform JSONL documents into markdown format optimized for LLM processing

**Key Features:**
- Converts structured documents to readable markdown
- Preserves metadata relationships
- Optimizes format for embedding generation
- Maintains document hierarchy and structure

**Usage:**
```bash
python 03_document_to_markdown.py
```

### 04_build_summary_index.py - Summary Index Builder

**Purpose:** Create hierarchical document summaries and searchable indexes

**Key Features:**
- Generates document summaries using LLM (GPT-4o-mini)
- Creates DocumentSummaryIndex with embeddings
- Chunks documents for optimal retrieval
- Builds foundation for recursive retrieval system
- Uses configurable chunk size and overlap

**Configuration:**
- Chunk Size: 1024 tokens
- Chunk Overlap: 50 tokens
- Embedding Model: text-embedding-3-small
- LLM Model: gpt-4o-mini

**Usage:**
```bash
python 04_build_summary_index.py
```

### 05_build_recursive_retriever.py - Recursive Retrieval System

**Purpose:** Create a multi-level recursive retrieval system for complex queries

**Key Features:**
- Builds IndexNodes for document hierarchy
- Creates VectorStoreIndex from IndexNodes
- Enables both summary-level and chunk-level retrieval
- Supports complex queries across multiple document levels
- Implements recursive query decomposition

**Usage:**
```bash
python 05_build_recursive_retriever.py
```

### 06_analyze_index_structure.py - Index Analyzer

**Purpose:** Inspect and analyze the built indexes for quality assurance and debugging

**Key Features:**
- Analyzes document coverage and quality
- Reports on chunk distribution and sizes
- Validates index structure integrity
- Provides performance insights and statistics
- Generates detailed analysis reports

**Usage:**
```bash
python 06_analyze_index_structure.py
```

### 07_extract_embeddings.py - Embedding Extractor

**Purpose:** Extract and save embeddings for external analysis or vector database storage

**Key Features:**
- Extracts IndexNode, chunk, and summary embeddings
- Saves embeddings in multiple formats (JSON, PKL, NPY)
- Provides comprehensive embedding statistics
- Enables integration with external vector stores
- Creates detailed embedding analysis reports

**Output Formats:**
- JSON: Metadata with embedding previews
- PKL: Complete Python objects with full embeddings
- NPY: Numpy arrays for vector operations
- Statistics: Analysis and quality metrics

**Usage:**
```bash
python 07_extract_embeddings.py
```

### 08_batch_process_embeddings.py - Batch Processor & Consolidated Pipeline

**Purpose:** Production-scale processing that consolidates scripts 04-07 functionality with batch processing capabilities

**Consolidated Functionality:**
- **Chunking** (from script 04): Uses `SentenceSplitter` for text segmentation
- **Indexing** (from scripts 04-05): Builds `DocumentSummaryIndex` and `IndexNodes`
- **Embedding Extraction** (from script 07): Extracts all embedding types (chunks, summaries, IndexNodes)

**Additional Features:**
- Processes files in configurable batches (default: 10 files)
- Implements API rate limiting with delays between batches
- Handles memory management for large datasets
- Provides batch-level statistics and monitoring
- Multiple output formats (JSON, PKL, NPY)

**When to Use:**
- ✅ Large datasets requiring batch processing
- ✅ Production deployments
- ✅ Memory-constrained environments
- ❌ Development/debugging (use individual scripts 04-07)
- ❌ Interactive analysis (use script 06)

**Configuration:**
- Batch Size: 10 files per batch
- Delay Between Batches: 3 seconds
- Memory-efficient processing
- Individual batch result tracking

**Usage:**
```bash
python 08_batch_process_embeddings.py
```

## 🔧 Configuration & Environment

### Required Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
CHUNK_SIZE=1024
CHUNK_OVERLAP=50
SUMMARY_EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### Directory Structure

```
src/
├── 02_csv_to_documents.py        # CSV → Documents
├── 03_document_to_markdown.py    # Documents → Markdown  
├── 04_build_summary_index.py     # Markdown → Summary Index
├── 05_build_recursive_retriever.py # Index → Retriever
├── 06_analyze_index_structure.py # Index Analysis
├── 07_extract_embeddings.py      # Embedding Extraction
├── 08_batch_process_embeddings.py # Batch Processing
├── csv_converter_cli.py          # CLI interface
├── 01_simple_converter.py        # Reference implementation
└── README.md                     # This file
```

## 🚀 Running the Complete Pipeline

### Development/Testing Approach (Individual Scripts)

```bash
# Step 1-3: Data preparation
python 02_csv_to_documents.py
python 03_document_to_markdown.py

# Step 4-7: Build and analyze pipeline components individually
python 04_build_summary_index.py      # Chunking + Indexing
python 05_build_recursive_retriever.py # Recursive retrieval setup
python 06_analyze_index_structure.py   # Analysis and inspection
python 07_extract_embeddings.py        # Embedding extraction
```

### Production Approach (Batch Processing)

```bash
# Step 1-3: Data preparation
python 02_csv_to_documents.py
python 03_document_to_markdown.py

# Step 4: Consolidated pipeline (replaces scripts 04-07)
python 08_batch_process_embeddings.py  # Chunking + Indexing + Embedding extraction
```

### Data Flow Comparison

**Individual Scripts (04-07):**
```
CSV Files → Documents → Markdown → Summary Index → Recursive Retrieval → Analysis
    ↓           ↓           ↓            ↓               ↓           ↓
  JSONL      Markdown    Index      Retriever      Reports

**Batch Processing (08):**
```
CSV Files → Documents → Markdown → Summary Index → Recursive Retrieval → Analysis → Embeddings
    ↓           ↓           ↓            ↓               ↓           ↓           ↓
  JSONL      Markdown    Index      Retriever      Reports    Analysis   Vectors
```

## 🔍 CLI Tools

### CSV Converter CLI

```bash
# Analyze CSV structure
python csv_converter_cli.py analyze ../data/input_docs/input.csv

# Generate configuration file  
python csv_converter_cli.py config ../data/input_docs/input.csv --config-name my_dataset

# Convert with existing config
python csv_converter_cli.py convert ../data/input_docs/input.csv --config-file my_config.yaml

# Interactive mode
python csv_converter_cli.py interactive
```

## 📊 Output Files & Formats

### Generated by Each Script

**02_csv_to_documents.py:**
- `*_documents.jsonl` - Structured documents
- `*_config.yaml` - Field mapping configuration
- `csv_analysis_report.json` - CSV analysis

**03_document_to_markdown.py:**
- `*.md` files in `../example/` directory
- Markdown-formatted documents ready for indexing

**04_build_summary_index.py:**
- DocumentSummaryIndex objects
- Document summaries and chunk embeddings

**05_build_recursive_retriever.py:**
- VectorStoreIndex from IndexNodes
- Recursive retrieval system

**06_analyze_index_structure.py:**
- Index analysis reports
- Quality metrics and statistics

**07_extract_embeddings.py:**
- `embeddings_*/` directories with:
  - JSON files (metadata + previews)
  - PKL files (complete embeddings)
  - NPY files (vector arrays)
  - Statistics files

**08_batch_process_embeddings.py:**
- `embeddings_batch_*/` directories
- Batch-organized embedding files
- Combined statistics across batches

## 🛠️ Development Notes

### Adding New Scripts

1. Follow the naming convention: `NN_descriptive_name.py`
2. Include comprehensive docstrings and error handling
3. Add progress indicators for long-running operations
4. Save intermediate results for debugging
5. Update this README with new script documentation

### Configuration Best Practices

1. Use environment variables for API keys and model settings
2. Create reusable YAML configurations for different datasets
3. Store configurations in version control
4. Document any custom field mappings or templates

### Performance Optimization

1. Monitor API usage and implement rate limiting
2. Use appropriate batch sizes for memory management
3. Cache expensive operations when possible
4. Profile scripts for bottlenecks in large datasets

## 🔧 Troubleshooting

### Common Issues

**API Key Problems:**
```bash
# Check environment variables
echo $OPENAI_API_KEY
```

**Memory Issues:**
- Reduce batch size in batch processing script
- Process smaller file sets
- Monitor system memory usage

**Path Issues:**
- Always run scripts from project root or src directory
- Use relative paths consistently
- Check file permissions

**Configuration Conflicts:**
- Delete old configuration files to regenerate
- Verify YAML syntax
- Check for duplicate metadata keys

### Debug Mode

Most scripts support verbose output. Check script headers for debug flags and logging options.

## 📈 Performance Monitoring

### Metrics to Track

- Processing time per document
- API calls and costs
- Memory usage during processing
- Embedding quality metrics
- Index structure statistics

### Optimization Strategies

- Batch processing for large datasets
- Parallel processing where applicable
- Caching of expensive operations
- Memory-efficient data structures

---

This pipeline provides a complete solution for converting CSV data into intelligent, searchable vector embeddings using state-of-the-art LLM and embedding technologies. 

**Batch Script (08):**
```
Markdown → [Chunking + Indexing + Embedding] → Batch Embeddings
   ↓              (All in one pipeline)           ↓
Multiple      Memory-efficient processing     Organized
Batches                                      Vector Files
``` 