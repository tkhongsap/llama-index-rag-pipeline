# LlamaIndex RAG Pipeline

A comprehensive, production-ready RAG (Retrieval-Augmented Generation) pipeline that transforms CSV data and documents into intelligent, searchable vector embeddings using LlamaIndex and OpenAI. This pipeline features advanced retrieval strategies, batch processing capabilities, and enhanced metadata extraction for sophisticated query filtering.

## 🚀 Overview

This pipeline provides a complete end-to-end solution for building intelligent document retrieval systems:

### 🎯 **Core Capabilities**
- **Document Processing**: Intelligent CSV-to-document conversion with structured metadata extraction
- **Batch Embedding Generation**: Memory-efficient processing with API rate limiting and error handling
- **Advanced Retrieval Strategies**: 7+ different retrieval approaches for various use cases
- **Metadata Intelligence**: Automatic field categorization and derived metadata (salary ranges, experience levels, etc.)
- **Production Ready**: Comprehensive error handling, logging, and performance monitoring

### 🧠 **Key Features**
- **Flexible Data Ingestion**: Auto-detects CSV structure and creates reusable configurations
- **Enhanced Metadata Processing**: Extracts and categorizes fields (demographic, education, career, compensation)
- **Multiple Retrieval Paradigms**: Vector search, hybrid search, metadata filtering, recursive retrieval
- **Full Thai Province Coverage**: Built-in list of all 77 provinces for precise location parsing
- **Batch Processing**: Handles large datasets with configurable batch sizes and memory management
- **Interactive Demos**: Complete pipeline demonstrations with performance comparisons
- **Rate Limiting & Error Handling**: Built-in API rate limiting, retry mechanisms, and comprehensive error handling
- **Memory Management**: Efficient processing of large datasets with configurable memory usage
- **Performance Monitoring**: Built-in timing, progress tracking, and performance analytics
- **Modular Architecture**: Clean separation of concerns with reusable components

## 📁 Project Structure

```
llama-index-rag-pipeline/
├── src/                                    # Main source code
│   ├── 02_prep_doc_for_embedding.py       # Document preparation and CSV conversion
│   ├── 09_enhanced_batch_embeddings.py    # Enhanced batch processing with rate limiting
│   ├── 10_basic_query_engine.py           # Basic vector search
│   ├── 11_document_summary_retriever.py   # Document summarization retrieval
│   ├── 12_recursive_retriever.py          # Recursive retrieval strategy
│   ├── 14_metadata_filtering.py           # Metadata-based filtering
│   ├── 15_chunk_decoupling.py             # Chunk decoupling strategy
│   ├── 16_hybrid_search.py                # Hybrid vector + keyword search
│   ├── 17_query_planning_agent.py         # Intelligent query planning
│   ├── demo_complete_pipeline.py          # Complete pipeline demonstration
│   ├── demo_embeddings.py                 # Embedding generation demo
│   ├── demo_retrieval_pipeline.py         # Retrieval strategy demo
│   ├── load_embeddings.py                 # Embedding loading utilities
│   ├── config.py                          # Configuration management
│   ├── metadata_processing.py             # Enhanced metadata extraction
│   ├── arxiv/                             # ArXiv document processing utilities
│   │   ├── arxiv_processor.py             # ArXiv-specific document processing
│   │   └── arxiv_utils.py                 # ArXiv utility functions
│   └── README.md                          # Technical documentation
├── data/
│   ├── input_docs/                        # Input documents and CSV files
│   ├── embedding/                         # Generated embeddings (created during processing)
│   │   ├── nodes_YYYYMMDD_HHMMSS.json     # Node data with metadata
│   │   ├── embeddings_YYYYMMDD_HHMMSS.pkl # Vector embeddings
│   │   └── stats_YYYYMMDD_HHMMSS.json     # Processing statistics
│   └── sample_docs/                       # Sample dataset for testing
├── tests/                                  # Test files and validation scripts
├── requirements.txt                        # Python dependencies
├── .env.example                           # Environment variables template
├── USAGE_GUIDE.md                         # Quick start guide
├── CLEANUP_SUMMARY.md                     # Maintenance and development notes
└── README.md                              # This file
```

## 🏗️ Pipeline Components

### Core Processing Pipeline

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Document Preparation** | Convert CSV/documents to structured format | CSV files, documents | JSONL documents |
| **Batch Embeddings** | Generate embeddings efficiently | Documents | Vector embeddings |
| **Retrieval Strategies** | Multiple search approaches | Embeddings + queries | Retrieved documents |
| **Query Engines** | Interactive query interfaces | User queries | Generated responses |

### Retrieval Strategies

1. **Basic Query Engine** (`10_basic_query_engine.py`)
   - Simple vector similarity search
   - Top-k retrieval with configurable parameters
   - Foundation for all other strategies

2. **Document Summary Retriever** (`11_document_summary_retriever.py`)
   - Hierarchical document summarization
   - Summary-based initial retrieval
   - Detailed content follow-up

3. **Recursive Retriever** (`12_recursive_retriever.py`)
   - Multi-level recursive search
   - Progressively refined results
   - Deep context understanding

4. **Metadata Filtering** (`14_metadata_filtering.py`)
   - Structured metadata queries
   - Field-specific filtering
   - Efficient data subset retrieval

5. **Chunk Decoupling** (`15_chunk_decoupling.py`)
   - Separate embedding and content storage
   - Optimized for large documents
   - Enhanced retrieval accuracy

6. **Hybrid Search** (`16_hybrid_search.py`)
   - Combines vector and keyword search
   - Best of both search paradigms
   - Robust query handling

7. **Query Planning Agent** (`17_query_planning_agent.py`)
   - Intelligent query decomposition
   - Multi-step reasoning
   - Complex question answering

## ⚡ Quick Start

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   # Create .env file with:
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Prepare your data:**
   ```bash
   # Place documents or CSV files in:
   data/input_docs/your_dataset.csv
   ```

### Basic Usage

#### 1. Document Preparation
```bash
# Convert CSV to documents
python src/02_prep_doc_for_embedding.py
```

#### 2. Generate Embeddings
```bash
# Create batch embeddings
python src/09_enhanced_batch_embeddings.py
```

#### 3. Run Retrieval Demos
```bash
# Test basic retrieval
python src/10_basic_query_engine.py

# Try hybrid search
python src/16_hybrid_search.py

# Full pipeline demo
python src/demo_complete_pipeline.py
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
CHUNK_SIZE=1024                    # Text chunk size for processing
CHUNK_OVERLAP=50                   # Overlap between chunks
EMBED_MODEL=text-embedding-3-small # OpenAI embedding model
LLM_MODEL=gpt-4o-mini              # LLM model for responses
BATCH_SIZE=10                      # Batch size for embedding generation
MAX_WORKERS=4                      # Concurrent processing workers
REQUEST_DELAY=0.1                  # Delay between API requests (seconds)
MAX_RETRIES=3                      # Maximum retry attempts for API calls
```

### Document Configuration

The pipeline automatically handles:
- **CSV files** with flexible schema detection and auto-configuration
- **Text documents** of various formats (PDF, TXT, DOCX, etc.)
- **Automatic field mapping** and intelligent categorization
- **Metadata extraction** with field classification (demographic, career, education, compensation)
- **Configurable text templates** for document formatting
- **Data validation** and cleaning during ingestion
- **Derived metadata** generation (salary ranges, experience levels, location parsing)

## 📊 Demo Scripts

### Complete Pipeline Demo
```bash
python src/demo_complete_pipeline.py
```
- Demonstrates all retrieval strategies
- Performance comparisons
- Interactive testing

### Embedding Generation Demo
```bash
python src/demo_embeddings.py
```
- Shows embedding creation process
- Validates embedding quality
- Performance monitoring

### Retrieval Strategy Demo
```bash
python src/demo_retrieval_pipeline.py
```
- Compares different retrieval approaches
- Query optimization examples
- Result analysis

## 🛠️ Development

### Architecture Overview
The pipeline follows a modular architecture with clear separation of concerns:
- **Data Layer**: Document processing and metadata extraction
- **Embedding Layer**: Vector generation and storage management  
- **Retrieval Layer**: Multiple search strategies and query engines
- **Interface Layer**: Demo scripts and interactive components

### Adding New Retrieval Strategies

1. **Create new script** following naming convention: `NN_strategy_name.py`
2. **Implement required interfaces** from existing strategies:
   ```python
   def create_query_engine(index, **kwargs):
       # Implementation here
       return query_engine
   
   def main():
       # Demo and testing code
       pass
   ```
3. **Add to demo pipeline** for testing and comparison
4. **Update documentation** and add usage examples

### Code Standards
- **Type hints**: Use type annotations for all functions
- **Docstrings**: Document all classes and functions
- **Error handling**: Implement comprehensive exception handling
- **Logging**: Use structured logging for debugging and monitoring
- **Testing**: Add unit tests for new functionality

### Testing Strategy

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific functionality
python src/10_basic_query_engine.py

# Integration testing
python src/demo_complete_pipeline.py --test-mode

# Performance testing
python src/demo_retrieval_pipeline.py --benchmark
```

## 📈 Performance Tips

### Optimization Strategies

1. **Batch Processing**: 
   - Use appropriate batch sizes for your hardware (default: 10)
   - Monitor memory usage during large dataset processing
   - Adjust `MAX_WORKERS` based on your CPU cores

2. **Model Selection**: 
   - **text-embedding-3-small**: Fast, cost-effective for most use cases
   - **text-embedding-3-large**: Higher quality for complex documents
   - **gpt-4o-mini**: Balanced performance for LLM responses

3. **Chunking Strategy**: 
   - **1024 tokens**: Good for most documents (default)
   - **512 tokens**: Better for short, dense content
   - **2048 tokens**: Better for long-form content with context dependencies

4. **Retrieval Strategy Selection**:
   - **Basic**: Fast, simple queries (< 100ms response time)
   - **Hybrid**: Best overall performance for mixed content types
   - **Metadata**: Structured data queries with filtering
   - **Recursive**: Complex, multi-step questions requiring deep analysis

### Performance Monitoring

The pipeline includes built-in performance tracking:
- **Processing time** per batch and overall
- **Memory usage** monitoring and alerts
- **API call statistics** and rate limiting metrics
- **Error rates** and retry attempt tracking
- **Embedding quality** metrics and validation

## 🔍 Usage Examples

### Basic Query
```python
from src.load_embeddings import create_index_from_latest_batch
from src import basic_query_engine

# Load embeddings
index = create_index_from_latest_batch()

# Create query engine
query_engine = basic_query_engine.create_query_engine(index)

# Query
response = query_engine.query("What are the key findings?")
print(response)
```

### Hybrid Search with Metadata
```python
from src import hybrid_search
from src.load_embeddings import create_index_from_latest_batch

# Setup hybrid search with metadata filtering
index = create_index_from_latest_batch()
engine = hybrid_search.create_hybrid_engine(index)

# Query with both vector and keyword search
response = engine.query("senior software engineer with Python experience")

# With metadata filtering
filtered_engine = hybrid_search.create_filtered_engine(
    index, 
    filters={"experience_level": "senior", "skills": "python"}
)
response = filtered_engine.query("backend development")
```

### Batch Processing Example
```python
from src.enhanced_batch_embeddings import process_documents_batch

# Process large dataset with custom configuration
config = {
    "batch_size": 20,
    "max_workers": 6,
    "chunk_size": 1024,
    "chunk_overlap": 50
}

results = process_documents_batch(
    input_path="data/input_docs/large_dataset.csv",
    config=config
)
print(f"Processed {results['total_documents']} documents")
print(f"Generated {results['total_embeddings']} embeddings")
```

## 🔬 Advanced Features

### Metadata Intelligence
The pipeline includes sophisticated metadata processing:
- **Automatic Field Classification**: Categorizes fields into demographic, education, career, and compensation
- **Derived Metadata Generation**: Creates salary ranges, experience levels, and normalized location data
- **Smart Field Mapping**: Intelligent detection and mapping of common field patterns
- **Data Validation**: Built-in validation and cleaning for common data quality issues

### Production-Ready Features
- **Error Handling**: Comprehensive exception handling with detailed logging
- **Rate Limiting**: Built-in API rate limiting to prevent quota exhaustion
- **Memory Management**: Efficient processing of large datasets with memory monitoring
- **Retry Mechanisms**: Automatic retry with exponential backoff for transient failures
- **Progress Tracking**: Real-time progress monitoring for long-running operations
- **Checkpointing**: Resume interrupted processing from the last successful batch

### Integration Capabilities
- **Multiple Input Formats**: CSV, JSON, TXT, PDF, DOCX, and more
- **Flexible Output**: JSON, pickle, or custom formats for embeddings
- **API Integration**: Easy integration with existing systems via Python modules
- **Scalable Architecture**: Designed for both small datasets and enterprise-scale processing

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors:**
   ```bash
   # Verify environment variables
   echo $OPENAI_API_KEY
   ```

2. **Memory Issues:**
   - Reduce batch size in configuration
   - Process smaller document sets
   - Monitor system memory usage

3. **Import Errors:**
   ```bash
   # Ensure you're in the project root
   python -c "import sys; print(sys.path)"
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

4. **Missing Embeddings:**
   ```bash
   # Generate embeddings first
   python src/09_enhanced_batch_embeddings.py
   
   # Verify embeddings were created
   ls -la data/embedding/
   ```

5. **Rate Limit Errors:**
   ```bash
   # Increase delay between requests
   export REQUEST_DELAY=0.5
   
   # Reduce batch size
   export BATCH_SIZE=5
   ```

6. **Performance Issues:**
   ```bash
   # Monitor memory usage
   python src/demo_embeddings.py --monitor-memory
   
   # Reduce chunk size for large documents
   export CHUNK_SIZE=512
   ```

### Debug Mode

Most scripts support verbose output:
```bash
python src/10_basic_query_engine.py --verbose
```

## 📚 Documentation

- [`src/README.md`](src/README.md) - Detailed technical documentation
- [`USAGE_GUIDE.md`](USAGE_GUIDE.md) - Quick start guide  
- [`CLEANUP_SUMMARY.md`](CLEANUP_SUMMARY.md) - Development and maintenance notes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
1. Check the troubleshooting section above
2. Review the detailed documentation in `src/README.md`
3. Check existing issues in the repository
4. Create a new issue with detailed description

---

Built with ❤️ using [LlamaIndex](https://www.llamaindex.ai/) and [OpenAI](https://openai.com/)