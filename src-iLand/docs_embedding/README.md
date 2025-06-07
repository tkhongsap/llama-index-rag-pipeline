# iLand Thai Land Deed Embedding Pipeline

A production-ready RAG (Retrieval-Augmented Generation) pipeline specifically designed for processing Thai land deed documents. This refactored modular system extracts embeddings from Thai land deed markdown files with comprehensive metadata extraction and structured storage.

## 🏗️ Architecture Overview

The pipeline follows LlamaIndex production RAG best practices with a modular architecture:

```
docs_embedding/
├── __init__.py              # Package initialization and exports
├── metadata_extractor.py    # Thai land deed metadata extraction
├── document_loader.py       # Markdown document loading and processing
├── embedding_processor.py   # DocumentSummaryIndex and embedding generation
├── file_storage.py         # Multi-format output storage (JSON, PKL, NPY)
├── batch_embedding.py      # Main orchestrator script
└── README.md               # This documentation
```

## 📋 Module Responsibilities

### `metadata_extractor.py`
**Core functionality for Thai land deed metadata extraction**

- **`iLandMetadataExtractor`**: Main extraction class with 30+ regex patterns
- **Thai-specific patterns**: Deed serial numbers, land names, area measurements (rai/ngan/wa)
- **Content classification**: Categorizes land use, ownership types, regions
- **Enhanced metadata**: Derives additional fields for filtering and retrieval

Key methods:
```python
extractor = iLandMetadataExtractor()
metadata = extractor.extract_metadata(content, file_path)
categories = extractor._derive_enhanced_categories(metadata)
```

### `document_loader.py`
**Markdown document loading and LlamaIndex Document creation**

- **`iLandDocumentLoader`**: Specialized loader for Thai land deed markdown files
- **Recursive file discovery**: Processes nested directory structures
- **Document title generation**: Creates meaningful titles from metadata
- **Content preprocessing**: Handles Thai text encoding and formatting

Key methods:
```python
loader = iLandDocumentLoader(data_dir="./example")
documents = loader.load_documents()
```

### `embedding_processor.py`
**Embedding generation using DocumentSummaryIndex**

- **`iLandEmbeddingProcessor`**: Handles DocumentSummaryIndex creation and processing
- **Multi-level embeddings**: Extracts embeddings for IndexNodes, chunks, and summaries
- **Structured retrieval**: Maintains metadata relationships in embeddings
- **Statistics tracking**: Comprehensive processing metrics

Key methods:
```python
processor = iLandEmbeddingProcessor(
    model_name="text-embedding-3-small",
    chunk_size=1024
)
results = processor.process_documents(documents)
```

### `file_storage.py`
**Multi-format output storage and management**

- **`iLandFileStorage`**: Handles saving embeddings in multiple formats
- **Format support**: JSON (metadata), PKL (objects), NPY (vectors)
- **Statistics generation**: Detailed processing and embedding statistics
- **Directory management**: Organized output structure with timestamps

Key methods:
```python
storage = iLandFileStorage(output_dir="./output")
storage.save_all_formats(embedding_data, stats)
```

### `batch_embedding.py`
**Main orchestrator script**

- **End-to-end pipeline**: Coordinates all modules for complete processing
- **Batch processing**: Handles large document sets with progress tracking
- **Error handling**: Robust error management with detailed logging
- **Configuration**: Flexible settings for different processing scenarios
- **Per-batch directories**: Saves IndexNode, chunk, and summary embeddings in
  organized subfolders
- **Metadata statistics**: Generates per-batch and combined statistics with
  metadata field analysis
- **API key validation**: Detects project vs standard keys with partial key
  display for debugging

## 🚀 Quick Start

### Prerequisites

```bash
pip install llama-index openai python-dotenv tqdm numpy
```

### Environment Setup

Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### Basic Usage

```python
from docs_embedding import create_iland_embeddings

# Process Thai land deed documents
results = create_iland_embeddings(
    data_dir="./example",
    output_dir="./output",
    model_name="text-embedding-3-small",
    chunk_size=1024,
    batch_delay=1.0
)

print(f"Processed {results['total_documents']} documents")
print(f"Generated {results['total_embeddings']} embeddings")
```

### Advanced Usage

```python
from docs_embedding.metadata_extractor import iLandMetadataExtractor
from docs_embedding.document_loader import iLandDocumentLoader
from docs_embedding.embedding_processor import iLandEmbeddingProcessor
from docs_embedding.file_storage import iLandFileStorage

# Step 1: Load documents with Thai metadata extraction
loader = iLandDocumentLoader("./example")
documents = loader.load_documents()

# Step 2: Process embeddings with DocumentSummaryIndex
processor = iLandEmbeddingProcessor(
    model_name="text-embedding-3-small",
    chunk_size=1024,
    chunk_overlap=200
)
embedding_data = processor.process_documents(documents)

# Step 3: Save in multiple formats
storage = iLandFileStorage("./output")
storage.save_all_formats(embedding_data, processor.get_statistics())
```

## 📊 Thai Land Deed Features

### Metadata Extraction Patterns

The system extracts 30+ metadata fields specific to Thai land deeds:

- **Deed Information**: Serial numbers, deed types, book/page references
- **Location Data**: Province, district, region, GPS coordinates
- **Land Details**: Land names, categories, passport information
- **Area Measurements**: Thai units (rai, ngan, wa) with automatic conversion
- **Dates**: Important dates in Thai and Gregorian calendars
- **System Info**: Processing timestamps and validation status

### Content Classification

Automatic categorization for enhanced retrieval:

```python
categories = {
    'area_category': 'large',      # small/medium/large based on area
    'deed_type': 'ownership',      # ownership/lease/mortgage/etc
    'region_type': 'rural',        # urban/suburban/rural
    'land_use': 'agricultural',    # residential/commercial/agricultural
    'ownership_type': 'individual' # individual/corporate/government
}
```

### Enhanced Filtering

Rich metadata enables sophisticated queries:
- Filter by land area ranges
- Search by deed types and ownership patterns
- Geographic filtering by province/district
- Date range queries
- Land use classification filtering

## 📁 Output Structure

The pipeline generates organized output:

```
output/
└── embeddings_iland_YYYYMMDD_HHMMSS/
    ├── batch_1/
    │   ├── indexnodes/                # IndexNode embeddings
    │   ├── chunks/                    # Chunk embeddings
    │   ├── summaries/                 # Summary embeddings
    │   ├── combined/                  # All embeddings combined
    │   └── batch_1_statistics.json    # Per-batch stats
    ├── batch_2/
    │   ...
    └── combined_statistics.json       # Overall statistics
```

### Statistics Include:
- Document processing metrics
- Embedding generation counts
- Thai metadata extraction success rates
- Unique metadata field analysis
- Processing time and performance data
- Error logs and validation results

## 🔧 Configuration

### Embedding Models
- Default: `text-embedding-3-small` (cost-effective)
- Alternative: `text-embedding-3-large` (higher accuracy)
- Custom: Any OpenAI or local embedding model

### Chunking Strategy
- Default chunk size: 1024 tokens
- Overlap: 200 tokens (configurable)
- Respects sentence boundaries in Thai text

### Batch Processing
- Configurable delays between API calls
- Progress tracking with tqdm
- Memory-efficient processing for large datasets

## 🏭 Production Considerations

### Best Practices Implemented
- ✅ **Decoupled retrieval vs synthesis chunks**: IndexNodes with summaries
- ✅ **Structured retrieval**: Extensive metadata for filtering
- ✅ **Modular architecture**: Clean separation of concerns
- ✅ **Error handling**: Robust error management
- ✅ **Performance monitoring**: Comprehensive statistics

### Scaling Considerations
- Supports batch processing of large document sets
- Memory-efficient streaming for large files
- Configurable API rate limiting
- Parallel processing capabilities (future enhancement)

## 🧪 Testing

Run the pipeline on sample data:

```bash
python batch_embedding.py
```

This will process documents from the `example/` folder and generate embeddings in the `output/` directory.

## 🤝 Contributing

When contributing to this pipeline:

1. **Follow coding rules**: Keep modules under 300 lines
2. **Maintain Thai specificity**: Preserve land deed extraction patterns
3. **Test thoroughly**: Validate with real Thai land deed documents
4. **Document changes**: Update this README for any architectural changes

## 📝 License

This project is part of the llama-index-rag-pipeline repository for processing Thai land deed documents using LlamaIndex and OpenAI embeddings. 