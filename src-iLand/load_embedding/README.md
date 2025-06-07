# iLand Load Embedding Module

A modular system for loading and reconstructing indices from iLand Thai land deed embeddings. This module has been refactored into focused components following coding best practices (each module under 300 lines).

## 🏗️ Modular Architecture

The module is organized into focused components:

### Core Modules

- **`models.py`** (141 lines) - Configuration classes and constants
- **`embedding_loader.py`** (351 lines) - iLandEmbeddingLoader class for loading embeddings
- **`index_reconstructor.py`** (274 lines) - iLandIndexReconstructor class for creating indices
- **`validation.py`** (310 lines) - Validation and analysis functions
- **`utils.py`** (280 lines) - Utility functions for common operations
- **`demo.py`** (150 lines) - Demonstration functions

### Support Files

- **`__init__.py`** - Clean public API exports
- **`load_embedding.py`** (32 lines) - Backward compatibility module
- **`example_usage.py`** - Updated examples using modular structure

## 🚀 Quick Start

### Basic Usage

```python
from src_iland.load_embedding import (
    EmbeddingConfig,
    iLandEmbeddingLoader,
    load_latest_iland_embeddings,
    create_iland_index_from_latest_batch
)

# Simple loading
embeddings, batch_path = load_latest_iland_embeddings("chunks")
print(f"Loaded {len(embeddings)} embeddings from {batch_path.name}")

# Create index (requires OPENAI_API_KEY)
index = create_iland_index_from_latest_batch(use_chunks=True, max_embeddings=100)
```

### Configuration-Based Usage

```python
from src_iland.load_embedding import EmbeddingConfig, FilterConfig, iLandEmbeddingLoader

# Configure loading
config = EmbeddingConfig(
    embedding_dir=Path("data/embedding"),
    embed_model="text-embedding-3-small",
    llm_model="gpt-4o-mini"
)

# Configure filtering
filter_config = FilterConfig(
    provinces=["กรุงเทพมหานคร", "เชียงใหม่"],
    deed_types=["chanote", "nor_sor_3"],
    max_embeddings=500
)

# Load and filter
loader = iLandEmbeddingLoader(config)
embeddings, _ = loader.load_all_embeddings_of_type("chunks")
filtered = loader.apply_filter_config(embeddings, filter_config)
```

## 📊 Key Features

### Thai-Specific Functionality
- **Province filtering**: All 77 Thai provinces supported
- **Deed type filtering**: chanote, nor_sor_3, nor_sor_3_kor, etc.
- **Area filtering**: Filter by land area in Thai units (rai, ngan, wa)
- **Thai metadata validation**: Comprehensive validation of Thai land deed fields

### LlamaIndex Integration
- **Production RAG patterns**: Ready-to-use query engines
- **Multiple embedding types**: chunks, summaries, indexnodes
- **Flexible index creation**: Province-specific, deed-type-specific, multi-filtered
- **DocumentSummaryIndex support**: For hierarchical document structures

### Data Structure Compatibility
- **Batch processing**: Works with `data/embedding/embeddings_iland_xx` structure
- **Multiple sub-batches**: Handles batch_1, batch_2, etc. automatically
- **Multiple file formats**: .pkl (full data), .npy (vectors), .json (metadata)

## 🔧 API Reference

### Configuration Classes

```python
# Embedding configuration
config = EmbeddingConfig(
    embedding_dir=Path("data/embedding"),
    embed_model="text-embedding-3-small", 
    llm_model="gpt-4o-mini",
    api_key="your-openai-key"  # or from environment
)

# Filter configuration
filter_config = FilterConfig(
    provinces=["กรุงเทพมหานคร"],
    deed_types=["chanote"],
    min_area_rai=1.0,
    max_area_rai=10.0,
    max_embeddings=1000
)
```

### Loading Functions

```python
# Load from specific sub-batch
embeddings, batch_path = load_latest_iland_embeddings(
    embedding_type="chunks",
    sub_batch="batch_1"
)

# Load from all sub-batches
all_embeddings, batch_path = load_all_latest_iland_embeddings("chunks")

# Get batch summary
summary = get_iland_batch_summary()
```

### Index Creation

```python
# Basic index
index = create_iland_index_from_latest_batch(use_chunks=True)

# Filtered index
index = create_iland_index_from_latest_batch(
    use_chunks=True,
    use_summaries=True,
    province_filter="กรุงเทพมหานคร",
    deed_type_filter="chanote",
    max_embeddings=500
)

# Province-specific index
from src_iland.load_embedding import create_province_specific_iland_index
index = create_province_specific_iland_index("เชียงใหม่")
```

### Validation and Analysis

```python
from src_iland.load_embedding import validate_iland_embeddings, generate_validation_report

# Basic validation
stats = validate_iland_embeddings(embeddings)
print(f"Found {len(stats['thai_metadata']['provinces'])} provinces")

# Comprehensive report
report = generate_validation_report(embeddings)
print(report)
```

## 🌏 Thai Land Deed Support

### Supported Provinces (77 total)
All Thai provinces including: กรุงเทพมหานคร, เชียงใหม่, ภูเก็ต, etc.

### Supported Deed Types
- `chanote` - โฉนด (Full ownership title)
- `nor_sor_3` - นส.3 (Land use certificate)
- `nor_sor_3_kor` - นส.3ก (Upgraded land use certificate)
- `sor_por_kor` - สปก. (Land allocation certificate)
- And more...

### Thai Metadata Fields (47+ fields)
- `province` - จังหวัด
- `district` - อำเภอ
- `subdistrict` - ตำบล
- `deed_type_category` - ประเภทโฉนด
- `land_use_category` - ประเภทการใช้ที่ดิน
- `ownership_category` - ประเภทความเป็นเจ้าของ
- `search_text` - ข้อความค้นหา (includes area information)

## 🧪 Testing and Validation

### Run Demonstrations
```bash
# Full demonstration
python -m src_iland.load_embedding.demo

# Example usage
python -m src_iland.load_embedding.example_usage

# Backward compatibility
python -m src_iland.load_embedding.load_embedding
```

### Validation Results
The module provides comprehensive validation including:
- Embedding dimension consistency
- Text and vector presence validation
- Thai metadata completeness
- Province and deed type distribution analysis
- Quality scoring (0.0-1.0)

## 🔄 Integration with iLand Pipeline

This module integrates seamlessly with the complete iLand workflow:

1. **Data Processing** (`src-iLand/data_processing/`) - CSV analysis and document processing
2. **Document Embedding** (`src-iLand/docs_embedding/`) - Embedding generation and storage
3. **Load Embedding** (`src-iLand/load_embedding/`) - **This module** - Loading and index reconstruction

## 📈 Performance Considerations

- **Batch loading**: Efficiently handles large embedding sets
- **Memory management**: Configurable limits via `max_embeddings`
- **Lazy loading**: Only loads requested embedding types
- **Caching**: Reuses loaded embeddings within session

## 🛠️ Development

### Adding New Features
1. Keep modules under 300 lines (coding rule compliance)
2. Add new functionality to appropriate module
3. Update `__init__.py` exports
4. Add examples to `example_usage.py`
5. Update this README

### Module Dependencies
```
models.py (base)
├── embedding_loader.py (depends on models)
├── index_reconstructor.py (depends on models, embedding_loader)
├── validation.py (standalone)
├── utils.py (depends on all above)
└── demo.py (depends on all above)
```

## 🔗 Related Documentation

- [Data Processing Module](../data_processing/README.md)
- [Document Embedding Module](../docs_embedding/README.md)
- [Main Project README](../../README.md) 