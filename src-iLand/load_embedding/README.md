# iLand Load Embedding Module

A comprehensive system for loading and reconstructing vector indices from iLand Thai land deed embeddings. This module provides production-ready tools for building RAG (Retrieval-Augmented Generation) systems with Thai land deed data.

## 🏗️ Architecture Overview

The module follows a modular architecture with clear separation of concerns:

```
load_embedding/
├── __init__.py                  # Module exports and public API
├── models.py                    # Configuration classes and constants  
├── embedding_loader.py          # Core loading functionality
├── index_reconstructor.py       # Index creation from embeddings
├── validation.py               # Data quality validation and analysis
├── utils.py                    # Convenient utility functions
├── load_embedding_complete.py   # Complete demo with validation & indexing
├── example_usage.py            # Comprehensive usage examples
└── load_embedding.py           # Backward compatibility module
```

### Key Components

- **`EmbeddingConfig`**: Configure embedding models, API keys, directories
- **`FilterConfig`**: Configure data filtering by province, deed type, area  
- **`iLandEmbeddingLoader`**: Load embeddings from iLand processing pipeline
- **`iLandIndexReconstructor`**: Create LlamaIndex indices for querying
- **Validation functions**: Analyze embedding quality and Thai metadata
- **Utility functions**: Quick access to common operations

## 🚀 Quick Start Guide

### Prerequisites

1. **Environment Setup**
   ```bash
   # Set your OpenAI API key (required for index creation)
   export OPENAI_API_KEY="your-openai-api-key"
   
   # Or create a .env file
   echo "OPENAI_API_KEY=your-openai-api-key" > .env
   ```

2. **Data Requirements**
   - Processed embedding files from the iLand docs_embedding pipeline
   - Expected location: `data/embedding/embeddings_iland_YYYYMMDD_HHMMSS/`
   - Files should contain: `batch_N/chunks/`, `batch_N/summaries/`, `batch_N/indexnodes/` directories

### Running Load Embedding - 3 Ways

#### 🎯 Method 1: Complete Loading (Recommended for First Time)

```bash
# From the src-iLand directory
cd src-iLand

# Run the complete embedding loading with validation, filtering, and index creation
python -m load_embedding.load_embedding_complete

# Or run specific examples
python -m load_embedding.example_usage
```

This will:
- ✅ Find and load your latest embedding batch
- ✅ Show statistics about your data  
- ✅ Validate embedding quality
- ✅ Create sample indices (if API key is set)
- ✅ Test Thai language queries

#### 🎯 Method 2: Programmatic Usage (Recommended for Development)

```python
from load_embedding import load_latest_iland_embeddings, create_iland_index_from_latest_batch

# Load embeddings from specific sub-batch
embeddings, batch_path = load_latest_iland_embeddings(
    embedding_type="chunks",
    sub_batch="batch_1"
)

# Load ALL embeddings of a type from latest batch (combines all sub-batches)
all_chunks, batch_path = load_all_latest_iland_embeddings("chunks")

# Create index for querying with filtering
index = create_iland_index_from_latest_batch(
    use_chunks=True,
    use_summaries=False,
    max_embeddings=100,
    province_filter="กรุงเทพมหานคร"
)

# Query in Thai
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("ที่ดินประเภทโฉนดในกรุงเทพมีกี่แปลง?")
```

#### 🎯 Method 3: Configuration-Based Usage (Recommended for Production)

```python
from pathlib import Path
from load_embedding import EmbeddingConfig, FilterConfig, iLandEmbeddingLoader

# Configure your setup
config = EmbeddingConfig(
    embedding_dir=Path("data/embedding"),
    embed_model="text-embedding-3-small",
    llm_model="gpt-4o-mini"
)

# Configure filtering
filter_config = FilterConfig(
    provinces=["กรุงเทพมหานคร", "เชียงใหม่"],
    deed_types=["chanote", "nor_sor_3"],
    min_area_rai=1.0,
    max_area_rai=50.0,
    max_embeddings=1000
)

# Load and filter
loader = iLandEmbeddingLoader(config)
result = loader.load_all_embeddings_of_type("chunks")
filtered = loader.apply_filter_config(result.embeddings, filter_config)

print(f"Filtered from {len(result.embeddings)} to {len(filtered)} embeddings")
```

## 📋 Step-by-Step Usage Instructions

### Step 1: Verify Your Data

```python
from load_embedding import get_iland_batch_summary

# Check what data is available
summary = get_iland_batch_summary()
print(f"Found {summary['total_batches']} batches")
print(f"Latest batch: {summary['latest_batch']}")
if summary['latest_batch_stats']:
    print(f"Total embeddings: {summary['latest_batch_stats']['total_embeddings']}")
```

### Step 2: Load and Validate Embeddings

```python
from load_embedding import load_all_latest_iland_embeddings, validate_iland_embeddings

# Load all chunk embeddings from latest batch (combines all sub-batches)
embeddings, batch_path = load_all_latest_iland_embeddings("chunks")

# Validate quality
stats = validate_iland_embeddings(embeddings)
print(f"✅ Loaded {stats['total_count']} embeddings")
print(f"📍 Found {len(stats['thai_metadata']['provinces'])} provinces")
print(f"📋 Found {len(stats['thai_metadata']['deed_types'])} deed types")
```

### Step 3: Create Index for Querying

```python
from load_embedding import create_iland_index_from_latest_batch

# Create index with filtering
index = create_iland_index_from_latest_batch(
    use_chunks=True,
    use_summaries=False,  # Optional: include summary embeddings
    use_indexnodes=False, # Optional: include indexnode embeddings
    province_filter="กรุงเทพมหานคร",  # Optional: filter by province
    max_embeddings=500  # Limit for performance
)

# Set up query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"
)
```

### Step 4: Query Your Data

```python
# Thai language queries
thai_queries = [
    "ที่ดินในจังหวัดใดมีพื้นที่มากที่สุด?",
    "โฉนดประเภทใดพบมากที่สุด?", 
    "ที่ดินเพื่อการเกษตรมีกี่แปลง?",
    "พื้นที่ที่ดินเฉลี่ยในกรุงเทพเมืองคือเท่าใด?"
]

for query in thai_queries:
    response = query_engine.query(query)
    print(f"❓ {query}")
    print(f"✅ {response}\n")
```

## 🛠️ Advanced Usage

### Multi-Batch Loading

The system supports loading from multiple sub-batches within a single embedding batch:

```python
from load_embedding import iLandEmbeddingLoader

loader = iLandEmbeddingLoader()
latest_batch = loader.get_latest_iland_batch()

# Load ALL embeddings from all sub-batches
all_batch_data = loader.load_all_embeddings_from_batch(latest_batch)

# Access by sub-batch and type
for sub_batch, emb_types in all_batch_data.items():
    print(f"{sub_batch}: {len(emb_types['chunks'])} chunks, {len(emb_types['summaries'])} summaries")
```

### Advanced Filtering

```python
from load_embedding import EmbeddingConfig, FilterConfig, iLandEmbeddingLoader

# Advanced filtering with multiple criteria
filter_config = FilterConfig(
    provinces=["กรุงเทพมหานคร", "นนทบุรี", "ปทุมธานี"],  # Multiple provinces
    deed_types=["chanote"],  # Only full ownership titles
    min_area_rai=1.0,  # Minimum 1 rai
    max_area_rai=50.0,  # Maximum 50 rai
    max_embeddings=1000  # Performance limit
)

loader = iLandEmbeddingLoader()
result = loader.load_all_embeddings_of_type("chunks")
filtered = loader.apply_filter_config(result.embeddings, filter_config)

print(f"Filtered from {len(result.embeddings)} to {len(filtered)} embeddings")
```

### Creating Specialized Indices

```python
# Province-specific index
from load_embedding import create_province_specific_iland_index
bangkok_index = create_province_specific_iland_index("กรุงเทพมหานคร")

# Deed-type-specific index  
from load_embedding import create_deed_type_specific_iland_index
chanote_index = create_deed_type_specific_iland_index(["chanote", "nor_sor_3"])

# Multi-type index with summaries and indexnodes
multi_index = create_iland_index_from_latest_batch(
    use_chunks=True,
    use_summaries=True,
    use_indexnodes=True,
    max_embeddings=200
)
```

### Batch Processing Multiple Provinces

```python
from load_embedding import get_available_provinces, create_province_specific_iland_index

# Get all available provinces
provinces = get_available_provinces()
print(f"Available provinces: {provinces[:5]}...")  # Show first 5

# Create indices for top provinces
top_provinces = ["กรุงเทพมหานคร", "เชียงใหม่", "ภูเก็ต", "ขอนแก่น", "สงขลา"]
province_indices = {}

for province in top_provinces:
    try:
        index = create_province_specific_iland_index(province)
        province_indices[province] = index
        print(f"✅ Created index for {province}")
    except Exception as e:
        print(f"❌ Failed to create index for {province}: {e}")
```

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **"No iLand embedding batches found"**
   ```bash
   # Check if data exists (from project root)
   ls data/embedding/embeddings_iland_*
   
   # Verify directory structure
   python -c "from load_embedding import get_iland_batch_summary; print(get_iland_batch_summary())"
   ```

2. **"No API key available"**
   ```bash
   # Set environment variable
   export OPENAI_API_KEY="your-key-here"
   
   # Or check current setting
   echo $OPENAI_API_KEY
   ```

3. **"Failed to create index"**
   ```python
   # Check embeddings are valid
   from load_embedding import validate_iland_embeddings, load_all_latest_iland_embeddings
   embeddings, _ = load_all_latest_iland_embeddings("chunks")
   stats = validate_iland_embeddings(embeddings)
   print(f"Issues found: {stats['issues']}")
   ```

4. **Memory issues with large datasets**
   ```python
   # Use smaller batch sizes
   index = create_iland_index_from_latest_batch(
       use_chunks=True,
       max_embeddings=100  # Reduce this number
   )
   ```

5. **Import errors**
   ```bash
   # Make sure you're in the src-iLand directory
   cd src-iLand
   
   # Run as module
   python -m load_embedding.load_embedding_complete
   ```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use validation report for detailed analysis
from load_embedding import generate_validation_report, load_all_latest_iland_embeddings
embeddings, _ = load_all_latest_iland_embeddings("chunks")
report = generate_validation_report(embeddings)
print(report)
```

## 📊 Understanding Your Data

### Available Embedding Types

- **`chunks`** - Document chunks (best for detailed queries)
- **`summaries`** - Document summaries (best for overview queries) 
- **`indexnodes`** - Hierarchical nodes (best for structured queries)

### Data Structure

iLand embedding batches are organized as:
```
data/embedding/embeddings_iland_YYYYMMDD_HHMMSS/
├── batch_1/
│   ├── chunks/
│   │   ├── batch_1_chunks_full.pkl
│   │   ├── batch_1_chunks_vectors.npy
│   │   └── batch_1_chunks_metadata.json
│   ├── summaries/
│   └── indexnodes/
├── batch_2/
├── combined_statistics.json
└── ...
```

### Thai Land Deed Metadata

The system recognizes these Thai-specific fields:

- **Location**: `province`, `district`, `subdistrict` (จังหวัด, อำเภอ, ตำบล)
- **Deed Types**: `chanote`, `nor_sor_3`, `sor_por_kor` (โฉนด, นส.3, สปก.)
- **Land Use**: `residential`, `agricultural`, `commercial` (ที่อยู่อาศัย, เกษตร, พาณิชย์)
- **Area**: `area_rai`, `area_ngan`, `area_wa` (ไร่, งาน, ตารางวา)
- **Ownership**: `individual`, `company`, `government` (บุคคล, บริษัท, รัฐ)
- **Coordinates**: `longitude`, `latitude`, `coordinates_formatted` (พิกัดภูมิศาสตร์)

### Supported Provinces (77 total)

```python
# See all supported provinces
from load_embedding import THAI_PROVINCES
print(f"Supported provinces: {len(THAI_PROVINCES)}")
print("Examples:", THAI_PROVINCES[:5])
```

## 📚 Module API Reference

### Configuration Classes

```python
# EmbeddingConfig - Core configuration
config = EmbeddingConfig(
    embedding_dir=Path("data/embedding"),
    embed_model="text-embedding-3-small", 
    llm_model="gpt-4o-mini",
    api_key=None  # Uses environment variable
)

# FilterConfig - Filtering options
filter_config = FilterConfig(
    provinces=["กรุงเทพมหานคร"],
    deed_types=["chanote"],
    min_area_rai=1.0,
    max_area_rai=100.0,
    max_embeddings=1000
)
```

### Core Classes

```python
# iLandEmbeddingLoader - Load embeddings from disk
loader = iLandEmbeddingLoader(config)
result = loader.load_all_embeddings_of_type("chunks")
filtered = loader.apply_filter_config(result.embeddings, filter_config)

# iLandIndexReconstructor - Create indices
reconstructor = iLandIndexReconstructor(config)
index = reconstructor.create_vector_index_from_embeddings(embeddings)
```

### Utility Functions

```python
# Quick loading functions
embeddings, path = load_latest_iland_embeddings("chunks", "batch_1")
all_embeddings, path = load_all_latest_iland_embeddings("chunks")

# Index creation functions
index = create_iland_index_from_latest_batch(use_chunks=True, max_embeddings=100)
bangkok_index = create_province_specific_iland_index("กรุงเทพมหานคร")

# Analysis functions
summary = get_iland_batch_summary()
provinces = get_available_provinces()
stats = validate_iland_embeddings(embeddings)
```

## 🔗 Integration

This module integrates with the complete iLand pipeline:

1. **Data Processing** (`../data_processing/`) → Process CSV files to documents
2. **Document Embedding** (`../docs_embedding/`) → Generate embeddings from documents  
3. **Load Embedding** (`../load_embedding/`) → **This module** → Load embeddings and create indices
4. **Your RAG Application** → Query the created indices

## 📈 Performance Tips

- Start with `max_embeddings=100` for testing
- Use province filtering to reduce data size
- Prefer `chunks` for detailed queries, `summaries` for overview
- Monitor memory usage with large datasets
- Use `similarity_top_k=3-5` for balanced performance
- Load specific sub-batches instead of all embeddings when possible

## 🎯 Running Examples

```bash
# Navigate to the module directory
cd src-iLand

# Run complete embedding loading with all features
python -m load_embedding.load_embedding_complete

# Run specific example scenarios
python -m load_embedding.example_usage

# Test backward compatibility
python -m load_embedding.load_embedding

# Or run individual functions
python -c "from load_embedding import demonstrate_iland_loading; demonstrate_iland_loading()"
```

## 📝 Example Use Cases

### 1. Basic Data Exploration
```python
from load_embedding import get_iland_batch_summary, validate_iland_embeddings, load_all_latest_iland_embeddings

# Get overview
summary = get_iland_batch_summary()
print(f"Found {summary['total_batches']} batches with {summary['latest_batch_stats']['total_embeddings']} embeddings")

# Load and analyze
embeddings, _ = load_all_latest_iland_embeddings("chunks")
stats = validate_iland_embeddings(embeddings)
print(f"Provinces: {list(stats['thai_metadata']['provinces'])[:5]}")
```

### 2. Province-Specific Analysis
```python
from load_embedding import create_province_specific_iland_index

# Create Bangkok-specific index
bangkok_index = create_province_specific_iland_index("กรุงเทพมหานคร")
query_engine = bangkok_index.as_query_engine()

# Query Bangkok land data
response = query_engine.query("พื้นที่ที่ดินเฉลี่ยในกรุงเทพคือเท่าใด?")
```

### 3. Multi-Type Index Creation
```python
from load_embedding import create_iland_index_from_latest_batch

# Comprehensive index with chunks, summaries, and indexnodes
comprehensive_index = create_iland_index_from_latest_batch(
    use_chunks=True,
    use_summaries=True,
    use_indexnodes=True,
    max_embeddings=500
)
```

---

**Ready to start?** Run `python -m load_embedding.load_embedding_complete` for complete embedding loading with validation and indexing! 