# BGE Postgres Pipeline: Detailed Process Documentation

## Pipeline Overview

The BGE Postgres Pipeline is a system for processing land deed data and creating embeddings for RAG (Retrieval Augmented Generation) systems. It uses the BAAI General Embedding (BGE) model and stores data in a PostgreSQL Vector Database to enable semantic search capabilities for land document retrieval.

The pipeline consists of 2 main steps:
1. **Data Processing**: Converts data from Excel/CSV into Markdown documents and stores them in PostgreSQL
2. **Embedding Generation**: Creates embeddings from documents and stores them in PostgreSQL Vector Tables

## System Architecture

```
┌─────────────────┐     ┌───────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│                 │     │                   │      │                  │      │                  │
│  Excel/CSV      │──►  │  Markdown         │──►   │  BGE Embeddings  │──►   │  PostgreSQL      │
│  Documents      │     │  Conversion       │      │  Generation      │      │  Vector Tables   │
│                 │     │                   │      │                  │      │                  │
└─────────────────┘     └───────────────────┘      └──────────────────┘      └──────────────────┘
```

## Detailed Workflow

### 1. Data Processing

#### 1.1 Data File Preparation and Reading
- **File**: `bge_postgres_pipeline.py` - Function `process_data(args)`
- **Details**:
  - Locates Excel/CSV data files in the `data/input_docs` folder
  - Default: `input_dataset_iLand.xlsx` (can specify other files via the `--input-file` argument)
  - Configures output directory for backup files (JSONL)
  - Verifies that the data file exists
  - Displays processing configuration settings (maximum rows, database host/port)

#### 1.2 Creating and Configuring iLand Converter
- **File**: `data_processing_postgres/iland_converter.py`
- **Details**:
  - Creates an `iLandCSVConverter` object to manage data conversion
  - Sets database connection parameters
  - Automatically generates configuration for data conversion by analyzing file structure

#### 1.3 Document Processing and Province Filtering
- **File**: `data_processing_postgres/iland_converter.py` - Function `process_csv_to_documents()`
- **Additional File**: `data_processing_postgres/document_processor.py`
- **Details**:
  - Reads data in chunks using the `_read_data_in_chunks()` function
  - **Province Filtering**: Default is "ชัยนาท" (Chainat) (can be changed with the `--filter-province` argument)
    - Checks the `deed_current_province_name_th` column for filtering
    - Filters only rows where the province value matches the specified one
  - Converts each row into `SimpleDocument` objects
  - Calculates statistics and displays progress during processing

#### 1.4 Data and Statistics Saving
- **File**: `data_processing_postgres/iland_converter.py`
- **Functions**: `save_documents_as_jsonl()`, `save_documents_to_database()`
- **Details**:
  - Saves documents as JSONL files for backup
  - Saves documents to PostgreSQL database (table `iland_md_data`)
  - Displays summary statistics of the conversion

#### 1.5 Database Management
- **File**: `data_processing_postgres/db_manager.py`
- **Details**:
  - Creates source table (`iland_md_data`) if it doesn't exist
  - Manages PostgreSQL database connection
  - Saves documents in batches for efficiency

### 2. Embedding Generation

#### 2.1 Creating Managers
- **File**: `bge_postgres_pipeline.py` - Function `generate_embeddings(args, document_count)`
- **Details**:
  - Creates `EmbeddingsManager` for managing embedding generation
  - Creates `PostgresManager` for managing embedding storage in PostgreSQL
  - Sets various parameters such as model type, chunk size, and table names

#### 2.2 BGE Model Preparation and PostgreSQL Table Creation
- **File**: `docs_embedding_postgres/embeddings_manager.py` - Function `_initialize_embedding_processor()`
- **File**: `docs_embedding_postgres/db_utils.py` - Function `setup_tables()`
- **Details**:
  - **BGE Model Preparation**:
    - Loads model from cache or downloads if necessary
    - Default: `bge-small-en-v1.5` (can be changed with the `--bge-model` argument)
    - Supported BGE Models: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, bge-m3
  - **PostgreSQL Table Creation**:
    - Creates vector extension if it doesn't exist
    - Creates 4 tables for storing embeddings:
      1. `iland_chunks`: For document chunks
      2. `iland_summaries`: For document summaries
      3. `iland_indexnodes`: For document index nodes
      4. `iland_combined`: For all combined embeddings
    - Creates indexes for fast searching (deed_id, vector similarity)

#### 2.3 Retrieving Documents from Database
- **File**: `docs_embedding_postgres/db_utils.py` - Function `fetch_documents()`
- **Details**:
  - Retrieves documents from the `iland_md_data` table
  - Limits the number of documents as specified (from data processing step count)
  - Converts data to a format suitable for embedding generation

#### 2.4 Document Processing and Embedding Generation
- **File**: `docs_embedding_postgres/embeddings_manager.py` - Function `process_documents()`
- **Details**:
  - **Document Summary Index Creation**:
    - Converts documents to LlamaIndex Document objects
    - Creates DocumentSummaryIndex using the BGE embedding model
    - Splits documents into chunks of specified size (default: 512 tokens)
  - **Index Node Creation**:
    - Creates IndexNodes for each document
    - Stores document metadata in nodes
    - Generates document summaries
  - **Embedding Extraction**:
    - Generates embeddings for chunks, summaries, and index nodes
    - Saves embeddings to files as backup

#### 2.5 Embedding Generation (Additional Details)
- **File**: `docs_embedding_postgres/bge_embedding_processor.py`
- **Functions**: `extract_chunk_embeddings()`, `extract_summary_embeddings()`, `extract_indexnode_embeddings()`
- **Details**:
  - **Chunk Embeddings**:
    - Generates embeddings for each document chunk
    - Stores metadata such as deed_id, document_id, chunk_index
  - **Summary Embeddings**:
    - Generates embeddings for document summaries
    - Stores all metadata from the original document
  - **IndexNode Embeddings**:
    - Generates embeddings for index nodes
    - Used for hierarchical document searching

#### 2.6 Saving Embeddings to Database
- **File**: `docs_embedding_postgres/db_utils.py`
- **Functions**: `save_all_embeddings()`, `save_chunk_embeddings()`, `save_summary_embeddings()`, `save_indexnode_embeddings()`, `save_combined_embeddings()`
- **Details**:
  - Saves chunk embeddings to the `iland_chunks` table
  - Saves summary embeddings to the `iland_summaries` table
  - Saves indexnode embeddings to the `iland_indexnodes` table
  - Saves all embeddings to the `iland_combined` table (unified table)
  - Creates vector indexes for similarity search queries

## Pipeline Usage

### Important Command-Line Arguments

```bash
python bge_postgres_pipeline.py [options]
```

#### Data Processing Arguments
- `--max-rows`: Maximum number of rows to process (default: all)
- `--batch-size`: Batch size for processing (default: 500)
- `--db-batch-size`: Batch size for database insertion (default: 100)
- `--input-file`: Custom input filename (default: input_dataset_iLand.xlsx)
- `--filter-province`: Filter data by province name (default: "ชัยนาท")
- `--no-province-filter`: Disable province filtering (process all provinces)

#### BGE Model Arguments
- `--bge-model`: BGE model name to use (default: bge-small-en-v1.5)
- `--cache-folder`: Folder for storing BGE model cache (default: ./cache/bge_models)
- `--chunk-size`: Chunk size for document splitting (default: 512)
- `--chunk-overlap`: Overlap between chunks (default: 50)
- `--embed-batch-size`: Batch size for embedding generation (default: 20)

#### Processing Control
- `--skip-processing`: Skip the data processing step (generate embeddings only)
- `--skip-embeddings`: Skip the embedding generation step (process data only)

### Results Summary

This pipeline will generate the following outputs:
1. JSONL backup files of processed documents
2. Markdown documents in PostgreSQL table `iland_md_data`
3. Embeddings in 4 PostgreSQL tables:
   - `iland_chunks`: Chunk data with embeddings (for sub-document level search)
   - `iland_summaries`: Summary data with embeddings (for document-level search)
   - `iland_indexnodes`: Index node data with embeddings (for hierarchical search)
   - `iland_combined`: All embedding data combined (for unified search)

## Applications

The generated embeddings can be used for:
1. Semantic document search systems
2. Question-answering systems
3. RAG (Retrieval Augmented Generation) systems
4. Document similarity analysis

## System Requirements

### Requirements
- Python 3.9+
- PostgreSQL 13+ with vector extension
- Dependencies as specified in requirements.txt

### Limitations
- Storage space required for BGE models (200MB - 1.5GB depending on model)
- RAM required for processing BGE models (2GB - 8GB depending on model)
- Processing time depends on document count and model size

## Additional Recommendations

1. Start with a small model (bge-small-en-v1.5) for initial testing
2. Use `--max-rows` to limit document count for testing
3. Check PostgreSQL connection before starting the pipeline
4. Consider using the bge-m3 model for Thai language (but requires more resources) 