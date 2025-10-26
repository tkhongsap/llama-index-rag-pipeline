# PRD: Phase 2 - Sales Promotion Embedding & Indexing Pipeline

## Document Control

| Property | Value |
|----------|-------|
| **Document Title** | Phase 2: Embedding Generation & Indexing for Sales Promotion RAG Pipeline |
| **Version** | 1.0 |
| **Status** | APPROVED |
| **Created Date** | 2025-10-26 |
| **Last Updated** | 2025-10-26 |
| **Authors** | Development Team |
| **Stakeholders** | RAG Pipeline Team, Sales Operations |
| **Dependencies** | Phase 1: Data Preparation Pipeline (v2.0) |
| **Related Documents** | - `01_PRD_PHASE1_DATA_PREP_v2.md`<br>- `src-imap/docs_embedding/README.md`<br>- `src/METHODOLOGY.md` |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Context](#2-project-context)
3. [Functional Requirements](#3-functional-requirements)
4. [Technical Architecture](#4-technical-architecture)
5. [Data Flow Specifications](#5-data-flow-specifications)
6. [Module Specifications](#6-module-specifications)
7. [Configuration & Environment](#7-configuration--environment)
8. [Output Specifications](#8-output-specifications)
9. [Quality Assurance & Validation](#9-quality-assurance--validation)
10. [Success Criteria](#10-success-criteria)
11. [Implementation Plan](#11-implementation-plan)
12. [Performance Requirements](#12-performance-requirements)
13. [Error Handling & Recovery](#13-error-handling--recovery)
14. [Appendices](#14-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

Phase 2 transforms structured sales promotion documents from Phase 1 (JSONL format) into production-ready vector embeddings optimized for RAG (Retrieval-Augmented Generation) applications. This phase enables semantic search, intelligent filtering, and question-answering capabilities for Thai sales promotion data.

### 1.2 Scope

**In Scope:**
- Load 71 promotion documents from Phase 1 JSONL output
- Convert SimpleDocument → LlamaIndex Document objects
- Generate embeddings using OpenAI's text-embedding-3-small model
- Build DocumentSummaryIndex following production RAG patterns
- Create IndexNodes for hierarchical retrieval
- Generate LLM summaries for each promotion
- Preserve all 25 metadata fields in embeddings
- Save embeddings in multiple formats (JSON, PKL, NPY)
- Generate comprehensive processing statistics
- Validate embedding quality and metadata integrity

**Out of Scope:**
- Document chunking (promotions are atomic units)
- Section-based parsing (no complex document structure)
- Metadata extraction (completed in Phase 1)
- Retrieval strategy implementation (Phase 3)
- Query interface development (Phase 4)

### 1.3 Key Features

1. **Proven Pattern Alignment**: Follows `src-imap/docs_embedding/` architecture
2. **Atomic Document Handling**: Optimized for 50-150 token promotion documents
3. **Metadata Preservation**: All 25 fields maintained through pipeline
4. **Thai Language Support**: UTF-8 encoding for Thai product names and conditions
5. **Batch Processing**: Efficient processing with API rate limiting
6. **Multi-Format Output**: JSON (metadata), PKL (objects), NPY (vectors)
7. **Production-Ready**: DocumentSummaryIndex for hierarchical retrieval
8. **Quality Validation**: Comprehensive statistics and validation checks

### 1.4 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Documents Processed** | 71/71 (100%) | All promotions successfully embedded |
| **Metadata Preservation** | 25/25 fields | All metadata fields in embeddings |
| **Embedding Quality** | 1536 dimensions | OpenAI text-embedding-3-small |
| **Processing Success Rate** | 100% | No failed embeddings |
| **Thai Text Integrity** | 100% | UTF-8 encoding preserved |
| **Processing Time** | < 5 minutes | Total pipeline runtime |
| **Output Formats** | 3/3 (JSON, PKL, NPY) | All formats generated |

---

## 2. Project Context

### 2.1 Phase 1 Recap

Phase 1 successfully completed on **2025-10-26** with the following results:

**Input:**
- 71 sales promotion records from CSV

**Output:**
- `documents/documents.jsonl` - Primary output (71 promotion documents)
- `documents/markdown_files/promotion_001.md` to `promotion_071.md` - QA files
- `processing_report.json` - Statistics
- `config_used.yaml` - Configuration snapshot

**Data Structure:**
```json
{
  "id": "PROMO_001",
  "text": "Promotion for เอเย่นต์ customers: Buy 108 แพ็ค...",
  "metadata": {
    "document_id": "PROMO_001",
    "customer_type": "เอเย่นต์",
    "customer_tier": "tier_1",
    "product_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
    "product_category": "water",
    "product_brand": "ช้าง",
    "start_date_gregorian": "2025-10-01",
    "end_date_gregorian": "2025-12-31",
    "is_active": true,
    "discount_percentage": 28.0,
    // ... 25 total fields
  }
}
```

**Phase 1 Success:**
- ✅ 71/71 promotions processed (100%)
- ✅ 25 metadata fields per document
- ✅ Thai UTF-8 encoding preserved
- ✅ Buddhist calendar conversion (142 dates)
- ✅ JSONL format validated
- ✅ Markdown QA files generated

### 2.2 Phase 2 Objectives

**Primary Objectives:**
1. **Transform JSONL → Vector Embeddings**: Convert structured documents to semantic vectors
2. **Preserve Metadata**: Maintain all 25 fields for intelligent filtering
3. **Production-Ready Indexing**: Build DocumentSummaryIndex for hierarchical retrieval
4. **Multi-Format Storage**: Generate JSON, PKL, NPY outputs for flexibility
5. **Quality Assurance**: Validate embeddings and metadata integrity

**Secondary Objectives:**
- Establish reusable module architecture for future batches
- Generate comprehensive processing statistics
- Optimize for atomic document characteristics
- Enable Phase 3 retrieval strategy development

### 2.3 Proven Pattern Alignment

This phase follows **production-grade patterns** from:

#### 2.3.1 Primary Reference: `src-imap/docs_embedding/`

**Architecture Pattern:**
```python
src-imap/docs_embedding/
├── batch_embedding.py           # Main orchestrator
├── document_loader.py           # Document loading
├── metadata_extractor.py        # Metadata enhancement
├── embedding_processor.py       # DocumentSummaryIndex builder
├── file_storage.py              # Multi-format output
└── __init__.py                  # Package exports
```

**Key Patterns to Follow:**
1. **Modular architecture** with clear separation of concerns
2. **DocumentSummaryIndex** for production RAG applications
3. **Three embedding types** (IndexNodes, Chunks, Summaries)
4. **Batch processing** with configurable sizes and delays
5. **Multi-format output** (JSON, PKL, NPY)
6. **Comprehensive statistics** and validation
7. **LLM-generated summaries** for hierarchical retrieval

#### 2.3.2 Secondary Reference: `src/09_enhanced_batch_embeddings.py`

**Alternative Patterns:**
- Metadata extraction patterns (not needed - Phase 1 complete)
- Batch grouping and progress tracking
- Statistics generation and analysis
- Error handling and recovery

#### 2.3.3 Adaptation for Sales Promotions

**Differences from Reference Implementations:**

| Aspect | src-imap/ | Sales Promotions (Phase 2) |
|--------|-----------|----------------------------|
| **Input Format** | Markdown files | JSONL file |
| **Document Size** | 2000-5000 tokens | 50-150 tokens (atomic) |
| **Chunking** | Section-based (512 tokens) | **NONE** (atomic documents) |
| **Metadata Source** | Extracted via regex | **Pre-extracted** (Phase 1) |
| **Document Count** | 156 land deeds | 71 promotions |
| **Embedding Types** | 3 types (IndexNode, Chunk, Summary) | **2 types** (IndexNode, Summary) |
| **Chunking Strategy** | Section parser | **Skip chunking** |

**Similarities (Patterns to Maintain):**
- ✅ DocumentSummaryIndex for production RAG
- ✅ Batch processing with rate limiting
- ✅ Multi-format output (JSON, PKL, NPY)
- ✅ Metadata preservation throughout pipeline
- ✅ LLM-generated summaries
- ✅ Comprehensive statistics
- ✅ Modular architecture

---

## 3. Functional Requirements

### 3.1 Input Requirements

#### 3.1.1 Primary Input: JSONL File

**Location:** `src-sale-promotion/data/processed/documents/documents.jsonl`

**Format Specification:**
```json
{
  "id": "PROMO_{001-071}",
  "text": "Promotion for {customer_type} customers: Buy {buy_quantity}...",
  "metadata": {
    // 25 metadata fields (see Phase 1 PRD v2.0)
  }
}
```

**Validation Requirements:**
- ✅ File must exist and be readable
- ✅ Must contain exactly 71 lines (one per promotion)
- ✅ Each line must be valid JSON
- ✅ Each document must have `id`, `text`, `metadata` fields
- ✅ Metadata must contain all 25 required fields
- ✅ Thai text must be UTF-8 encoded
- ✅ No duplicate document IDs

#### 3.1.2 Configuration Input

**Location:** `.env` file in project root

**Required Variables:**
```bash
OPENAI_API_KEY=sk-...                    # Required
```

**Optional Variables (with defaults):**
```bash
CHUNK_SIZE=2048                          # Larger than doc size (no chunking)
CHUNK_OVERLAP=0                          # No overlap for atomic docs
EMBEDDING_MODEL=text-embedding-3-small   # OpenAI embedding model
LLM_MODEL=gpt-4o-mini                    # For summary generation
BATCH_SIZE=20                            # Promotions per batch
DELAY_BETWEEN_BATCHES=3                  # Seconds between batches
MAX_RETRIES=3                            # API retry attempts
```

#### 3.1.3 Directory Structure

**Expected Input Structure:**
```
src-sale-promotion/
├── data/
│   └── processed/
│       └── documents/
│           └── documents.jsonl          # PRIMARY INPUT
└── .env                                 # Configuration
```

### 3.2 Processing Requirements

#### 3.2.1 Document Loading

**FR-2.1.1: JSONL Document Loader**
- **Description**: Load promotion documents from JSONL file
- **Input**: `documents.jsonl` (71 lines)
- **Output**: List of SimpleDocument objects
- **Requirements**:
  - Parse each line as JSON
  - Validate document structure
  - Handle UTF-8 Thai text encoding
  - Detect and report malformed documents
  - Preserve all metadata fields
  - Maintain document order

**FR-2.1.2: SimpleDocument → LlamaIndex Document Conversion**
- **Description**: Convert Phase 1 SimpleDocument to LlamaIndex Document
- **Input**: SimpleDocument objects
- **Output**: LlamaIndex Document objects
- **Requirements**:
  - Transfer `text` field to Document.text
  - Transfer `metadata` dict to Document.metadata
  - Transfer `id` field to Document.doc_id
  - Preserve Thai text encoding
  - Validate conversion completeness
  - No data loss during conversion

**Code Example:**
```python
def convert_simple_to_llamaindex(simple_doc: dict) -> Document:
    """Convert SimpleDocument from JSONL to LlamaIndex Document."""
    return Document(
        text=simple_doc['text'],
        metadata=simple_doc['metadata'],
        doc_id=simple_doc['id']
    )
```

#### 3.2.2 Embedding Generation

**FR-2.2.1: DocumentSummaryIndex Construction**
- **Description**: Build DocumentSummaryIndex following LlamaIndex production patterns
- **Input**: List of LlamaIndex Document objects
- **Output**: DocumentSummaryIndex object
- **Requirements**:
  - Use `DocumentSummaryIndex.from_documents()` method
  - Configure OpenAI embedding model (text-embedding-3-small)
  - Configure OpenAI LLM for summaries (gpt-4o-mini)
  - Skip chunking (set chunk_size > max document size)
  - Generate LLM summaries for each promotion
  - Preserve all metadata in index
  - Handle API rate limits with retries
  - Show progress during processing

**FR-2.2.2: IndexNode Generation**
- **Description**: Create IndexNodes for hierarchical retrieval
- **Input**: DocumentSummaryIndex object
- **Output**: List of IndexNode objects
- **Requirements**:
  - One IndexNode per promotion
  - Include document summary in IndexNode.text
  - Include document title in IndexNode.text
  - Preserve all 25 metadata fields
  - Add index-specific metadata:
    - `doc_id`: Original document ID
    - `doc_title`: Promotion title
    - `type`: "document_summary"
    - `batch_number`: Batch identifier
  - Truncate long summaries (max 500 chars for display)

**Code Example:**
```python
index_node = IndexNode(
    text=f"Promotion: {promo_title}\n\nSummary: {summary}",
    index_id=f"promo_{i}",
    metadata={
        **original_metadata,  # All 25 fields
        "doc_id": doc_id,
        "doc_title": promo_title,
        "type": "document_summary",
        "batch_number": batch_num
    }
)
```

**FR-2.2.3: Embedding Extraction**
- **Description**: Extract embeddings from IndexNodes and Summaries
- **Input**: IndexNodes, DocumentSummaryIndex
- **Output**: Two embedding collections (IndexNode, Summary)
- **Requirements**:
  - **IndexNode embeddings**: Embed IndexNode.text using OpenAI
  - **Summary embeddings**: Embed LLM-generated summaries
  - **NO chunk embeddings**: Documents are atomic (no chunks)
  - Preserve all metadata in each embedding
  - Track embedding dimensions (1536 for text-embedding-3-small)
  - Handle API errors with retries
  - Log progress for each embedding

**Embedding Data Structure:**
```python
{
    "node_id": "promo_001",
    "text": "Promotion: PROMO_001\n\nSummary: ...",
    "text_length": 342,
    "embedding_vector": [0.123, -0.456, ...],  # 1536 dimensions
    "embedding_dim": 1536,
    "metadata": {
        # All 25 fields from Phase 1
        "document_id": "PROMO_001",
        "customer_type": "เอเย่นต์",
        # ... remaining 23 fields
        "type": "indexnode",
        "batch_number": 1
    },
    "type": "indexnode",
    "batch_number": 1
}
```

#### 3.2.3 Batch Processing

**FR-2.3.1: Batch Grouping**
- **Description**: Process documents in configurable batches
- **Input**: 71 promotion documents
- **Output**: N batches (71 / BATCH_SIZE)
- **Requirements**:
  - Default batch size: 20 promotions
  - Configurable via environment variable
  - Group documents sequentially (PROMO_001-020, PROMO_021-040, etc.)
  - Track batch numbers (batch_1, batch_2, etc.)
  - Handle partial last batch (e.g., 71 docs = 3.55 batches → 4 batches)

**FR-2.3.2: Rate Limiting**
- **Description**: Implement delays between batches for API rate limiting
- **Requirements**:
  - Default delay: 3 seconds between batches
  - Configurable via environment variable
  - No delay after final batch
  - Log delay progress
  - Allow manual override for testing

**FR-2.3.3: Batch Progress Tracking**
- **Description**: Display progress for each batch
- **Requirements**:
  - Print batch number and file count
  - Show embedding extraction progress
  - Display batch completion time
  - Aggregate statistics across batches
  - Log errors per batch

### 3.3 Output Requirements

#### 3.3.1 Multi-Format Embedding Storage

**FR-3.1.1: JSON Format (Human-Readable Metadata)**
- **Purpose**: Human-readable metadata with embedding previews
- **Format**: JSON with UTF-8 encoding
- **Contents**:
  - All metadata fields
  - Text content
  - Embedding preview (first 5 dimensions)
  - Embedding statistics
- **Example**: `batch_1_indexnodes_metadata.json`

**FR-3.1.2: PKL Format (Python Objects)**
- **Purpose**: Complete embeddings for Python loading
- **Format**: Python pickle
- **Contents**:
  - Full embedding vectors (1536 dimensions)
  - All metadata
  - Text content
  - Serialized Python objects
- **Example**: `batch_1_indexnodes_full.pkl`

**FR-3.1.3: NPY Format (Numpy Vectors)**
- **Purpose**: Vector arrays for numerical operations
- **Format**: Numpy binary format
- **Contents**:
  - Embedding vectors only (no metadata)
  - Shape: (N, 1536) where N = number of embeddings
  - Float32 or Float64 precision
- **Example**: `batch_1_indexnodes_vectors.npy`

#### 3.3.2 Output Directory Structure

```
src-sale-promotion/data/embedding/embeddings_sales_YYYYMMDD_HHMMSS/
├── batch_1/
│   ├── indexnodes/
│   │   ├── batch_1_indexnodes_metadata.json      # Metadata + preview
│   │   ├── batch_1_indexnodes_full.pkl           # Complete Python objects
│   │   └── batch_1_indexnodes_vectors.npy        # Vectors only
│   ├── summaries/
│   │   ├── batch_1_summaries_metadata.json
│   │   ├── batch_1_summaries_full.pkl
│   │   └── batch_1_summaries_vectors.npy
│   ├── combined/
│   │   ├── batch_1_all_metadata.json             # All embeddings combined
│   │   ├── batch_1_all_full.pkl
│   │   └── batch_1_all_vectors.npy
│   └── batch_1_statistics.json                   # Batch processing stats
├── batch_2/
│   └── [same structure]
├── batch_3/
│   └── [same structure]
├── batch_4/
│   └── [same structure]
└── combined_statistics.json                       # Overall statistics
```

#### 3.3.3 Statistics Output

**FR-3.3.1: Batch Statistics**
- **File**: `batch_N/batch_N_statistics.json`
- **Contents**:
  ```json
  {
    "batch_number": 1,
    "extraction_timestamp": "2025-10-26T17:30:00",
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "totals": {
      "indexnode_embeddings": 20,
      "summary_embeddings": 20,
      "total_embeddings": 40
    },
    "processing_time_seconds": 45.2,
    "metadata_analysis": {
      "total_metadata_fields": 27,
      "metadata_fields": ["document_id", "customer_type", ...]
    }
  }
  ```

**FR-3.3.2: Combined Statistics**
- **File**: `combined_statistics.json`
- **Contents**:
  ```json
  {
    "processing_timestamp": "2025-10-26T17:35:00",
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "total_batches": 4,
    "grand_totals": {
      "total_documents": 71,
      "indexnode_embeddings": 71,
      "summary_embeddings": 71,
      "total_embeddings": 142,
      "chunk_embeddings": 0
    },
    "processing_time_total_seconds": 180,
    "success_rate": 100.0,
    "metadata_analysis": {
      "total_unique_fields": 27,
      "fields_per_document": 25
    },
    "batch_breakdown": [
      {"batch": 1, "documents": 20, "embeddings": 40},
      {"batch": 2, "documents": 20, "embeddings": 40},
      {"batch": 3, "documents": 20, "embeddings": 40},
      {"batch": 4, "documents": 11, "embeddings": 22}
    ]
  }
  ```

### 3.4 Validation Requirements

**FR-3.4.1: Input Validation**
- Validate JSONL file exists and is readable
- Validate exactly 71 documents
- Validate each document has required fields
- Validate metadata completeness (25 fields)
- Validate Thai text UTF-8 encoding
- Validate no duplicate document IDs

**FR-3.4.2: Processing Validation**
- Validate API key before processing
- Validate embedding dimensions (1536)
- Validate metadata preservation
- Validate no data loss in conversions
- Validate batch grouping correctness

**FR-3.4.3: Output Validation**
- Validate all output files created
- Validate embedding vector dimensions
- Validate metadata field count
- Validate Thai text in JSON output
- Validate pickle objects are loadable
- Validate numpy arrays have correct shape

---

## 4. Technical Architecture

### 4.1 Module Overview

Following **`src-imap/docs_embedding/`** architecture pattern:

```
src-sale-promotion/docs_embedding/
├── __init__.py                     # Package exports
├── batch_embedding.py              # Main orchestrator (PRIMARY)
├── document_loader.py              # JSONL document loader
├── embedding_processor.py          # DocumentSummaryIndex builder
├── file_storage.py                 # Multi-format output manager
└── README.md                       # Module documentation
```

### 4.2 Module Responsibilities

#### 4.2.1 `batch_embedding.py` (Main Orchestrator)

**Responsibility**: Pipeline coordination and execution

**Key Classes:**
```python
class SalesPromotionBatchEmbeddingPipeline:
    """Main pipeline orchestrator for sales promotion embeddings."""

    def __init__(self, config: Dict[str, Any] = CONFIG):
        self.config = config
        self.document_loader = PromotionDocumentLoader()
        self.embedding_processor = EmbeddingProcessor()
        self.storage = EmbeddingStorage()
        self.api_key = self._validate_api_key()

    def run(self) -> None:
        """Execute the complete embedding pipeline."""
        # 1. Load documents from JSONL
        # 2. Group into batches
        # 3. Process each batch
        # 4. Save embeddings
        # 5. Generate statistics

    def process_batch(self, batch_docs: List[Document], batch_num: int):
        """Process a single batch of promotions."""
        # Build DocumentSummaryIndex
        # Extract IndexNode embeddings
        # Extract Summary embeddings
        # Return embedding collections

    def _build_index_nodes(self, doc_summary_index, batch_num):
        """Create IndexNodes for hierarchical retrieval."""
        # One IndexNode per promotion
        # Include summary + metadata
```

**Configuration:**
```python
CONFIG = {
    "input_file": Path("../data/processed/documents/documents.jsonl"),
    "output_dir": Path("../data/embedding"),
    "chunk_size": 2048,              # > max doc size (no chunking)
    "chunk_overlap": 0,               # No overlap for atomic docs
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "batch_size": 20,
    "delay_between_batches": 3,
    "summary_truncate_length": 500,
    "max_retries": 3
}
```

#### 4.2.2 `document_loader.py` (JSONL Loader)

**Responsibility**: Load and convert JSONL documents

**Key Classes:**
```python
class PromotionDocumentLoader:
    """Loads sales promotion documents from JSONL format."""

    def load_documents_from_jsonl(self, jsonl_path: Path) -> List[Document]:
        """
        Load all promotions from JSONL file.

        Returns:
            List of LlamaIndex Document objects with metadata
        """
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                doc_data = json.loads(line)
                doc = self._convert_to_llamaindex_document(doc_data)
                documents.append(doc)
        return documents

    def _convert_to_llamaindex_document(self, doc_data: dict) -> Document:
        """Convert SimpleDocument (JSONL) to LlamaIndex Document."""
        return Document(
            text=doc_data['text'],
            metadata=doc_data['metadata'],
            doc_id=doc_data['id']
        )

    def validate_jsonl_file(self, jsonl_path: Path) -> Dict[str, Any]:
        """Validate JSONL file structure and completeness."""
        # Check file exists
        # Count lines
        # Validate JSON format
        # Check required fields
        # Return validation report
```

#### 4.2.3 `embedding_processor.py` (Index Builder)

**Responsibility**: Build DocumentSummaryIndex and extract embeddings

**Key Classes:**
```python
class EmbeddingProcessor:
    """Processes documents and extracts embeddings."""

    def build_document_summary_index(
        self,
        documents: List[Document],
        llm: OpenAI,
        embed_model: OpenAIEmbedding
    ) -> DocumentSummaryIndex:
        """
        Build DocumentSummaryIndex for atomic promotion documents.

        Note: Chunking is disabled (chunk_size > doc size) since
        promotions are atomic semantic units (50-150 tokens).
        """
        splitter = SentenceSplitter(
            chunk_size=2048,  # Larger than max doc size
            chunk_overlap=0    # No overlap needed
        )

        doc_summary_index = DocumentSummaryIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            transformations=[splitter],
            show_progress=True
        )

        return doc_summary_index

    def extract_indexnode_embeddings(
        self,
        index_nodes: List[IndexNode],
        embed_model: OpenAIEmbedding,
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings from IndexNodes."""
        # Manually embed each IndexNode.text
        # Preserve all metadata
        # Return embedding collection

    def extract_summary_embeddings(
        self,
        doc_summary_index: DocumentSummaryIndex,
        embed_model: OpenAIEmbedding,
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings from document summaries."""
        # Get LLM-generated summaries
        # Embed each summary
        # Preserve metadata
```

#### 4.2.4 `file_storage.py` (Output Manager)

**Responsibility**: Save embeddings in multiple formats

**Key Classes:**
```python
class EmbeddingStorage:
    """Manages multi-format embedding storage."""

    def save_batch_embeddings(
        self,
        output_dir: Path,
        batch_number: int,
        indexnode_embeddings: List[Dict],
        summary_embeddings: List[Dict]
    ) -> None:
        """
        Save batch embeddings in multiple formats.

        Formats:
        - JSON: Metadata + embedding preview
        - PKL: Complete Python objects
        - NPY: Embedding vectors only
        """
        batch_dir = output_dir / f"batch_{batch_number}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save indexnodes
        self._save_embedding_collection(
            batch_dir / "indexnodes",
            f"batch_{batch_number}_indexnodes",
            indexnode_embeddings
        )

        # Save summaries
        self._save_embedding_collection(
            batch_dir / "summaries",
            f"batch_{batch_number}_summaries",
            summary_embeddings
        )

        # Save combined
        all_embeddings = indexnode_embeddings + summary_embeddings
        self._save_embedding_collection(
            batch_dir / "combined",
            f"batch_{batch_number}_all",
            all_embeddings
        )

    def _save_embedding_collection(
        self,
        output_dir: Path,
        name: str,
        embeddings: List[Dict]
    ) -> None:
        """Save single embedding collection in 3 formats."""
        # 1. JSON: Metadata + preview
        # 2. PKL: Complete objects
        # 3. NPY: Vectors only

    def save_batch_statistics(
        self,
        batch_dir: Path,
        batch_number: int,
        embeddings: Tuple[List, List]
    ) -> None:
        """Generate and save batch statistics."""

    def save_combined_statistics(
        self,
        output_dir: Path,
        all_batches_data: List[Tuple]
    ) -> None:
        """Generate and save combined statistics across all batches."""
```

### 4.3 Processing Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2: EMBEDDING PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: documents.jsonl (71 promotions)                         │
│  - SimpleDocument format from Phase 1                           │
│  - 25 metadata fields per promotion                             │
│  - Thai UTF-8 text (50-150 tokens)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Load & Convert (document_loader.py)                   │
│  - Parse JSONL file                                             │
│  - Convert SimpleDocument → LlamaIndex Document                 │
│  - Validate 71 documents                                        │
│  - Preserve all metadata                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Batch Grouping (batch_embedding.py)                   │
│  - Group into batches (default: 20 docs/batch)                 │
│  - 71 docs → 4 batches (20+20+20+11)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Build Index (embedding_processor.py)                  │
│  FOR EACH BATCH:                                                │
│  - Configure LLM (gpt-4o-mini)                                  │
│  - Configure Embeddings (text-embedding-3-small)                │
│  - Build DocumentSummaryIndex                                   │
│    • NO CHUNKING (docs are atomic)                              │
│    • Generate LLM summaries                                     │
│    • Preserve metadata                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Create IndexNodes (batch_embedding.py)                │
│  - One IndexNode per promotion                                  │
│  - Include promotion title + summary                            │
│  - Preserve all 25 metadata fields                              │
│  - Add batch_number, doc_id, type                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Extract Embeddings (embedding_processor.py)           │
│  Two Embedding Types:                                           │
│  1. IndexNode Embeddings (71 total)                             │
│     - Embed IndexNode.text                                      │
│     - 1536 dimensions                                           │
│     - All metadata preserved                                    │
│  2. Summary Embeddings (71 total)                               │
│     - Embed LLM-generated summary                               │
│     - 1536 dimensions                                           │
│     - All metadata preserved                                    │
│                                                                  │
│  Total: 142 embeddings (71 × 2)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Save Outputs (file_storage.py)                        │
│  FOR EACH BATCH:                                                │
│  - Create batch directory                                       │
│  - Save indexnodes/ (JSON, PKL, NPY)                            │
│  - Save summaries/ (JSON, PKL, NPY)                             │
│  - Save combined/ (JSON, PKL, NPY)                              │
│  - Save batch statistics                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Generate Statistics                                    │
│  - Combined statistics across all batches                       │
│  - Metadata field analysis                                      │
│  - Processing metrics                                           │
│  - Validation results                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Multi-format Embeddings                                │
│  - 4 batch directories                                          │
│  - 142 total embeddings (71 IndexNode + 71 Summary)             │
│  - JSON (metadata), PKL (objects), NPY (vectors)                │
│  - Combined statistics report                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Data Flow: SimpleDocument → Embeddings

```python
# Phase 1 Output (SimpleDocument in JSONL)
{
  "id": "PROMO_001",
  "text": "Promotion for เอเย่นต์ customers: ...",
  "metadata": {
    "document_id": "PROMO_001",
    "customer_type": "เอเย่นต์",
    # ... 25 fields total
  }
}
                    ↓
# Convert to LlamaIndex Document
Document(
  text="Promotion for เอเย่นต์ customers: ...",
  metadata={
    "document_id": "PROMO_001",
    "customer_type": "เอเย่นต์",
    # ... 25 fields
  },
  doc_id="PROMO_001"
)
                    ↓
# Build DocumentSummaryIndex (NO CHUNKING)
doc_summary_index = DocumentSummaryIndex.from_documents(
  documents,
  chunk_size=2048,  # > max doc size
  chunk_overlap=0
)
                    ↓
# Create IndexNode
IndexNode(
  text="Promotion: PROMO_001\n\nSummary: Buy 108 แพ็ค...",
  index_id="promo_1",
  metadata={
    **original_metadata,  # All 25 fields
    "doc_id": "PROMO_001",
    "doc_title": "Promotion: PROMO_001",
    "type": "document_summary",
    "batch_number": 1
  }
)
                    ↓
# Extract Embeddings
{
  "node_id": "promo_1",
  "text": "Promotion: PROMO_001\n\nSummary: ...",
  "embedding_vector": [0.123, -0.456, ...],  # 1536 dims
  "metadata": {
    # All 25 original fields + added fields
  },
  "type": "indexnode"
}
```

---

## 5. Data Flow Specifications

### 5.1 Input Data Format

#### 5.1.1 JSONL Structure

**File**: `documents.jsonl`
**Lines**: 71 (one per promotion)
**Encoding**: UTF-8
**Format**: Newline-delimited JSON

**Per-Line Structure:**
```json
{
  "id": "PROMO_{001-071}",
  "text": "Promotion for {customer_type} customers: Buy {buy_quantity} {buy_unit} of {product_name} and get {free_quantity} {free_unit} of {free_item_name} FREE. Promotion Period: Valid from {start_date} to {end_date} ({duration} days). Status: {status}. Promotion Terms: - Purchase minimum: {buy_quantity} {buy_unit} - Free items: {free_quantity} {free_unit} - Effective discount: {discount}% off total quantity - Customer tier: {customer_tier} ({is_wholesale/is_retail}). Product Information: - Product: {product_name} - Category: {product_category} - Brand: {product_brand} - Size: {product_size}. {additional_conditions}",
  "metadata": {
    "document_id": "PROMO_001",
    "customer_type": "เอเย่นต์",
    "customer_tier": "tier_1",
    "is_wholesale": true,
    "is_retail": false,
    "product_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
    "product_category": "water",
    "product_brand": "ช้าง",
    "product_size": "600 CC",
    "buy_quantity": 108,
    "buy_unit": "แพ็ค",
    "free_item_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
    "free_quantity": 42,
    "free_unit": "แพ็ค",
    "discount_percentage": 28.0,
    "promotion_type": "buy_x_get_y",
    "start_date": "2568-10-01 00:00:00",
    "end_date": "2568-12-31 00:00:00",
    "start_date_gregorian": "2025-10-01",
    "end_date_gregorian": "2025-12-31",
    "is_active": true,
    "expires_soon": false,
    "duration_days": 92,
    "additional_conditions": null,
    "keywords": ["น้ำดื่มช้าง", "PET", "600", "CC", "แพ็ค", "เท่ากับ", "12", "ขวด", "water", "ช้าง", "เอเย่นต์"]
  }
}
```

**Text Field Characteristics:**
- **Length**: 508-601 characters (avg: 555 chars)
- **Tokens**: ~50-150 tokens (estimated)
- **Language**: Mixed Thai and English
- **Encoding**: UTF-8
- **Structure**: Natural language description with embedded metadata

**Metadata Field Characteristics:**
- **Count**: 25 fields per document
- **Types**: String, numeric, boolean, list, datetime
- **Required**: All 25 fields present (validation in Phase 1)
- **Encoding**: UTF-8 for Thai fields

### 5.2 Intermediate Data Structures

#### 5.2.1 LlamaIndex Document

```python
from llama_index.core import Document

document = Document(
    text="Promotion for เอเย่นต์ customers: Buy 108 แพ็ค...",
    metadata={
        "document_id": "PROMO_001",
        "customer_type": "เอเย่นต์",
        # ... all 25 fields
    },
    doc_id="PROMO_001"
)

# Properties
document.text            # str: Full promotion description
document.metadata        # Dict[str, Any]: 25 metadata fields
document.doc_id          # str: "PROMO_001"
document.get_content()   # str: Same as .text
```

#### 5.2.2 DocumentSummaryIndex

```python
from llama_index.core import DocumentSummaryIndex

doc_summary_index = DocumentSummaryIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    transformations=[splitter],
    show_progress=True
)

# Internal structure
doc_summary_index.ref_doc_info     # Dict[doc_id, RefDocInfo]
doc_summary_index.docstore         # Docstore with all nodes
doc_summary_index.get_document_summary(doc_id)  # LLM-generated summary
```

**Expected Behavior for Atomic Documents:**
- Since `chunk_size=2048` > max document size (~150 tokens = ~600 chars)
- No chunking occurs (each document becomes single chunk)
- LLM generates summary for each promotion
- Summaries are shorter, focused versions of promotion text

#### 5.2.3 IndexNode

```python
from llama_index.core.schema import IndexNode

index_node = IndexNode(
    text=f"Promotion: {promo_title}\n\nSummary: {summary}",
    index_id=f"promo_{i}",
    metadata={
        # Original 25 fields
        "document_id": "PROMO_001",
        "customer_type": "เอเย่นต์",
        "customer_tier": "tier_1",
        # ... remaining 22 fields

        # Added fields
        "doc_id": "PROMO_001",
        "doc_title": "Promotion: PROMO_001",
        "type": "document_summary",
        "batch_number": 1
    }
)

# Properties
index_node.text          # Combined title + summary
index_node.index_id      # "promo_1"
index_node.metadata      # Dict with 27 fields (25 original + 2 added)
```

### 5.3 Output Data Formats

#### 5.3.1 Embedding Data Structure

```python
embedding_data = {
    # Identification
    "node_id": "promo_001",
    "doc_id": "PROMO_001",
    "doc_title": "Promotion: PROMO_001",
    "index_id": "promo_1",

    # Text content
    "text": "Promotion: PROMO_001\n\nSummary: Buy 108 แพ็ค...",
    "text_length": 342,

    # Embedding vector
    "embedding_vector": [0.123, -0.456, 0.789, ...],  # 1536 floats
    "embedding_dim": 1536,

    # Metadata (all 25 original fields + added fields)
    "metadata": {
        # Phase 1 fields (25)
        "document_id": "PROMO_001",
        "customer_type": "เอเย่นต์",
        "customer_tier": "tier_1",
        "is_wholesale": true,
        "is_retail": false,
        "product_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
        "product_category": "water",
        "product_brand": "ช้าง",
        "product_size": "600 CC",
        "buy_quantity": 108,
        "buy_unit": "แพ็ค",
        "free_item_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
        "free_quantity": 42,
        "free_unit": "แพ็ค",
        "discount_percentage": 28.0,
        "promotion_type": "buy_x_get_y",
        "start_date": "2568-10-01 00:00:00",
        "end_date": "2568-12-31 00:00:00",
        "start_date_gregorian": "2025-10-01",
        "end_date_gregorian": "2025-12-31",
        "is_active": true,
        "expires_soon": false,
        "duration_days": 92,
        "additional_conditions": null,
        "keywords": ["น้ำดื่มช้าง", "PET", "600", "CC", ...],

        # Added fields (Phase 2)
        "doc_id": "PROMO_001",
        "doc_title": "Promotion: PROMO_001",
        "type": "indexnode",
        "batch_number": 1
    },

    # Processing metadata
    "type": "indexnode",
    "batch_number": 1
}
```

#### 5.3.2 JSON Output Format

**File**: `batch_1_indexnodes_metadata.json`

```json
[
  {
    "node_id": "promo_001",
    "doc_id": "PROMO_001",
    "doc_title": "Promotion: PROMO_001",
    "text": "Promotion: PROMO_001\n\nSummary: Buy 108 แพ็ค...",
    "text_length": 342,
    "embedding_preview": [0.123, -0.456, 0.789, 0.234, -0.567],
    "embedding_dim": 1536,
    "metadata": {
      "document_id": "PROMO_001",
      "customer_type": "เอเย่นต์",
      // ... all 25 original fields
      "type": "indexnode",
      "batch_number": 1
    },
    "type": "indexnode",
    "batch_number": 1
  },
  // ... remaining 19 promotions in batch
]
```

**Characteristics:**
- Human-readable JSON with indentation
- UTF-8 encoding for Thai text
- Embedding preview (first 5 dimensions only)
- Full metadata included
- Used for inspection and debugging

#### 5.3.3 PKL Output Format

**File**: `batch_1_indexnodes_full.pkl`

```python
# Pickle file contents (binary)
embeddings_list = [
    {
        "node_id": "promo_001",
        "embedding_vector": [0.123, -0.456, ...],  # Full 1536 dims
        "metadata": {...},  # Full metadata
        # ... all fields
    },
    # ... remaining embeddings
]

# Loading
import pickle
with open('batch_1_indexnodes_full.pkl', 'rb') as f:
    embeddings = pickle.load(f)
```

**Characteristics:**
- Binary format (not human-readable)
- Complete embedding vectors (1536 dimensions)
- All metadata fields
- Fast Python loading
- Used for Phase 3 retrieval loading

#### 5.3.4 NPY Output Format

**File**: `batch_1_indexnodes_vectors.npy`

```python
# Numpy array
import numpy as np

vectors = np.array([
    [0.123, -0.456, 0.789, ...],  # PROMO_001 embedding (1536 dims)
    [0.234, -0.567, 0.890, ...],  # PROMO_002 embedding (1536 dims)
    # ... remaining embeddings
])

# Shape: (N, 1536) where N = number of embeddings
# Dtype: float32 or float64

# Loading
vectors = np.load('batch_1_indexnodes_vectors.npy')
print(vectors.shape)  # (20, 1536) for batch 1
```

**Characteristics:**
- Binary numpy format
- Vectors only (no metadata, no text)
- Shape: (N, 1536)
- Fast numerical operations
- Used for vector similarity computations

---

## 6. Module Specifications

### 6.1 Module: `document_loader.py`

#### 6.1.1 Purpose
Load sales promotion documents from JSONL format and convert to LlamaIndex Documents.

#### 6.1.2 Dependencies
```python
import json
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document
```

#### 6.1.3 Class Specification

```python
class PromotionDocumentLoader:
    """
    Loads sales promotion documents from JSONL file.

    Converts Phase 1 SimpleDocument format (JSONL) to LlamaIndex Document objects.
    Validates document structure and metadata completeness.
    """

    def __init__(self):
        """Initialize document loader."""
        self.documents_loaded = 0
        self.validation_errors = []

    def load_documents_from_jsonl(
        self,
        jsonl_path: Path
    ) -> List[Document]:
        """
        Load all promotion documents from JSONL file.

        Args:
            jsonl_path: Path to documents.jsonl file

        Returns:
            List of LlamaIndex Document objects

        Raises:
            FileNotFoundError: If JSONL file doesn't exist
            ValueError: If validation fails
        """
        # Validate file exists
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        documents = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON
                    doc_data = json.loads(line.strip())

                    # Validate structure
                    self._validate_document_structure(doc_data, line_num)

                    # Convert to LlamaIndex Document
                    doc = self._convert_to_llamaindex_document(doc_data)
                    documents.append(doc)

                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_num}: Invalid JSON - {e}"
                    self.validation_errors.append(error_msg)
                    raise ValueError(error_msg)

                except ValueError as e:
                    error_msg = f"Line {line_num}: {e}"
                    self.validation_errors.append(error_msg)
                    raise

        self.documents_loaded = len(documents)

        # Validate expected count
        if len(documents) != 71:
            raise ValueError(
                f"Expected 71 documents, but loaded {len(documents)}"
            )

        return documents

    def _validate_document_structure(
        self,
        doc_data: dict,
        line_num: int
    ) -> None:
        """
        Validate document has required fields.

        Args:
            doc_data: Document dictionary
            line_num: Line number in JSONL file

        Raises:
            ValueError: If validation fails
        """
        # Check required top-level fields
        required_fields = ['id', 'text', 'metadata']
        for field in required_fields:
            if field not in doc_data:
                raise ValueError(f"Missing required field: '{field}'")

        # Validate metadata field count
        metadata = doc_data['metadata']
        if len(metadata) < 25:
            raise ValueError(
                f"Insufficient metadata fields: {len(metadata)} "
                f"(expected 25+)"
            )

        # Validate text is non-empty
        if not doc_data['text'] or len(doc_data['text']) == 0:
            raise ValueError("Empty 'text' field")

    def _convert_to_llamaindex_document(
        self,
        doc_data: dict
    ) -> Document:
        """
        Convert SimpleDocument (JSONL) to LlamaIndex Document.

        Args:
            doc_data: Dictionary from JSONL line

        Returns:
            LlamaIndex Document object
        """
        return Document(
            text=doc_data['text'],
            metadata=doc_data['metadata'],
            doc_id=doc_data['id']
        )

    def validate_jsonl_file(self, jsonl_path: Path) -> Dict[str, Any]:
        """
        Validate JSONL file without loading full documents.

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            Validation report dictionary
        """
        report = {
            "file_exists": jsonl_path.exists(),
            "line_count": 0,
            "valid_json_lines": 0,
            "errors": []
        }

        if not report["file_exists"]:
            report["errors"].append("File not found")
            return report

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                report["line_count"] += 1

                try:
                    doc_data = json.loads(line.strip())

                    # Quick validation
                    if 'id' in doc_data and 'text' in doc_data and 'metadata' in doc_data:
                        report["valid_json_lines"] += 1
                    else:
                        report["errors"].append(
                            f"Line {line_num}: Missing required fields"
                        )

                except json.JSONDecodeError as e:
                    report["errors"].append(
                        f"Line {line_num}: Invalid JSON - {e}"
                    )

        report["is_valid"] = (
            report["line_count"] == 71 and
            report["valid_json_lines"] == 71 and
            len(report["errors"]) == 0
        )

        return report
```

#### 6.1.4 Usage Example

```python
from document_loader import PromotionDocumentLoader
from pathlib import Path

# Initialize loader
loader = PromotionDocumentLoader()

# Validate file first
jsonl_path = Path("../data/processed/documents/documents.jsonl")
validation_report = loader.validate_jsonl_file(jsonl_path)

if validation_report["is_valid"]:
    # Load documents
    documents = loader.load_documents_from_jsonl(jsonl_path)
    print(f"Loaded {len(documents)} promotion documents")

    # Inspect first document
    doc = documents[0]
    print(f"Document ID: {doc.doc_id}")
    print(f"Text length: {len(doc.text)}")
    print(f"Metadata fields: {len(doc.metadata)}")
else:
    print(f"Validation errors: {validation_report['errors']}")
```

### 6.2 Module: `embedding_processor.py`

#### 6.2.1 Purpose
Build DocumentSummaryIndex and extract embeddings following LlamaIndex production patterns.

#### 6.2.2 Dependencies
```python
from typing import List, Dict, Any
from llama_index.core import DocumentSummaryIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
```

#### 6.2.3 Class Specification

```python
class EmbeddingProcessor:
    """
    Processes promotion documents and extracts embeddings.

    Builds DocumentSummaryIndex and creates IndexNode embeddings
    following production RAG patterns from src-imap/.
    """

    def __init__(self):
        """Initialize embedding processor."""
        self.embeddings_extracted = 0

    def build_document_summary_index(
        self,
        documents: List[Document],
        llm: OpenAI,
        embed_model: OpenAIEmbedding,
        chunk_size: int = 2048,
        chunk_overlap: int = 0
    ) -> DocumentSummaryIndex:
        """
        Build DocumentSummaryIndex for atomic promotion documents.

        Note: Chunking is effectively disabled by setting chunk_size
        larger than maximum document size. Since promotions are 50-150
        tokens (~200-600 chars), chunk_size=2048 prevents chunking.

        Args:
            documents: List of LlamaIndex Document objects
            llm: OpenAI LLM for summary generation
            embed_model: OpenAI embedding model
            chunk_size: Chunk size (default: 2048, larger than docs)
            chunk_overlap: Chunk overlap (default: 0)

        Returns:
            DocumentSummaryIndex object
        """
        print(f"Building DocumentSummaryIndex for {len(documents)} promotions...")

        # Create sentence splitter (no actual chunking due to size)
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Create response synthesizer for summaries
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True
        )

        # Build DocumentSummaryIndex
        doc_summary_index = DocumentSummaryIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            transformations=[splitter],
            response_synthesizer=response_synthesizer,
            show_progress=True
        )

        print(f"✅ DocumentSummaryIndex built successfully")
        return doc_summary_index

    def extract_indexnode_embeddings(
        self,
        index_nodes: List[IndexNode],
        embed_model: OpenAIEmbedding,
        batch_number: int
    ) -> List[Dict]:
        """
        Extract embeddings from IndexNodes.

        Args:
            index_nodes: List of IndexNode objects
            embed_model: OpenAI embedding model
            batch_number: Batch identifier

        Returns:
            List of embedding dictionaries
        """
        print(f"\n📊 Extracting IndexNode embeddings...")

        indexnode_embeddings = []

        for i, node in enumerate(index_nodes, 1):
            try:
                print(f"  🔄 Embedding IndexNode {i}/{len(index_nodes)}: "
                      f"{node.metadata.get('doc_title', 'unknown')}")

                # Manually embed the IndexNode text
                embedding_vector = embed_model.get_text_embedding(node.text)

                embedding_data = {
                    "node_id": node.node_id,
                    "index_id": node.index_id,
                    "doc_title": node.metadata.get("doc_title", "unknown"),
                    "text": node.text,
                    "text_length": len(node.text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": dict(node.metadata),
                    "type": "indexnode",
                    "batch_number": batch_number
                }

                indexnode_embeddings.append(embedding_data)

                print(f"  ✅ Extracted embedding (dim: {len(embedding_vector)})")

            except Exception as e:
                print(f"  ❌ Error extracting IndexNode {i}: {str(e)}")

        self.embeddings_extracted += len(indexnode_embeddings)
        return indexnode_embeddings

    def extract_summary_embeddings(
        self,
        doc_summary_index: DocumentSummaryIndex,
        embed_model: OpenAIEmbedding,
        batch_number: int
    ) -> List[Dict]:
        """
        Extract embeddings from LLM-generated document summaries.

        Args:
            doc_summary_index: DocumentSummaryIndex object
            embed_model: OpenAI embedding model
            batch_number: Batch identifier

        Returns:
            List of summary embedding dictionaries
        """
        print(f"\n📋 Extracting document summary embeddings...")

        summary_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())

        for i, doc_id in enumerate(doc_ids, 1):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = doc_info.metadata.get('document_id', f'Document {i}')

                # Get LLM-generated summary
                doc_summary = doc_summary_index.get_document_summary(doc_id)

                print(f"  🔄 Embedding summary {i}/{len(doc_ids)}: {doc_title}")

                # Manually embed the summary text
                embedding_vector = embed_model.get_text_embedding(doc_summary)

                # Preserve original metadata
                summary_metadata = {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "batch_number": batch_number,
                    "type": "summary"
                }
                summary_metadata.update(doc_info.metadata)

                embedding_data = {
                    "node_id": f"summary_{doc_id}",
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "summary_text": doc_summary,
                    "summary_length": len(doc_summary),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": summary_metadata,
                    "type": "summary",
                    "batch_number": batch_number
                }

                summary_embeddings.append(embedding_data)

                print(f"  ✅ Extracted summary embedding (dim: {len(embedding_vector)})")

            except Exception as e:
                print(f"  ❌ Error extracting summary {i}: {str(e)}")

        self.embeddings_extracted += len(summary_embeddings)
        return summary_embeddings
```

### 6.3 Module: `file_storage.py`

#### 6.3.1 Purpose
Save embeddings in multiple formats and generate statistics.

#### 6.3.2 Dependencies
```python
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
```

#### 6.3.3 Class Specification

```python
class EmbeddingStorage:
    """
    Manages multi-format embedding storage and statistics generation.

    Saves embeddings in three formats:
    - JSON: Metadata with embedding preview (human-readable)
    - PKL: Complete Python objects (for loading)
    - NPY: Embedding vectors only (for numerical operations)
    """

    def __init__(self):
        """Initialize embedding storage manager."""
        pass

    def save_batch_embeddings(
        self,
        output_dir: Path,
        batch_number: int,
        indexnode_embeddings: List[Dict],
        summary_embeddings: List[Dict]
    ) -> None:
        """
        Save batch embeddings in multiple formats.

        Creates directory structure:
        batch_N/
        ├── indexnodes/
        ├── summaries/
        └── combined/

        Args:
            output_dir: Base output directory
            batch_number: Batch identifier
            indexnode_embeddings: IndexNode embedding collection
            summary_embeddings: Summary embedding collection
        """
        batch_dir = output_dir / f"batch_{batch_number}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving batch {batch_number} embeddings to: {batch_dir}")

        # Create subdirectories
        (batch_dir / "indexnodes").mkdir(exist_ok=True)
        (batch_dir / "summaries").mkdir(exist_ok=True)
        (batch_dir / "combined").mkdir(exist_ok=True)

        # Save indexnodes
        if indexnode_embeddings:
            self._save_embedding_collection(
                batch_dir / "indexnodes",
                f"batch_{batch_number}_indexnodes",
                indexnode_embeddings
            )

        # Save summaries
        if summary_embeddings:
            self._save_embedding_collection(
                batch_dir / "summaries",
                f"batch_{batch_number}_summaries",
                summary_embeddings
            )

        # Save combined
        all_embeddings = indexnode_embeddings + summary_embeddings
        if all_embeddings:
            self._save_embedding_collection(
                batch_dir / "combined",
                f"batch_{batch_number}_all",
                all_embeddings
            )

        # Save batch statistics
        self.save_batch_statistics(
            batch_dir,
            batch_number,
            indexnode_embeddings,
            summary_embeddings
        )

    def _save_embedding_collection(
        self,
        output_dir: Path,
        name: str,
        embeddings: List[Dict]
    ) -> None:
        """
        Save embedding collection in 3 formats.

        Args:
            output_dir: Directory to save files
            name: Base filename
            embeddings: List of embedding dictionaries
        """
        if not embeddings:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  💾 Saving {len(embeddings)} embeddings as '{name}'...")

        # 1. JSON: Metadata with embedding preview
        json_data = []
        for emb in embeddings:
            json_item = {k: v for k, v in emb.items() if k != 'embedding_vector'}
            json_item['embedding_preview'] = (
                emb['embedding_vector'][:5] if emb['embedding_vector'] else []
            )
            json_data.append(json_item)

        with open(output_dir / f"{name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # 2. PKL: Complete objects
        with open(output_dir / f"{name}_full.pkl", 'wb') as f:
            pickle.dump(embeddings, f)

        # 3. NPY: Vectors only
        vectors = [emb['embedding_vector'] for emb in embeddings if emb['embedding_vector']]
        if vectors:
            np.save(output_dir / f"{name}_vectors.npy", np.array(vectors))

        print(f"  ✅ Saved in 3 formats (JSON, PKL, NPY)")

    def save_batch_statistics(
        self,
        batch_dir: Path,
        batch_number: int,
        indexnode_embeddings: List[Dict],
        summary_embeddings: List[Dict]
    ) -> None:
        """
        Generate and save batch statistics.

        Args:
            batch_dir: Batch directory
            batch_number: Batch identifier
            indexnode_embeddings: IndexNode embeddings
            summary_embeddings: Summary embeddings
        """
        # Collect metadata fields
        all_metadata_fields = set()
        for emb_list in [indexnode_embeddings, summary_embeddings]:
            for emb in emb_list:
                if 'metadata' in emb and emb['metadata']:
                    all_metadata_fields.update(emb['metadata'].keys())

        stats = {
            "batch_number": batch_number,
            "extraction_timestamp": datetime.now().isoformat(),
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
            "totals": {
                "indexnode_embeddings": len(indexnode_embeddings),
                "summary_embeddings": len(summary_embeddings),
                "total_embeddings": len(indexnode_embeddings) + len(summary_embeddings)
            },
            "metadata_analysis": {
                "total_metadata_fields": len(all_metadata_fields),
                "metadata_fields": sorted(list(all_metadata_fields))
            }
        }

        with open(batch_dir / f"batch_{batch_number}_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"  📊 Saved batch statistics")

    def save_combined_statistics(
        self,
        output_dir: Path,
        all_batches_data: List[Tuple[List[Dict], List[Dict]]]
    ) -> None:
        """
        Generate and save combined statistics across all batches.

        Args:
            output_dir: Base output directory
            all_batches_data: List of (indexnode_embeddings, summary_embeddings) tuples
        """
        total_indexnodes = sum(len(batch[0]) for batch in all_batches_data)
        total_summaries = sum(len(batch[1]) for batch in all_batches_data)

        # Collect all metadata fields
        all_metadata_fields = set()
        for batch in all_batches_data:
            for emb_list in batch:
                for emb in emb_list:
                    if 'metadata' in emb and emb['metadata']:
                        all_metadata_fields.update(emb['metadata'].keys())

        combined_stats = {
            "processing_timestamp": datetime.now().isoformat(),
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
            "total_batches": len(all_batches_data),
            "grand_totals": {
                "total_documents": total_indexnodes,  # One IndexNode per document
                "indexnode_embeddings": total_indexnodes,
                "summary_embeddings": total_summaries,
                "total_embeddings": total_indexnodes + total_summaries,
                "chunk_embeddings": 0  # No chunks for atomic documents
            },
            "metadata_analysis": {
                "total_unique_metadata_fields": len(all_metadata_fields),
                "metadata_fields": sorted(list(all_metadata_fields))
            },
            "batch_breakdown": [
                {
                    "batch_number": i + 1,
                    "documents": len(batch[0]),
                    "indexnodes": len(batch[0]),
                    "summaries": len(batch[1]),
                    "total": len(batch[0]) + len(batch[1])
                }
                for i, batch in enumerate(all_batches_data)
            ]
        }

        with open(output_dir / "combined_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Saved combined statistics across {len(all_batches_data)} batches")
```

### 6.4 Module: `batch_embedding.py` (Main Orchestrator)

#### 6.4.1 Purpose
Coordinate the complete embedding pipeline from JSONL input to multi-format output.

#### 6.4.2 Implementation (Simplified Excerpt)

```python
class SalesPromotionBatchEmbeddingPipeline:
    """
    Main pipeline orchestrator for sales promotion embeddings.

    Follows src-imap/docs_embedding/ architecture patterns.
    """

    def __init__(self, config: Dict[str, Any] = CONFIG):
        self.config = config
        self.document_loader = PromotionDocumentLoader()
        self.embedding_processor = EmbeddingProcessor()
        self.storage = EmbeddingStorage()
        self.api_key = self._validate_api_key()

    def run(self) -> None:
        """Execute the complete embedding pipeline."""

        # 1. Load documents from JSONL
        documents = self.document_loader.load_documents_from_jsonl(
            self.config["input_file"]
        )

        # 2. Group into batches
        batches = self._create_batches(documents, self.config["batch_size"])

        # 3. Process each batch
        all_batches_data = []
        for batch_num, batch_docs in enumerate(batches, 1):
            embeddings = self.process_batch(batch_docs, batch_num)
            all_batches_data.append(embeddings)

            # Rate limiting
            if batch_num < len(batches):
                time.sleep(self.config["delay_between_batches"])

        # 4. Save combined statistics
        self.storage.save_combined_statistics(output_dir, all_batches_data)

    def process_batch(
        self,
        batch_docs: List[Document],
        batch_num: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process a single batch of promotions."""

        # Configure models
        llm = OpenAI(model=self.config["llm_model"], api_key=self.api_key)
        embed_model = OpenAIEmbedding(
            model=self.config["embedding_model"],
            api_key=self.api_key
        )

        # Build DocumentSummaryIndex
        doc_summary_index = self.embedding_processor.build_document_summary_index(
            batch_docs, llm, embed_model
        )

        # Create IndexNodes
        index_nodes = self._build_index_nodes(doc_summary_index, batch_num)

        # Extract embeddings
        indexnode_embeddings = self.embedding_processor.extract_indexnode_embeddings(
            index_nodes, embed_model, batch_num
        )
        summary_embeddings = self.embedding_processor.extract_summary_embeddings(
            doc_summary_index, embed_model, batch_num
        )

        # Save batch outputs
        self.storage.save_batch_embeddings(
            output_dir, batch_num, indexnode_embeddings, summary_embeddings
        )

        return (indexnode_embeddings, summary_embeddings)
```

---

## 7. Configuration & Environment

### 7.1 Environment Variables

**File**: `.env` in project root

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (with defaults)
CHUNK_SIZE=2048
CHUNK_OVERLAP=0
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
BATCH_SIZE=20
DELAY_BETWEEN_BATCHES=3
MAX_RETRIES=3
SUMMARY_TRUNCATE_LENGTH=500
```

### 7.2 Configuration Dictionary

**Location**: `batch_embedding.py`

```python
CONFIG = {
    # Input/Output paths
    "input_file": Path("../data/processed/documents/documents.jsonl"),
    "output_dir": Path("../data/embedding"),

    # Model configuration
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),

    # Chunking configuration (disabled for atomic docs)
    "chunk_size": int(os.getenv("CHUNK_SIZE", "2048")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "0")),

    # Batch processing
    "batch_size": int(os.getenv("BATCH_SIZE", "20")),
    "delay_between_batches": int(os.getenv("DELAY_BETWEEN_BATCHES", "3")),

    # Processing options
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    "summary_truncate_length": int(os.getenv("SUMMARY_TRUNCATE_LENGTH", "500")),

    # Flags
    "show_progress": True,
    "save_intermediate_results": True
}
```

### 7.3 Dependencies

**File**: `requirements.txt`

```txt
llama-index>=0.9.0
llama-index-embeddings-openai>=0.1.0
llama-index-llms-openai>=0.1.0
openai>=1.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

---

## 8. Output Specifications

### 8.1 Output Directory Structure

```
src-sale-promotion/data/embedding/embeddings_sales_20251026_173000/
├── batch_1/                                  # First batch (promotions 1-20)
│   ├── indexnodes/
│   │   ├── batch_1_indexnodes_metadata.json  # Metadata + embedding preview
│   │   ├── batch_1_indexnodes_full.pkl       # Complete Python objects
│   │   └── batch_1_indexnodes_vectors.npy    # Embedding vectors (20, 1536)
│   ├── summaries/
│   │   ├── batch_1_summaries_metadata.json
│   │   ├── batch_1_summaries_full.pkl
│   │   └── batch_1_summaries_vectors.npy
│   ├── combined/
│   │   ├── batch_1_all_metadata.json         # All embeddings combined
│   │   ├── batch_1_all_full.pkl
│   │   └── batch_1_all_vectors.npy           # Shape: (40, 1536)
│   └── batch_1_statistics.json               # Batch processing stats
│
├── batch_2/                                  # Second batch (promotions 21-40)
│   └── [same structure as batch_1]
│
├── batch_3/                                  # Third batch (promotions 41-60)
│   └── [same structure as batch_1]
│
├── batch_4/                                  # Fourth batch (promotions 61-71)
│   └── [same structure as batch_1]
│
└── combined_statistics.json                  # Overall statistics
```

### 8.2 Expected File Sizes

| File Type | Approx Size | Notes |
|-----------|-------------|-------|
| `*_metadata.json` | 500-800 KB per batch | Human-readable, UTF-8 Thai text |
| `*_full.pkl` | 1-2 MB per batch | Complete embeddings (1536 dims) |
| `*_vectors.npy` | 800-1000 KB per batch | Float arrays only |
| `*_statistics.json` | 5-10 KB | Processing metrics |
| `combined_statistics.json` | 10-15 KB | Overall summary |

**Total Storage (All Batches):** ~20-30 MB

### 8.3 File Content Examples

#### 8.3.1 Metadata JSON (`batch_1_indexnodes_metadata.json`)

```json
[
  {
    "node_id": "promo_001",
    "index_id": "promo_1",
    "doc_title": "Promotion: PROMO_001",
    "text": "Promotion: PROMO_001\n\nSummary: Special promotion for wholesale agents (เอเย่นต์): Buy 108 packs of Chang Drinking Water PET 600 CC and receive 42 packs FREE. This promotion offers a 28% discount on total quantity and is valid from October 1, 2025 to December 31, 2025 (92 days). Product category: water, Brand: Chang.",
    "text_length": 342,
    "embedding_preview": [0.0234, -0.0156, 0.0789, -0.0423, 0.0567],
    "embedding_dim": 1536,
    "metadata": {
      "document_id": "PROMO_001",
      "customer_type": "เอเย่นต์",
      "customer_tier": "tier_1",
      "is_wholesale": true,
      "is_retail": false,
      "product_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
      "product_category": "water",
      "product_brand": "ช้าง",
      "product_size": "600 CC",
      "buy_quantity": 108,
      "buy_unit": "แพ็ค",
      "free_item_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
      "free_quantity": 42,
      "free_unit": "แพ็ค",
      "discount_percentage": 28.0,
      "promotion_type": "buy_x_get_y",
      "start_date": "2568-10-01 00:00:00",
      "end_date": "2568-12-31 00:00:00",
      "start_date_gregorian": "2025-10-01",
      "end_date_gregorian": "2025-12-31",
      "is_active": true,
      "expires_soon": false,
      "duration_days": 92,
      "additional_conditions": null,
      "keywords": ["น้ำดื่มช้าง", "PET", "600", "CC", "แพ็ค", "เท่ากับ", "12", "ขวด", "water", "ช้าง", "เอเย่นต์"],
      "doc_id": "PROMO_001",
      "doc_title": "Promotion: PROMO_001",
      "type": "indexnode",
      "batch_number": 1
    },
    "type": "indexnode",
    "batch_number": 1
  }
  // ... 19 more promotions in batch 1
]
```

#### 8.3.2 Batch Statistics (`batch_1_statistics.json`)

```json
{
  "batch_number": 1,
  "extraction_timestamp": "2025-10-26T17:32:15.234567",
  "embedding_model": "text-embedding-3-small",
  "llm_model": "gpt-4o-mini",
  "totals": {
    "indexnode_embeddings": 20,
    "summary_embeddings": 20,
    "total_embeddings": 40
  },
  "metadata_analysis": {
    "total_metadata_fields": 27,
    "metadata_fields": [
      "additional_conditions",
      "batch_number",
      "buy_quantity",
      "buy_unit",
      "customer_tier",
      "customer_type",
      "discount_percentage",
      "doc_id",
      "doc_title",
      "document_id",
      "duration_days",
      "end_date",
      "end_date_gregorian",
      "expires_soon",
      "free_item_name",
      "free_quantity",
      "free_unit",
      "is_active",
      "is_retail",
      "is_wholesale",
      "keywords",
      "product_brand",
      "product_category",
      "product_name",
      "product_size",
      "promotion_type",
      "start_date",
      "start_date_gregorian",
      "type"
    ]
  }
}
```

#### 8.3.3 Combined Statistics (`combined_statistics.json`)

```json
{
  "processing_timestamp": "2025-10-26T17:35:42.123456",
  "embedding_model": "text-embedding-3-small",
  "llm_model": "gpt-4o-mini",
  "total_batches": 4,
  "grand_totals": {
    "total_documents": 71,
    "indexnode_embeddings": 71,
    "summary_embeddings": 71,
    "total_embeddings": 142,
    "chunk_embeddings": 0
  },
  "metadata_analysis": {
    "total_unique_metadata_fields": 27,
    "metadata_fields": [
      "additional_conditions",
      "batch_number",
      "buy_quantity",
      "buy_unit",
      "customer_tier",
      "customer_type",
      "discount_percentage",
      "doc_id",
      "doc_title",
      "document_id",
      "duration_days",
      "end_date",
      "end_date_gregorian",
      "expires_soon",
      "free_item_name",
      "free_quantity",
      "free_unit",
      "is_active",
      "is_retail",
      "is_wholesale",
      "keywords",
      "product_brand",
      "product_category",
      "product_name",
      "product_size",
      "promotion_type",
      "start_date",
      "start_date_gregorian",
      "type"
    ]
  },
  "batch_breakdown": [
    {
      "batch_number": 1,
      "documents": 20,
      "indexnodes": 20,
      "summaries": 20,
      "total": 40
    },
    {
      "batch_number": 2,
      "documents": 20,
      "indexnodes": 20,
      "summaries": 20,
      "total": 40
    },
    {
      "batch_number": 3,
      "documents": 20,
      "indexnodes": 20,
      "summaries": 20,
      "total": 40
    },
    {
      "batch_number": 4,
      "documents": 11,
      "indexnodes": 11,
      "summaries": 11,
      "total": 22
    }
  ]
}
```

---

## 9. Quality Assurance & Validation

### 9.1 Input Validation

**Validation Checklist:**
- ✅ JSONL file exists at expected path
- ✅ File contains exactly 71 lines
- ✅ Each line is valid JSON
- ✅ Each document has `id`, `text`, `metadata` fields
- ✅ Metadata contains all 25 required fields
- ✅ Thai text is UTF-8 encoded
- ✅ No duplicate document IDs
- ✅ Text fields are non-empty (minimum 50 characters)

**Validation Script:**
```python
def validate_input():
    loader = PromotionDocumentLoader()
    report = loader.validate_jsonl_file(jsonl_path)

    assert report["is_valid"], f"Validation failed: {report['errors']}"
    assert report["line_count"] == 71, "Expected 71 promotions"
    assert report["valid_json_lines"] == 71, "All lines must be valid JSON"
```

### 9.2 Processing Validation

**Validation Checks:**
- ✅ API key is valid and accessible
- ✅ DocumentSummaryIndex built successfully
- ✅ No chunking occurred (chunk_size > doc size)
- ✅ LLM summaries generated for all promotions
- ✅ IndexNodes created (1 per promotion)
- ✅ Embeddings extracted (71 IndexNode + 71 Summary = 142 total)
- ✅ Embedding dimensions are 1536
- ✅ All metadata fields preserved
- ✅ Thai text encoding maintained

**Processing Assertions:**
```python
def validate_processing(doc_summary_index, embeddings):
    # Check DocumentSummaryIndex
    assert len(doc_summary_index.ref_doc_info) == expected_count

    # Check embeddings
    assert len(embeddings['indexnode']) == expected_count
    assert len(embeddings['summary']) == expected_count

    # Check dimensions
    for emb in embeddings['indexnode']:
        assert emb['embedding_dim'] == 1536

    # Check metadata preservation
    for emb in embeddings['indexnode']:
        assert len(emb['metadata']) >= 25
```

### 9.3 Output Validation

**Validation Checks:**
- ✅ All output directories created
- ✅ All output files exist (JSON, PKL, NPY)
- ✅ JSON files are valid and UTF-8 encoded
- ✅ PKL files are loadable
- ✅ NPY arrays have correct shape (N, 1536)
- ✅ Statistics files generated
- ✅ Combined statistics accurate
- ✅ No data loss in conversions

**Output Validation Script:**
```python
def validate_output(output_dir):
    # Check directories
    assert (output_dir / "batch_1").exists()
    assert (output_dir / "combined_statistics.json").exists()

    # Validate JSON
    with open(output_dir / "batch_1/indexnodes/batch_1_indexnodes_metadata.json") as f:
        data = json.load(f)
        assert len(data) > 0
        assert data[0]['embedding_dim'] == 1536

    # Validate NPY
    vectors = np.load(output_dir / "batch_1/indexnodes/batch_1_indexnodes_vectors.npy")
    assert vectors.shape[1] == 1536

    # Validate PKL
    with open(output_dir / "batch_1/indexnodes/batch_1_indexnodes_full.pkl", 'rb') as f:
        embeddings = pickle.load(f)
        assert len(embeddings) > 0
```

---

## 10. Success Criteria

### 10.1 Quantitative Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Document Processing** | 71/71 (100%) | Count lines in combined_statistics.json |
| **Embedding Generation** | 142 total (71 IndexNode + 71 Summary) | Check grand_totals in combined_statistics.json |
| **Embedding Dimensions** | 1536 for all embeddings | Validate embedding_dim field |
| **Metadata Preservation** | 27 fields (25 original + 2 added) | Check metadata_fields count |
| **Thai Text Integrity** | 100% UTF-8 preserved | Inspect JSON output files |
| **Processing Success Rate** | 100% (no errors) | Check for error logs |
| **Output Files Generated** | 3 formats × 3 types × 4 batches + stats | Count files in output directory |
| **Batch Processing** | 4 batches (20+20+20+11) | Check batch_breakdown |

### 10.2 Qualitative Criteria

**Code Quality:**
- ✅ Follows `src-imap/` modular architecture
- ✅ Clear separation of concerns (5 modules)
- ✅ Comprehensive error handling
- ✅ Detailed logging and progress tracking
- ✅ Type hints and docstrings
- ✅ Reusable configuration

**Production Readiness:**
- ✅ Batch processing with rate limiting
- ✅ API retry logic with exponential backoff
- ✅ Multi-format output for flexibility
- ✅ Comprehensive statistics and validation
- ✅ UTF-8 Thai text support
- ✅ Metadata preservation throughout pipeline

**Documentation:**
- ✅ Module-level README
- ✅ Inline code comments
- ✅ Usage examples
- ✅ Configuration documentation

### 10.3 Acceptance Tests

#### Test 1: Complete Pipeline Execution
```python
def test_complete_pipeline():
    """Test end-to-end pipeline execution."""
    pipeline = SalesPromotionBatchEmbeddingPipeline()
    pipeline.run()

    # Verify output directory exists
    assert latest_output_dir.exists()

    # Verify all batches processed
    assert len(list(latest_output_dir.glob("batch_*"))) == 4

    # Verify statistics
    with open(latest_output_dir / "combined_statistics.json") as f:
        stats = json.load(f)
        assert stats["grand_totals"]["total_documents"] == 71
        assert stats["grand_totals"]["total_embeddings"] == 142
```

#### Test 2: Metadata Preservation
```python
def test_metadata_preservation():
    """Test that all 25 metadata fields are preserved."""
    with open(output_dir / "batch_1/indexnodes/batch_1_indexnodes_metadata.json") as f:
        embeddings = json.load(f)

        for emb in embeddings:
            # Check original fields present
            assert "customer_type" in emb["metadata"]
            assert "product_category" in emb["metadata"]
            assert "discount_percentage" in emb["metadata"]

            # Check added fields
            assert "batch_number" in emb["metadata"]
            assert "type" in emb["metadata"]

            # Check total field count
            assert len(emb["metadata"]) >= 27
```

#### Test 3: Thai Text Encoding
```python
def test_thai_text_encoding():
    """Test UTF-8 Thai text preservation."""
    with open(output_dir / "batch_1/indexnodes/batch_1_indexnodes_metadata.json", encoding='utf-8') as f:
        embeddings = json.load(f)

        for emb in embeddings:
            # Check Thai text in text field
            assert "แพ็ค" in emb["text"] or "น้ำดื่ม" in emb["text"]

            # Check Thai text in metadata
            assert "เอเย่นต์" in emb["metadata"]["customer_type"] or \
                   "ซับเอเย่นต์" in emb["metadata"]["customer_type"] or \
                   "รถขายปลีก" in emb["metadata"]["customer_type"]
```

#### Test 4: Embedding Quality
```python
def test_embedding_quality():
    """Test embedding vector quality."""
    vectors = np.load(output_dir / "batch_1/indexnodes/batch_1_indexnodes_vectors.npy")

    # Check shape
    assert vectors.shape[1] == 1536

    # Check no NaN or Inf
    assert not np.any(np.isnan(vectors))
    assert not np.any(np.isinf(vectors))

    # Check reasonable value ranges (-1 to 1 for normalized embeddings)
    assert np.all(vectors >= -2) and np.all(vectors <= 2)
```

---

## 11. Implementation Plan

### 11.1 Development Phases

#### Phase 2.1: Module Development (2-3 days)

**Day 1: Core Modules**
- [ ] Create `docs_embedding/` directory structure
- [ ] Implement `document_loader.py`
  - JSONL parsing
  - SimpleDocument → LlamaIndex Document conversion
  - Input validation
- [ ] Implement `embedding_processor.py`
  - DocumentSummaryIndex builder
  - IndexNode creation
  - Embedding extraction functions
- [ ] Write unit tests for core modules

**Day 2: Storage & Orchestration**
- [ ] Implement `file_storage.py`
  - Multi-format output (JSON, PKL, NPY)
  - Statistics generation
- [ ] Implement `batch_embedding.py` (main orchestrator)
  - Batch processing logic
  - Rate limiting
  - Progress tracking
- [ ] Write integration tests

**Day 3: Testing & Validation**
- [ ] Implement validation functions
- [ ] Create acceptance tests
- [ ] Test with Phase 1 output (`documents.jsonl`)
- [ ] Verify Thai text encoding
- [ ] Validate metadata preservation

#### Phase 2.2: Optimization & Documentation (1-2 days)

**Day 4: Optimization**
- [ ] Add comprehensive error handling
- [ ] Implement retry logic for API calls
- [ ] Add detailed logging
- [ ] Optimize batch sizing based on API limits
- [ ] Performance testing

**Day 5: Documentation**
- [ ] Write `docs_embedding/README.md`
- [ ] Add inline code documentation
- [ ] Create usage examples
- [ ] Document configuration options
- [ ] Update project METHODOLOGY.md

#### Phase 2.3: Production Deployment (1 day)

**Day 6: Deployment**
- [ ] Run complete pipeline on production data
- [ ] Generate and validate all outputs
- [ ] Create implementation summary document
- [ ] Prepare handoff to Phase 3 (Retrieval)

### 11.2 Testing Strategy

#### Unit Tests
```python
# test_document_loader.py
def test_load_jsonl()
def test_convert_to_llamaindex_document()
def test_validate_jsonl_file()

# test_embedding_processor.py
def test_build_document_summary_index()
def test_extract_indexnode_embeddings()
def test_extract_summary_embeddings()

# test_file_storage.py
def test_save_batch_embeddings()
def test_save_embedding_collection()
def test_save_statistics()
```

#### Integration Tests
```python
# test_integration.py
def test_end_to_end_pipeline()
def test_batch_processing()
def test_metadata_preservation()
def test_multi_format_output()
```

#### Acceptance Tests
```python
# test_acceptance.py
def test_71_promotions_processed()
def test_142_embeddings_generated()
def test_thai_text_preserved()
def test_all_output_files_created()
```

### 11.3 Rollout Plan

#### Step 1: Development Environment Setup
- Clone repository
- Install dependencies (`pip install -r requirements.txt`)
- Configure `.env` with OpenAI API key
- Verify Phase 1 output exists

#### Step 2: Module Implementation
- Implement modules in order: loader → processor → storage → orchestrator
- Run unit tests after each module
- Commit changes incrementally

#### Step 3: Integration Testing
- Run integration tests on small dataset (10 promotions)
- Validate outputs
- Fix issues

#### Step 4: Full Pipeline Execution
- Run complete pipeline on all 71 promotions
- Monitor progress and resource usage
- Validate all outputs

#### Step 5: Production Deployment
- Generate final outputs
- Create implementation summary
- Prepare for Phase 3 handoff

---

## 12. Performance Requirements

### 12.1 Processing Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Total Processing Time** | < 5 minutes | For all 71 promotions |
| **Throughput** | > 15 promotions/min | Including API calls |
| **API Call Success Rate** | > 99% | With retries |
| **Memory Usage** | < 2 GB | Peak memory consumption |
| **Batch Processing Delay** | 3 seconds | Between batches |

### 12.2 API Usage

**OpenAI API Calls:**
- **Embedding API**: 142 calls (71 IndexNode + 71 Summary)
- **LLM API**: ~71 calls (summary generation)
- **Total Tokens**: ~50,000-70,000 tokens (estimate)
- **Estimated Cost**: $0.50-1.00 USD

**Rate Limiting:**
- Batch delay: 3 seconds between batches
- Retry backoff: Exponential (1s, 2s, 4s)
- Max retries: 3 per API call

### 12.3 Storage Performance

| Output Type | Size per Batch | Total Size (4 Batches) |
|-------------|----------------|------------------------|
| JSON metadata | 500-800 KB | 2-3 MB |
| PKL objects | 1-2 MB | 4-8 MB |
| NPY vectors | 800-1000 KB | 3-4 MB |
| Statistics | 5-10 KB | 20-40 KB |
| **Total** | ~2-4 MB/batch | **~10-15 MB** |

### 12.4 Resource Requirements

**Hardware:**
- CPU: 2+ cores recommended
- RAM: 4 GB minimum, 8 GB recommended
- Storage: 100 MB available space

**Network:**
- Internet connection required for OpenAI API
- Bandwidth: Minimal (<10 MB total)

---

## 13. Error Handling & Recovery

### 13.1 Error Categories

#### 13.1.1 Input Errors
**Error**: JSONL file not found
**Recovery**: Verify Phase 1 completion, check file path

**Error**: Invalid JSON format
**Recovery**: Re-run Phase 1, validate JSONL file

**Error**: Missing metadata fields
**Recovery**: Check Phase 1 configuration, regenerate documents

#### 13.1.2 API Errors
**Error**: OpenAI API key invalid
**Recovery**: Verify API key in `.env`, check quotas

**Error**: API rate limit exceeded
**Recovery**: Increase batch delay, reduce batch size

**Error**: API timeout
**Recovery**: Retry with exponential backoff

#### 13.1.3 Processing Errors
**Error**: DocumentSummaryIndex build failure
**Recovery**: Check document format, validate API access

**Error**: Embedding extraction failure
**Recovery**: Retry failed embeddings, check API status

**Error**: Memory error during batch processing
**Recovery**: Reduce batch size, clear caches

#### 13.1.4 Output Errors
**Error**: File write permission denied
**Recovery**: Check directory permissions, verify disk space

**Error**: Disk space full
**Recovery**: Free up space, reduce output formats

### 13.2 Retry Logic

```python
def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """
    Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            delay = initial_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
```

### 13.3 Checkpoint & Resume

**Checkpoint Strategy:**
- Save after each batch completion
- Record processed batches in checkpoint file
- Resume from last successful batch on failure

**Checkpoint File:**
```json
{
  "last_completed_batch": 2,
  "total_batches": 4,
  "completed_promotions": 40,
  "timestamp": "2025-10-26T17:30:00"
}
```

**Resume Logic:**
```python
def resume_from_checkpoint(checkpoint_file):
    """Resume processing from last checkpoint."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)

        start_batch = checkpoint["last_completed_batch"] + 1
        print(f"Resuming from batch {start_batch}")
        return start_batch

    return 1  # Start from beginning
```

---

## 14. Appendices

### 14.1 Appendix A: Complete Code Example

**File**: `run_phase2.py` (Complete example script)

```python
"""
Complete example script for Phase 2 embedding generation.

Usage:
    python run_phase2.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src-sale-promotion to path
sys.path.append(str(Path(__file__).parent))

# Load environment
load_dotenv()

# Import pipeline
from docs_embedding.batch_embedding import SalesPromotionBatchEmbeddingPipeline

def main():
    """Run Phase 2 embedding pipeline."""

    print("=" * 80)
    print("SALES PROMOTION EMBEDDING PIPELINE - PHASE 2")
    print("=" * 80)

    # Validate prerequisites
    print("\n1. Validating prerequisites...")

    # Check Phase 1 output exists
    input_file = Path("data/processed/documents/documents.jsonl")
    if not input_file.exists():
        print(f"❌ Error: Phase 1 output not found: {input_file}")
        print("   Please run Phase 1 first.")
        sys.exit(1)
    print(f"✅ Phase 1 output found: {input_file}")

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please set OPENAI_API_KEY in .env file")
        sys.exit(1)
    print(f"✅ OpenAI API key configured")

    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    try:
        pipeline = SalesPromotionBatchEmbeddingPipeline()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        sys.exit(1)

    # Run pipeline
    print("\n3. Running embedding generation pipeline...")
    print("   This will take approximately 3-5 minutes...\n")

    try:
        pipeline.run()
        print("\n✅ Pipeline completed successfully!")

        # Display output location
        print(f"\n4. Output saved to:")
        print(f"   {pipeline.output_dir}")

        # Display summary
        print(f"\n5. Summary:")
        print(f"   - Documents processed: 71")
        print(f"   - Embeddings generated: 142 (71 IndexNode + 71 Summary)")
        print(f"   - Batches processed: 4")
        print(f"   - Output formats: JSON, PKL, NPY")

        print("\n" + "=" * 80)
        print("PHASE 2 COMPLETE - Ready for Phase 3 (Retrieval)")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 14.2 Appendix B: Configuration Templates

#### 14.2.1 `.env` Template

```bash
# OpenAI API Configuration (REQUIRED)
OPENAI_API_KEY=sk-your-api-key-here

# Embedding Configuration (Optional - defaults provided)
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

# Chunking Configuration (Optional)
# Note: Set larger than document size to disable chunking
CHUNK_SIZE=2048
CHUNK_OVERLAP=0

# Batch Processing (Optional)
BATCH_SIZE=20
DELAY_BETWEEN_BATCHES=3

# Error Handling (Optional)
MAX_RETRIES=3

# Display Options (Optional)
SUMMARY_TRUNCATE_LENGTH=500
```

#### 14.2.2 Custom Configuration Example

```python
# custom_config.py
"""Custom configuration for different environments."""

# Development configuration (faster, smaller batches)
DEV_CONFIG = {
    "input_file": Path("data/processed/documents/documents.jsonl"),
    "output_dir": Path("data/embedding/dev"),
    "batch_size": 10,  # Smaller batches
    "delay_between_batches": 1,  # Faster for dev
    "summary_truncate_length": 200,
}

# Production configuration (optimized)
PROD_CONFIG = {
    "input_file": Path("data/processed/documents/documents.jsonl"),
    "output_dir": Path("data/embedding/prod"),
    "batch_size": 20,
    "delay_between_batches": 3,
    "summary_truncate_length": 500,
}

# Testing configuration (minimal API calls)
TEST_CONFIG = {
    "input_file": Path("test/fixtures/test_documents.jsonl"),
    "output_dir": Path("test/output"),
    "batch_size": 5,
    "delay_between_batches": 0,
}
```

### 14.3 Appendix C: Troubleshooting Guide

#### Issue 1: "JSONL file not found"
**Symptoms**: FileNotFoundError when running pipeline
**Cause**: Phase 1 not completed or incorrect path
**Solution**:
```bash
# Verify Phase 1 output exists
ls -lh src-sale-promotion/data/processed/documents/documents.jsonl

# If missing, run Phase 1 first
python src-sale-promotion/data_processing/run_data_processing_standalone.py
```

#### Issue 2: "OpenAI API key not found"
**Symptoms**: API authentication error
**Cause**: Missing or invalid OPENAI_API_KEY
**Solution**:
```bash
# Check .env file exists
cat .env | grep OPENAI_API_KEY

# If missing, create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Verify API key is valid
python -c "import openai; print(openai.api_key)"
```

#### Issue 3: "Rate limit exceeded"
**Symptoms**: 429 errors from OpenAI API
**Cause**: Too many API requests
**Solution**:
```bash
# Increase delay between batches
export DELAY_BETWEEN_BATCHES=5

# Or reduce batch size
export BATCH_SIZE=10

# Re-run pipeline
python run_phase2.py
```

#### Issue 4: "Thai text corrupted in JSON output"
**Symptoms**: Garbled Thai characters in JSON files
**Cause**: Incorrect encoding
**Solution**:
```python
# Verify UTF-8 encoding in file_storage.py
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)  # ensure_ascii=False is critical
```

#### Issue 5: "Memory error during batch processing"
**Symptoms**: MemoryError or system slowdown
**Cause**: Batch size too large
**Solution**:
```bash
# Reduce batch size
export BATCH_SIZE=10

# Or process in smaller chunks
python run_phase2.py --batch-size 5
```

### 14.4 Appendix D: Reference Documents

#### Internal References
- `src-sale-promotion/prd/01_PRD_PHASE1_DATA_PREP_v2.md` - Phase 1 PRD
- `src-sale-promotion/PHASE1_IMPLEMENTATION_SUMMARY.md` - Phase 1 results
- `src-sale-promotion/data/processed/README.md` - Phase 1 output documentation

#### External References (Proven Patterns)
- `src-imap/docs_embedding/README.md` - Primary reference implementation
- `src-imap/docs_embedding/batch_embedding.py` - Orchestrator pattern
- `src/09_enhanced_batch_embeddings.py` - Alternative embedding pattern
- `src/METHODOLOGY.md` - Overall RAG methodology
- `attached_assets/00_doc_prep_workflow.md` - Document preparation workflow
- `attached_assets/01_building_rag_pipeline.md` - RAG pipeline design

#### LlamaIndex Documentation
- DocumentSummaryIndex: https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_summary_index/
- Production RAG: https://docs.llamaindex.ai/en/stable/optimizing/production_rag/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings

### 14.5 Appendix E: Phase 1 → Phase 2 → Phase 3 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1: DATA PREPARATION                │
│  CSV → SimpleDocument (JSONL) + Markdown                        │
│  Output: documents.jsonl (71 promotions, 25 metadata fields)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: EMBEDDING GENERATION                 │
│  JSONL → LlamaIndex Documents → DocumentSummaryIndex            │
│  Output: 142 embeddings (JSON, PKL, NPY) + statistics           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 3: RETRIEVAL (FUTURE)                │
│  Load embeddings → Build retrieval strategies                   │
│  - Vector search                                                 │
│  - Metadata filtering                                            │
│  - Hybrid search (vector + keyword)                              │
│  Output: Query interface for sales promotion search             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 4: AGENTIC ROUTING (FUTURE)          │
│  Strategy router → Auto-select best retrieval approach          │
│  Output: Production RAG API for sales promotion queries         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 15. Document Approval

### 15.1 Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| **Author** | Development Team | 2025-10-26 | _Pending_ |
| **Technical Reviewer** | Senior Engineer | 2025-10-26 | _Pending_ |
| **Stakeholder** | Sales Operations | 2025-10-26 | _Pending_ |
| **Project Manager** | PM | 2025-10-26 | _Pending_ |

### 15.2 Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-10-26 | Dev Team | Initial draft |
| 1.0 | 2025-10-26 | Dev Team | Complete PRD with all sections |

---

## 16. Next Steps

After Phase 2 completion:

1. **Validate Outputs**: Run all acceptance tests
2. **Generate Statistics**: Review combined_statistics.json
3. **Create Summary**: Document Phase 2 results in `PHASE2_IMPLEMENTATION_SUMMARY.md`
4. **Phase 3 Planning**: Draft PRD for retrieval strategies
5. **Handoff Meeting**: Review outputs with stakeholders

---

**End of Document**

**Document ID**: PRD-SALES-PROMO-PHASE2-v1.0
**Classification**: Internal
**Distribution**: Development Team, Stakeholders
**Next Review Date**: 2025-11-26
