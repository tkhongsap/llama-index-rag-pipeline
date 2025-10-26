# RAG Pipeline Methodology

**Complete methodology for building a production-ready RAG (Retrieval-Augmented Generation) pipeline with LlamaIndex and OpenAI.**

This document describes the complete end-to-end process implemented in the `src/` folder, from raw CSV data to intelligent multi-strategy retrieval systems.

---

## 📚 Methodology Origins

This methodology represents a **battle-tested, production-ready approach** developed over several months of POC work and successfully applied to multiple projects:

### **Theoretical Foundation**
The architectural decisions and best practices documented here are based on comprehensive research captured in `attached_assets/`:
- **`00_doc_prep_workflow.md`** - Document preparation and LlamaIndex best practices
- **`01_building_rag_pipeline.md`** - Complete ingestion pipeline design for real estate data (20-30K records)
- **`02_rag_workflow_explain.md`** - RAG workflow components (parsing, chunking, embedding, routing)
- **`03_retrieval_pipeline_plan.md`** - 12-phase implementation plan (foundation → agentic workflows)
- **`04_agentic_retrieval.md`** - Product Requirements Document for agentic retrieval layer
- **`06_RAG_pipeline_50k_strategy.md`** - Scaling strategy for 50K+ documents
- **`07_agentic_retrieval_workflow.md`** - Workflow for iLand implementation

### **Proven Applications**
This methodology has been successfully applied to:

1. **Main Pipeline** (`src/`)
   - General-purpose RAG with 7 retrieval strategies
   - Candidate profile data processing
   - Complete agentic retrieval system with two-stage routing

2. **iLand Pipeline** (`src-iLand/`)
   - Thai land deed document processing (50K+ documents)
   - Specialized geographic and legal metadata
   - Fast metadata filtering (sub-50ms)
   - Production deployment with Streamlit UI

3. **Sales Promotion Pipeline** (`src-sale-promotion/`) - *In Development*
   - Streamlined approach for atomic transaction data
   - Focus on metadata filtering and hybrid search
   - Thai language product and customer data

### **Key Learnings**
- **Chunking**: 512 tokens with 50-token overlap is optimal for most use cases
- **Hybrid Search**: Essential for exact keyword matching (product names, codes)
- **Metadata Filtering**: Pre-filtering can improve retrieval speed by 90%
- **Strategy Selection**: LLM-based routing with fallback heuristics is most reliable
- **Scalability**: Streaming pipelines with persistent vector stores handle 50K+ docs efficiently

### **Document Structure**
- **This document** (`src/METHODOLOGY.md`) - Implementation reference for the 7-strategy approach
- **Theoretical docs** (`attached_assets/*.md`) - Architectural decisions and design rationale
- **Application-specific** (`src-*/METHODOLOGY.md`) - Adaptations for specific use cases

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Phase 1: Data Preparation](#phase-1-data-preparation)
4. [Phase 2: Embedding Generation](#phase-2-embedding-generation)
5. [Phase 3: Retrieval Strategies](#phase-3-retrieval-strategies)
6. [Phase 4: Agentic Retrieval](#phase-4-agentic-retrieval)
7. [Key Design Patterns](#key-design-patterns)
8. [Configuration & Environment](#configuration--environment)
9. [Running the Pipeline](#running-the-pipeline)
10. [When to Use Each Strategy](#when-to-use-each-strategy)

---

## Overview

This RAG pipeline transforms structured data (CSV files) into intelligent, queryable vector embeddings with multiple retrieval strategies. The methodology supports:

- **Flexible data ingestion** from CSV with automatic field mapping
- **Multi-level embeddings** (document, summary, chunk)
- **7 retrieval strategies** with different performance characteristics
- **Agentic routing** for automatic strategy selection
- **Production-ready** batch processing with rate limiting

**Technology Stack:**
- LlamaIndex (orchestration framework)
- OpenAI (embeddings: text-embedding-3-small, LLM: gpt-4o-mini)
- Python 3.9+

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA PREPARATION                     │
│  CSV Files → Documents (JSONL) → Markdown → Structured Metadata │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2: EMBEDDING GENERATION                   │
│    Batch Processing → DocumentSummaryIndex → 3 Embedding Types  │
│         (IndexNodes, Chunks, Summaries) → Multiple Formats       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 3: RETRIEVAL STRATEGIES                   │
│   7 Strategies: Vector │ Summary │ Recursive │ Metadata │       │
│              Chunk Decoupling │ Hybrid │ Planner                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 4: AGENTIC RETRIEVAL                      │
│        LLM-based Router → Strategy Selection → Answer            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Preparation

### Script: `02_prep_doc_for_embedding.py`

**Purpose:** Transform CSV data into structured documents with rich metadata.

### Process Flow

1. **CSV Analysis**
   - Auto-detect column structure and data types
   - Suggest field mappings based on column names
   - Generate reusable YAML configuration

2. **Field Categorization**
   - `identifier`: ID, Number, Code
   - `demographic`: Age, Province, Region, Location
   - `education`: Degree, Major, Institute, School
   - `career`: Position, Company, Industry, Experience
   - `compensation`: Salary, Bonus, Currency
   - `assessment`: Test, Score, Skill, Evaluation
   - `generic`: Other fields

3. **Document Generation**
   - Convert each CSV row to structured document
   - Extract and normalize metadata
   - Generate readable text content
   - Create both JSONL and Markdown formats

### Input/Output

**Input:**
- CSV files in `data/input_docs/`

**Output:**
- `*_documents.jsonl` - Structured documents
- `*_config.yaml` - Field mapping configuration
- `markdown_files/*.md` - Individual markdown documents
- `csv_analysis_report.json` - Analysis report

### Key Features

- **Auto-configuration**: Generates YAML configs from CSV structure
- **Flexible mapping**: Supports custom field types and aliases
- **Metadata preservation**: Maintains all structured information
- **Batch processing**: Handles large datasets efficiently

### Example Configuration

```yaml
name: candidate_profiles
description: Auto-generated configuration
field_mappings:
  - csv_column: "Age"
    metadata_key: "age"
    field_type: "demographic"
    data_type: "numeric"
    required: true
  - csv_column: "Degree"
    metadata_key: "degree"
    field_type: "education"
    data_type: "string"
    required: false
```

---

## Phase 2: Embedding Generation

### Script: `09_enhanced_batch_embeddings.py`

**Purpose:** Generate embeddings with structured metadata extraction and batch processing.

### Process Flow

1. **Document Loading**
   - Load markdown files with metadata extraction
   - Extract structured fields using regex patterns
   - Derive additional metadata (salary ranges, experience levels, etc.)

2. **Chunking & Indexing**
   - Split documents using `SentenceSplitter` (chunk_size=1024, overlap=50)
   - Build `DocumentSummaryIndex` with LLM-generated summaries
   - Create `IndexNodes` for hierarchical retrieval

3. **Embedding Extraction**
   - **IndexNode embeddings**: Document-level summaries
   - **Chunk embeddings**: Individual text segments
   - **Summary embeddings**: Document summaries
   - All embeddings preserve original metadata

4. **Batch Processing**
   - Process files in configurable batches (default: 20 files)
   - Implement rate limiting (3-second delays between batches)
   - Individual batch tracking and statistics

### Input/Output

**Input:**
- Markdown files from Phase 1 (`example/*.md`)

**Output:**
```
data/embedding/embeddings_enhanced_TIMESTAMP/
├── batch_1/
│   ├── indexnodes/
│   │   ├── batch_1_indexnodes_metadata.json
│   │   ├── batch_1_indexnodes_full.pkl
│   │   ├── batch_1_indexnodes_vectors.npy
│   │   └── batch_1_indexnodes_metadata_only.json
│   ├── chunks/
│   │   └── [same structure as indexnodes]
│   ├── summaries/
│   │   └── [same structure as indexnodes]
│   ├── combined/
│   │   └── batch_1_all_* [all embeddings combined]
│   └── batch_1_statistics.json
├── batch_2/
│   └── [same structure]
└── combined_statistics.json
```

### Three Embedding Types

1. **IndexNode Embeddings**
   - Document-level representations
   - Contains summary + metadata
   - Used for document-level retrieval

2. **Chunk Embeddings**
   - Fine-grained text segments
   - Preserves parent document metadata
   - Used for detailed retrieval

3. **Summary Embeddings**
   - LLM-generated document summaries
   - High-level semantic representation
   - Used for summary-first retrieval

### Metadata Enhancement

The script extracts and derives:
- **Direct extraction**: age, province, degree, salary, etc.
- **Derived categories**:
  - `salary_range`: under_20k, 20k_30k, 30k_50k, 50k_80k, over_80k
  - `experience_category`: entry, junior, mid, senior
  - `education_category`: bachelor, master, doctorate
  - `location_category`: bangkok, greater_bangkok, other_province
- **Content classification**: education, career, compensation, demographics, assessment

### Configuration

```python
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50
SUMMARY_EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
BATCH_SIZE = 20
DELAY_BETWEEN_BATCHES = 3  # seconds
```

---

## Phase 3: Retrieval Strategies

Seven specialized retrieval strategies, each optimized for different query types.

### 1. Vector Search (Basic Query Engine)
**Script:** `10_basic_query_engine.py`

**How it works:**
- Semantic similarity search using embeddings
- Simple top-K retrieval
- Response synthesis with LLM

**Best for:**
- General semantic queries
- Fast, reliable results
- Simple information retrieval

**Characteristics:**
- ⚡ Speed: Fast
- 🎯 Accuracy: High for semantic queries
- 🔧 Complexity: Low

**Example queries:**
- "What are the main topics?"
- "Show me information about education"

---

### 2. Document Summary Retrieval
**Script:** `11_document_summary_retriever.py`

**How it works:**
- Two-stage hierarchical retrieval:
  1. **Stage 1**: Retrieve top-K documents by summary similarity
  2. **Stage 2**: Get detailed chunks from selected documents
- Better context awareness than flat retrieval

**Best for:**
- Document-level questions
- Overview queries
- When you need to understand which documents are relevant before diving into details

**Characteristics:**
- ⚡ Speed: Moderate
- 🎯 Accuracy: High for document-level questions
- 🔧 Complexity: Moderate

**Example queries:**
- "Which documents discuss educational background and career progression?"
- "What documents contain compensation information?"

**Settings:**
- `summary_top_k`: Number of documents to retrieve (default: 5)
- `chunks_per_doc`: Chunks to retrieve per document (default: 3)

---

### 3. Recursive Retrieval
**Script:** `12_recursive_retriever.py`

**How it works:**
- Multi-level hierarchical retrieval
- Builds IndexNodes from DocumentSummaryIndex
- Can drill down from document → summary → chunks
- Recursive query decomposition

**Best for:**
- Complex queries requiring multiple levels of information
- Hierarchical document structures
- When you need to explore document relationships

**Characteristics:**
- ⚡ Speed: Slower (multiple retrieval steps)
- 🎯 Accuracy: High for hierarchical queries
- 🔧 Complexity: High

**Example queries:**
- "Explain the career progression patterns across different educational backgrounds"
- "How do compensation trends relate to experience levels in different industries?"

---

### 4. Metadata Filtering
**Script:** `14_metadata_filtering.py`

**How it works:**
- Auto-detect filter fields from query
- Apply metadata filters before vector search
- Pre-filter large document sets efficiently

**Best for:**
- Queries with specific attributes
- Structured data queries
- Filtering by categories (salary range, education level, location, etc.)

**Characteristics:**
- ⚡ Speed: Fast (reduces search space)
- 🎯 Accuracy: High when metadata is accurate
- 🔧 Complexity: Moderate

**Example queries:**
- "Show profiles with salary over 50k"
- "Find candidates with bachelor's degree in Bangkok"
- "List all senior-level positions in technology industry"

**Auto-detected filters:**
- Salary ranges, education levels, experience categories
- Geographic locations
- Custom derived metadata

---

### 5. Chunk Decoupling
**Script:** `15_chunk_decoupling.py`

**How it works:**
- Separates embedding chunks from content chunks
- Retrieve using summaries → return full content chunks
- Better context preservation

**Best for:**
- Detailed chunk analysis
- When you need precise chunk-level information with broader context
- Preserving document structure in results

**Characteristics:**
- ⚡ Speed: Moderate
- 🎯 Accuracy: High for detail-oriented queries
- 🔧 Complexity: Moderate

**Example queries:**
- "What specific qualifications are mentioned in the profiles?"
- "Show exact compensation details from relevant documents"

---

### 6. Hybrid Search
**Script:** `16_hybrid_search.py`

**How it works:**
- Combines semantic vector search + keyword search
- Multiple fusion methods:
  - **RRF (Reciprocal Rank Fusion)**: Combines ranks from both methods
  - **Weighted Fusion**: Weighted combination of scores
  - **Borda Count**: Rank-based voting
- Custom keyword indexing with TF-IDF scoring

**Best for:**
- Queries needing both conceptual and exact keyword matches
- Comprehensive search results
- When semantic search alone misses important exact matches

**Characteristics:**
- ⚡ Speed: Moderate (runs two retrievers)
- 🎯 Accuracy: **Highest for mixed content** ⭐
- 🔧 Complexity: Moderate

**Example queries:**
- "Find profiles with 'Python' skills and software engineering experience"
- "MBA degree holders in finance industry"

**Settings:**
```python
semantic_weight = 0.7
keyword_weight = 0.3
fusion_method = "rrf"  # or "weighted", "borda"
```

**Fusion Methods:**
- **RRF**: Best overall, handles varying score scales
- **Weighted**: Good when you know relative importance
- **Borda**: Good for consensus ranking

---

### 7. Query Planning Agent
**Script:** `17_query_planning_agent.py`

**How it works:**
- LLM-based query decomposition into sub-questions
- Parallel query execution across multiple strategies
- Result synthesis into comprehensive answer
- Complexity analysis and automatic decomposition decision

**Best for:**
- Complex multi-part questions
- Analytical tasks requiring step-by-step reasoning
- Questions requiring information from multiple sources/aspects

**Characteristics:**
- ⚡ Speed: Slowest (multiple queries + synthesis)
- 🎯 Accuracy: Highest for complex questions ⭐
- 🔧 Complexity: Very High

**Example queries:**
- "Compare educational qualifications, work experience, and salary expectations between technical vs business profiles"
- "What are the most common institutions, what salary ranges do their graduates expect, and how do assessment scores compare?"

**Process:**
1. **Complexity Analysis**: Analyze query for multiple topics, comparisons, temporal aspects
2. **Decomposition**: Break into sub-questions with priorities
3. **Parallel Execution**: Run sub-queries on appropriate strategies
4. **Synthesis**: Combine results into comprehensive answer

**Settings:**
- `max_sub_questions`: Maximum decompositions (default: 5)
- `max_parallel_queries`: Concurrent executions (default: 3)
- `query_timeout`: Per-query timeout (default: 30s)

---

## Phase 4: Agentic Retrieval

### Router-based Strategy Selection
**Location:** `src/agentic_retriever/`

**How it works:**
- Two-stage routing system:
  1. **Index Classification**: Determine which index to search (for multi-index setups)
  2. **Strategy Selection**: Choose best retrieval strategy for the query

**Strategy Selection Methods:**

1. **LLM-based Selection** (Recommended)
   - Analyzes query semantics
   - Considers strategy strengths/weaknesses
   - Provides reasoning for selection
   ```python
   strategy_selector = "llm"
   llm_strategy_mode = "enhanced"  # or "simple"
   ```

2. **Fallback Heuristics**
   - Reliability ranking: vector → hybrid → recursive → chunk_decoupling → planner → metadata → summary
   - Query pattern matching:
     - Simple queries → vector
     - Complex multi-step → planner
     - Keyword-heavy → hybrid
     - Attribute filtering → metadata

3. **Round-robin** (Testing)
   - Cycle through strategies evenly
   - Good for performance comparison

### CLI Interface
**Script:** `agentic_retriever/cli.py`

Interactive command-line interface for testing all strategies:
```bash
python -m src.agentic_retriever
```

**Features:**
- Interactive mode with auto-routing
- Manual strategy selection
- Strategy comparison mode
- Retrieval statistics and timing
- Debug logging

---

## Key Design Patterns

### 1. Modular Architecture
- Each retrieval strategy is independent
- Strategies implement common interface (BaseRetriever)
- Easy to add new strategies without breaking existing ones

### 2. Metadata Preservation
- Metadata flows through entire pipeline:
  - CSV → Documents → Embeddings → Retrieval Results
- Enables powerful filtering and analysis
- Structured data + semantic search

### 3. Batch Processing
- Configurable batch sizes for memory management
- Rate limiting to respect API limits
- Progress tracking and error handling
- Individual batch statistics

### 4. Multiple Output Formats
- **JSON**: Metadata with embedding previews (human-readable)
- **PKL**: Complete Python objects (for loading into Python)
- **NPY**: Numpy arrays (for vector operations)
- **Statistics**: Analysis and quality metrics

### 5. LLM-Enhanced Features
- Document summarization for better retrieval
- Query decomposition for complex questions
- Strategy routing with reasoning
- Result synthesis

### 6. Hierarchical Retrieval
- Document → Summary → Chunk levels
- Better context awareness
- More relevant results for complex queries

### 7. Ensemble Methods
- Multiple retrievers + fusion
- Improved accuracy through consensus
- Configurable weights and fusion algorithms

---

## Configuration & Environment

### Required Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (with defaults)
CHUNK_SIZE=1024
CHUNK_OVERLAP=50
SUMMARY_EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
BATCH_SIZE=20
MAX_WORKERS=4
REQUEST_DELAY=0.1
MAX_RETRIES=3
```

### Directory Structure

```
src/
├── 02_prep_doc_for_embedding.py      # Phase 1: Data prep
├── 09_enhanced_batch_embeddings.py   # Phase 2: Embeddings
├── 10_basic_query_engine.py          # Strategy 1: Vector
├── 11_document_summary_retriever.py  # Strategy 2: Summary
├── 12_recursive_retriever.py         # Strategy 3: Recursive
├── 14_metadata_filtering.py          # Strategy 4: Metadata
├── 15_chunk_decoupling.py            # Strategy 5: Chunk Decoupling
├── 16_hybrid_search.py               # Strategy 6: Hybrid
├── 17_query_planning_agent.py        # Strategy 7: Planner
├── load_embeddings.py                # Embedding loader utilities
├── agentic_retriever/                # Phase 4: Router
│   ├── cli.py                        # Interactive CLI
│   ├── router.py                     # Main router logic
│   ├── index_classifier.py           # Index classification
│   └── retrievers/                   # Strategy adapters
└── README.md                         # Script documentation
```

---

## Running the Pipeline

### Complete Pipeline (Development)

```bash
# Phase 1: Data Preparation
python src/02_prep_doc_for_embedding.py

# Phase 2: Embedding Generation
python src/09_enhanced_batch_embeddings.py

# Phase 3: Test Individual Strategies
python src/10_basic_query_engine.py interactive
python src/11_document_summary_retriever.py interactive
python src/16_hybrid_search.py

# Phase 4: Agentic Retrieval
python -m src.agentic_retriever
```

### Quick Testing

```bash
# Interactive query with auto-routing
python -m src.agentic_retriever

# Test specific strategy
python src/16_hybrid_search.py  # Hybrid search
python src/17_query_planning_agent.py  # Query planner
```

---

## When to Use Each Strategy

### Decision Tree

```
Start with query type:

📊 Simple semantic query ("What is...", "Show me...")
   → Use: Vector Search (10_basic_query_engine.py)
   → Speed: ⚡⚡⚡ | Accuracy: 🎯🎯🎯

📄 Document-level question ("Which documents...", "What documents...")
   → Use: Document Summary Retrieval (11_document_summary_retriever.py)
   → Speed: ⚡⚡ | Accuracy: 🎯🎯🎯🎯

🏗️ Complex hierarchical query (relationships, nested information)
   → Use: Recursive Retrieval (12_recursive_retriever.py)
   → Speed: ⚡ | Accuracy: 🎯🎯🎯🎯

🔍 Query with specific attributes ("salary > 50k", "bachelor degree")
   → Use: Metadata Filtering (14_metadata_filtering.py)
   → Speed: ⚡⚡⚡ | Accuracy: 🎯🎯🎯🎯

🔤 Query needs exact keyword matches + semantic understanding
   → Use: Hybrid Search (16_hybrid_search.py) ⭐ RECOMMENDED
   → Speed: ⚡⚡ | Accuracy: 🎯🎯🎯🎯🎯

🧠 Multi-part analytical question
   → Use: Query Planning Agent (17_query_planning_agent.py)
   → Speed: ⚡ | Accuracy: 🎯🎯🎯🎯🎯

❓ Not sure?
   → Use: Agentic Retrieval with auto-routing
   → Let LLM choose the best strategy
```

### Performance Characteristics

| Strategy | Speed | Accuracy | Complexity | Best For |
|----------|-------|----------|------------|----------|
| Vector | ⚡⚡⚡ | 🎯🎯🎯 | Low | General queries |
| Summary | ⚡⚡ | 🎯🎯🎯🎯 | Medium | Document-level |
| Recursive | ⚡ | 🎯🎯🎯🎯 | High | Hierarchical |
| Metadata | ⚡⚡⚡ | 🎯🎯🎯🎯 | Medium | Attribute filters |
| Chunk Decoupling | ⚡⚡ | 🎯🎯🎯🎯 | Medium | Detail-oriented |
| **Hybrid** | ⚡⚡ | **🎯🎯🎯🎯🎯** | Medium | **Mixed content** ⭐ |
| Planner | ⚡ | 🎯🎯🎯🎯🎯 | Very High | Complex questions |

### Reliability Ranking (Production)

Based on actual performance analysis:

1. **Vector** ✅ - Most reliable, always works
2. **Hybrid** ✅ - Best accuracy, slightly slower
3. **Recursive** ✅ - Good for complex queries
4. **Chunk Decoupling** ✅ - Good for details
5. **Planner** ⚠️ - Good for multi-step, but slower
6. **Metadata** ⚠️ - Can return empty if metadata doesn't match
7. **Summary** ⚠️ - Can return empty results (needs tuning)

---

## Applying This Methodology to New Datasets

When building a RAG pipeline for new data (e.g., sales promotions):

### 1. Adapt Data Preparation (Phase 1)
- Update field mappings for your domain
- Define relevant metadata categories
- Customize text generation templates
- Add domain-specific metadata extraction

### 2. Configure Embedding Generation (Phase 2)
- Adjust chunk size based on document length
- Tune batch size for your dataset size
- Consider domain-specific metadata derivation
- Add custom content classification

### 3. Select Retrieval Strategies (Phase 3)
- Start with Vector + Hybrid (most reliable)
- Add Metadata if you have structured attributes
- Add Summary for document-level queries
- Add Planner for complex analytical queries

### 4. Customize Router (Phase 4)
- Update strategy descriptions for your domain
- Add domain-specific heuristics
- Train/tune strategy selection based on query patterns
- Implement custom index classification if multi-index

### 5. Test and Iterate
- Start with small dataset subset
- Test each strategy with representative queries
- Compare performance and accuracy
- Tune configurations based on results

---

## Best Practices

1. **Start Simple**: Begin with Vector search, add complexity as needed
2. **Preserve Metadata**: Maintain structured information throughout pipeline
3. **Batch Processing**: Always use batching for large datasets
4. **Monitor API Usage**: Implement rate limiting and delays
5. **Multiple Formats**: Save embeddings in multiple formats for flexibility
6. **Test Strategies**: Compare strategies on your specific use cases
7. **Use Hybrid**: When in doubt, Hybrid search provides best overall results
8. **Log Everything**: Track retrieval performance and strategy selection
9. **Iterate**: Continuously improve based on real query patterns
10. **Document**: Maintain configuration documentation for reproducibility

---

## Troubleshooting

### Empty Results
- **Metadata Filtering**: Check if metadata values match exactly
- **Summary Retrieval**: May need tuning or different summary strategy
- **Solution**: Fall back to Vector or Hybrid search

### Slow Performance
- **Planner**: Expected, runs multiple queries
- **Recursive**: Multiple retrieval steps
- **Solution**: Use Vector for simple queries, cache results

### Memory Issues
- Reduce batch size in embedding generation
- Process smaller dataset subsets
- Clear caches between batches

### API Rate Limits
- Increase delays between batches
- Reduce max workers
- Implement exponential backoff

---

## Next Steps

After understanding this methodology:

1. **Review the Scripts**: Read through each script to understand implementation details
2. **Test with Your Data**: Run the pipeline with your dataset
3. **Compare Strategies**: Test different strategies with your query types
4. **Customize**: Adapt configurations and metadata for your domain
5. **Deploy**: Choose the best strategies for production based on testing

---

## References

- LlamaIndex Documentation: https://docs.llamaindex.ai/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- RAG Best Practices: https://docs.llamaindex.ai/en/stable/optimizing/production_rag/

---

**Last Updated:** 2025-10-26
**Pipeline Version:** POC v1.0
**Authors:** Development Team
