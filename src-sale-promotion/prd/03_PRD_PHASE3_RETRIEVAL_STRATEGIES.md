# PRD: Phase 3 - Sales Promotion Retrieval Strategies & Query Interface

## Document Control

| Property | Value |
|----------|-------|
| **Document Title** | Phase 3: Retrieval Strategies & Query Interface for Sales Promotion RAG Pipeline |
| **Version** | 1.0 |
| **Status** | DRAFT |
| **Created Date** | 2025-10-26 |
| **Last Updated** | 2025-10-26 |
| **Authors** | Development Team |
| **Stakeholders** | RAG Pipeline Team, Sales Operations, End Users |
| **Dependencies** | Phase 2: Embedding & Indexing Pipeline (v1.0 - COMPLETE) |
| **Related Documents** | - `02_PRD_PHASE2_EMBEDDING_INDEXING.md`<br>- `PHASE2_IMPLEMENTATION_SUMMARY.md`<br>- `src-imap/retrieval/README.md`<br>- `src/METHODOLOGY.md` |

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

Phase 3 implements intelligent retrieval strategies that enable users to query and interact with the 71 sales promotion documents embedded in Phase 2. This phase transforms static embeddings into an interactive, production-ready query system with 7 specialized retrieval strategies, automatic routing, and both CLI and web interfaces.

### 1.2 Scope

**In Scope:**
- Implement 7 retrieval strategies optimized for sales promotion queries
- Build two-stage routing system (index classification → strategy selection)
- Implement fast metadata indexing for sub-50ms filtering
- Create interactive CLI interface with multiple query modes
- Develop Streamlit web UI with WhatsApp-style chat interface
- Support Thai language queries (product names, customer types)
- Leverage all 29 metadata fields from Phase 2
- Enable intelligent query routing based on LLM analysis
- Implement caching for repeated queries
- Provide comprehensive logging and debugging capabilities

**Out of Scope:**
- Data preparation (Phase 1 - COMPLETE)
- Embedding generation (Phase 2 - COMPLETE)
- Production API deployment (Phase 4)
- Authentication and user management (Phase 4)
- Multi-user session management (Phase 4)
- Advanced analytics and reporting (Phase 4)

### 1.3 Key Features

1. **Seven Specialized Retrieval Strategies**
   - Vector Search (semantic similarity)
   - Hybrid Search (semantic + keyword)
   - Metadata Filtering (structured queries)
   - Summary Retrieval (document-level)
   - Recursive Retrieval (hierarchical)
   - Chunk Decoupling (context preservation)
   - Query Planner (multi-step reasoning)

2. **Intelligent Routing**
   - LLM-based strategy selection
   - Two-stage routing (index → strategy)
   - Fallback heuristics for reliability
   - Performance-based optimization

3. **High Performance**
   - Sub-100ms for simple queries
   - Sub-50ms metadata filtering
   - Query result caching
   - Efficient batch processing

4. **Thai Language Support**
   - Customer types: เอเย่นต์, ซับเอเย่นต์, ลูกค้าปลีก
   - Product categories: น้ำดื่ม, เบียร์, สุรา
   - Keyword extraction from Thai text
   - UTF-8 encoding throughout

5. **User Interfaces**
   - Interactive CLI with multiple modes
   - Streamlit web UI with chat interface
   - Batch query processing
   - Strategy comparison tools

### 1.4 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Retrieval Accuracy** | >85% | Relevant results in top-5 |
| **Response Time (Simple)** | <100ms | Vector/metadata queries |
| **Response Time (Complex)** | <2s | Planner/recursive queries |
| **Metadata Filtering Speed** | <50ms | Pre-filtering on 142 embeddings |
| **Strategy Selection Accuracy** | >90% | LLM selects appropriate strategy |
| **Thai Query Support** | 100% | UTF-8 handling, keyword extraction |
| **Cache Hit Rate** | >30% | Repeated query performance |
| **CLI Responsiveness** | <500ms | Total query-to-display time |

### 1.5 User Personas

**Persona 1: Sales Operations Manager**
- Needs: Quick access to promotion details, filter by customer type, date ranges
- Queries: "โปรโมชันสำหรับเอเย่นต์ที่ยังใช้งานได้", "ส่วนลดน้ำดื่มเดือนนี้"
- Strategy: Metadata filtering, Hybrid search

**Persona 2: Sales Representative**
- Needs: Find similar promotions, product-specific offers
- Queries: "โปรโมชันแบบซื้อ 1 แถม 1", "ของแถมสำหรับเบียร์ช้าง"
- Strategy: Vector search, Summary retrieval

**Persona 3: Data Analyst**
- Needs: Complex queries, trend analysis, multi-step reasoning
- Queries: "เปรียบเทียบส่วนลดระหว่างลูกค้าขายส่งและปลีก", "โปรโมชันไหนมีเงื่อนไขซับซ้อน"
- Strategy: Query planner, Recursive retrieval

---

## 2. Project Context

### 2.1 Phase 2 Recap

Phase 2 **successfully completed** on 2025-10-26 with the following outputs:

**Embeddings Generated:**
- 71 promotion documents processed (100% success)
- 142 total embeddings:
  - 71 IndexNode embeddings (document-level navigation)
  - 71 Summary embeddings (LLM-generated summaries)
- Embedding dimensions: 3072 (text-embedding-3-large)
- Models used: GPT-5 (summaries) + text-embedding-3-large (embeddings)

**Metadata Available (29 fields):**

From Phase 1 (25 fields):
- **Customer**: document_id, customer_type, customer_tier, is_wholesale, is_retail
- **Product**: product_name, product_category, product_brand, product_size
- **Promotion**: buy_quantity, buy_unit, free_item_name, free_quantity, free_unit, discount_percentage, promotion_type
- **Dates**: start_date, end_date, start_date_gregorian, end_date_gregorian, is_active, expires_soon, duration_days
- **Additional**: additional_conditions, keywords

Added in Phase 2 (4 fields):
- doc_id, doc_title, type (indexnode/summary), batch_number

**Output Location:**
```
data/embedding/embeddings_sales_20251026_172927/
├── batch_1/ (20 promotions → 40 embeddings)
├── batch_2/ (20 promotions → 40 embeddings)
├── batch_3/ (20 promotions → 40 embeddings)
├── batch_4/ (11 promotions → 22 embeddings)
└── combined_statistics.json
```

**File Formats:**
- JSON: Human-readable metadata + embedding preview
- PKL: Complete Python objects with full 3072-dim vectors
- NPY: Pure numpy arrays (N, 3072) for vector operations

### 2.2 Integration Points

**Phase 3 will integrate with:**

1. **Embedding Files (Phase 2 Output)**
   - Load PKL files for complete embedding objects
   - Load NPY files for fast vector operations
   - Parse JSON for metadata inspection

2. **Existing Infrastructure**
   - LlamaIndex VectorStoreIndex construction
   - OpenAI API for LLM-based routing
   - text-embedding-3-large for query embeddings

3. **Reference Implementations**
   - `src-imap/retrieval/`: Production patterns for Thai language
   - `src/agentic_retriever/`: Router and strategy selection
   - `src/METHODOLOGY.md`: 7 retrieval strategies proven patterns

### 2.3 Current State Analysis

**What EXISTS (from commit 028ed8e):**
- ✅ Retrieval directory structure (`src-sale-promotion/retrieval/`)
- ✅ Base retriever adapter (`retrievers/base.py` - 65 lines)
- ✅ 7 retriever stubs (459 total lines):
  - vector.py (60 lines) - Basic implementation
  - metadata.py (55 lines) - Stub with TODOs
  - hybrid.py (50 lines) - Stub
  - summary.py (32 lines) - Stub
  - recursive.py (32 lines) - Stub
  - chunk_decoupling.py (32 lines) - Stub
  - planner.py (32 lines) - Stub
- ✅ CLI placeholder (`cli.py` - 70 lines)
- ✅ Comprehensive README (349 lines) - Documentation only

**What's MISSING (Phase 3 deliverables):**
- ❌ router.py (0 lines / ~300 lines needed)
- ❌ index_classifier.py (0 lines / ~200 lines needed)
- ❌ fast_metadata_index.py (0 lines / ~250 lines needed)
- ❌ streamlit_cli.py (0 lines / ~300 lines needed)
- ⚠️ Complete retriever implementations (~1,200 lines needed)
- ⚠️ Working CLI (~330 lines needed to complete)

**Implementation Gap:** ~2,600 lines of production code needed

### 2.4 Design Principles

Following proven patterns from src-imap/retrieval/:

1. **Modular Architecture**: Each retriever is independent and testable
2. **Adapter Pattern**: All retrievers inherit from BaseRetrieverAdapter
3. **Two-Stage Routing**: Index classification before strategy selection
4. **Performance First**: Fast metadata indexing, caching, efficient vector ops
5. **Thai Language First**: UTF-8 throughout, Thai keyword extraction
6. **Production Ready**: Comprehensive logging, error handling, graceful fallbacks

---

## 3. Functional Requirements

### 3.1 Retrieval Strategy 1: Vector Search

**Purpose:** Fast semantic similarity search using 3072-dim embeddings

**Best For:**
- General semantic queries
- Product/campaign similarity
- Quick information retrieval
- When no specific filters needed

**How It Works:**
1. Embed query using text-embedding-3-large (3072 dims)
2. Calculate cosine similarity with all 142 embeddings
3. Return top-K most similar nodes
4. Filter by similarity threshold (default: 0.7)

**Example Queries:**
```
"โปรโมชันที่คล้ายกับการซื้อขวดน้ำแถมแก้ว"
"ส่วนลดสำหรับลูกค้าเอเย่นต์"
"ของแถมเบียร์ช้าง"
```

**Parameters:**
- `top_k`: Number of results (default: 5)
- `similarity_threshold`: Minimum score (default: 0.7)

**Expected Performance:**
- Response time: <100ms
- Accuracy: High for semantic queries
- Memory: Loads all 142 embeddings into memory (~5MB)

**Implementation Requirements:**
- Use LlamaIndex VectorStoreIndex.as_retriever()
- Support both IndexNode and Summary embeddings
- Enhance metadata with retriever_type and similarity_score

### 3.2 Retrieval Strategy 2: Hybrid Search

**Purpose:** Combine semantic vector search with Thai keyword matching

**Best For:**
- Queries needing both semantic and exact matches
- Product names, promotion IDs, specific terms
- When users mention specific Thai keywords
- Best overall performance for mixed content

**How It Works:**
1. **Vector Search**: Semantic similarity (weight: 0.7)
2. **Keyword Search**: Thai BM25 scoring (weight: 0.3)
3. **Fusion**: Reciprocal Rank Fusion (RRF) or weighted combination
4. Return unified ranked results

**Thai Keyword Extraction:**
- Extract Thai characters (Unicode \u0e00-\u0e7f)
- Tokenize: customer types (เอเย่นต์, ซับเอเย่นต์, ลูกค้าปลีก)
- Product categories: น้ำดื่ม, เบียร์, สุรา
- Promotion types: ซื้อ 1 แถม 1, ลด X%, แถมของ

**Example Queries:**
```
"โปรโมชัน AND เบียร์ช้าง"
"ส่วนลดมากกว่า 20% สำหรับน้ำดื่ม"
"ซื้อ 2 แถม 1 เฉพาะเอเย่นต์"
```

**Parameters:**
- `alpha`: Vector weight (default: 0.7)
- `top_k`: Results per method (default: 10)
- `fusion_mode`: "rrf", "weighted", "borda" (default: "rrf")

**Expected Performance:**
- Response time: <200ms
- Accuracy: Highest for mixed queries
- Memory: Vector index + BM25 inverted index (~6MB)

**Implementation Requirements:**
- Build custom BM25 index from Thai text
- Implement RRF fusion algorithm
- Support Thai tokenization
- Handle Unicode edge cases

### 3.3 Retrieval Strategy 3: Metadata Filtering

**Purpose:** Fast structured queries using 29 metadata fields

**Best For:**
- Attribute-based queries
- Numeric range filtering
- Date-based queries
- Geographic/categorical filters
- When users specify exact criteria

**Fast Metadata Indexing:**
- **Inverted indices** for categorical fields
- **B-tree indices** for numeric fields
- **Range queries** for dates and percentages
- **Sub-50ms** filtering before vector search

**Indexed Fields (29 total):**

**Categorical (inverted indices):**
- customer_type: เอเย่นต์, ซับเอเย่นต์, ลูกค้าขายส่ง, ลูกค้าปลีก
- product_category: water, beer, spirits
- product_brand: Chang, Sangsom, Leo, etc.
- promotion_type: buy_X_get_Y_free, discount_percentage, bundle, gift_with_purchase

**Numeric (B-tree indices):**
- buy_quantity, free_quantity: 1-999
- discount_percentage: 0-100%
- duration_days: 1-365

**Boolean:**
- is_active, is_wholesale, is_retail, expires_soon

**Dates:**
- start_date, end_date (Thai + Gregorian)

**Example Queries:**
```
"โปรโมชันที่ยังใช้งานได้สำหรับเอเย่นต์"
"ส่วนลดมากกว่า 30% สำหรับเบียร์"
"โปรโมชันที่จะหมดอายุภายใน 7 วัน"
"แคมเปญสำหรับน้ำดื่มช้างที่เปิดใช้งาน"
```

**Filter Extraction (LLM-based):**
```python
Query: "ส่วนลด 20% ขึ้นไปสำหรับเอเย่นต์"
Filters: {
    "discount_percentage": {"min": 20},
    "customer_type": "เอเย่นต์",
    "is_active": True
}
```

**Parameters:**
- `enable_fast_filtering`: Enable fast indexing (default: True)
- `top_k`: Results after filtering (default: 5)
- `auto_detect_filters`: Use LLM to extract filters (default: True)

**Expected Performance:**
- Filtering time: <50ms for 142 embeddings
- Document reduction: 45-90% before vector search
- Memory overhead: <10% of vector store size
- Response time: <150ms total

**Implementation Requirements:**
- Build inverted and B-tree indices from nodes
- Auto-initialize during CLI startup
- LLM-based filter extraction from natural language
- Support range queries and multiple filters

### 3.4 Retrieval Strategy 4: Summary Retrieval

**Purpose:** Two-stage hierarchical retrieval using document summaries

**Best For:**
- Document-level overview questions
- "Which promotions..." queries
- When exploring available campaigns
- High-level information retrieval

**How It Works:**
1. **Stage 1**: Retrieve top-K promotions by summary similarity
2. **Stage 2**: Get IndexNodes from selected documents
3. Combine results with summary context

**Example Queries:**
```
"มีโปรโมชันอะไรบ้างสำหรับเบียร์ช้าง"
"สรุปแคมเปญที่เปิดอยู่ในเดือนนี้"
"โปรโมชันไหนบ้างที่ให้ของแถม"
```

**Parameters:**
- `summary_top_k`: Documents to retrieve (default: 5)
- `nodes_per_doc`: IndexNodes per document (default: 2)

**Expected Performance:**
- Response time: <300ms
- Accuracy: High for document-level questions
- Memory: Both summary and IndexNode embeddings (~10MB)

**Implementation Requirements:**
- Load both summary and IndexNode embeddings
- Use DocumentSummaryIndex retriever
- Preserve GPT-5 generated summaries in results

### 3.5 Retrieval Strategy 5: Recursive Retrieval

**Purpose:** Multi-level hierarchical retrieval with drill-down

**Best For:**
- Complex hierarchical queries
- Multi-step information needs
- When relationships matter
- Detailed exploration

**How It Works:**
1. **Level 1**: Retrieve document summaries
2. **Level 2**: For each summary, retrieve related IndexNodes
3. **Level 3**: (Optional) Retrieve chunks if chunking exists
4. Recursive query decomposition

**Example Queries:**
```
"อธิบายโครงสร้างโปรโมชันของเบียร์ช้างและเงื่อนไขทั้งหมด"
"เปรียบเทียบโปรโมชันน้ำดื่มกับสุราในแง่ของส่วนลด"
```

**Parameters:**
- `max_depth`: Recursion levels (default: 2)
- `top_k_per_level`: Results per level (default: 3)

**Expected Performance:**
- Response time: <1s (multiple retrieval steps)
- Accuracy: High for hierarchical queries
- Complexity: Moderate (multiple LLM calls)

**Implementation Requirements:**
- Build RecursiveRetriever from IndexNodes
- Handle multi-level query decomposition
- Preserve document hierarchy in results

### 3.6 Retrieval Strategy 6: Chunk Decoupling

**Purpose:** Separate embedding chunks from content chunks for better context

**Best For:**
- Detailed chunk-level queries
- When precise information is needed
- Context preservation matters
- Specific detail extraction

**How It Works:**
1. Retrieve using IndexNode embeddings (summaries)
2. Return full document content for context
3. Better context than flat chunking

**Note:** For sales promotions (atomic documents), this strategy:
- Uses IndexNode for retrieval
- Returns full document text for context
- Preserves all metadata fields

**Example Queries:**
```
"รายละเอียดเงื่อนไขการรับของแถมแบบละเอียด"
"อธิบายข้อกำหนดการซื้อสำหรับโปรโมชันนี้"
```

**Parameters:**
- `chunk_top_k`: Chunks to retrieve (default: 5)
- `context_window`: Content window size (default: full doc)

**Expected Performance:**
- Response time: <250ms
- Accuracy: High for detailed queries
- Memory: Same as vector search

**Implementation Requirements:**
- Build AutoMergingRetriever pattern
- Map IndexNodes to source documents
- Preserve full metadata in results

### 3.7 Retrieval Strategy 7: Query Planner

**Purpose:** Multi-step query decomposition and execution

**Best For:**
- Complex analytical questions
- Multi-part queries
- Comparison tasks
- When simple retrieval is insufficient

**How It Works:**
1. LLM analyzes query complexity
2. Decomposes into sub-queries
3. Executes each sub-query with appropriate strategy
4. Synthesizes final answer

**Example Queries:**
```
"เปรียบเทียบส่วนลดระหว่างเอเย่นต์กับลูกค้าปลีก จากนั้นบอกว่าโปรโมชันไหนคุ้มที่สุด"
"หาโปรโมชันเบียร์ที่มีของแถม แล้วอธิบายเงื่อนไขและระยะเวลาของแต่ละโปรโมชัน"
"วิเคราะห์ว่าโปรโมชันไหนมีเงื่อนไขซับซ้อนที่สุด และทำไม"
```

**Query Decomposition Example:**
```
Original: "เปรียบเทียบส่วนลดน้ำดื่มกับเบียร์สำหรับเอเย่นต์"

Decomposed:
1. "ส่วนลดน้ำดื่มสำหรับเอเย่นต์" → Metadata filtering
2. "ส่วนลดเบียร์สำหรับเอเย่นต์" → Metadata filtering
3. Compare results and synthesize answer
```

**Parameters:**
- `max_steps`: Maximum decomposition steps (default: 5)
- `use_llm`: Enable LLM planning (default: True)
- `fallback_strategy`: If planning fails (default: "hybrid")

**Expected Performance:**
- Response time: <2s (multiple LLM calls)
- Accuracy: Highest for complex queries
- Cost: Higher (multiple API calls)

**Implementation Requirements:**
- Implement LLMMultiStepQueryEngine
- Query decomposition with GPT-5
- Sub-query routing to appropriate strategies
- Result synthesis

---

## 4. Technical Architecture

### 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Layer                          │
│  ┌──────────────┐              ┌─────────────────────────┐  │
│  │  CLI (cli.py)│              │ Streamlit UI            │  │
│  │  - Interactive│              │ (streamlit_cli.py)      │  │
│  │  - Single query│             │ - Chat interface        │  │
│  │  - Batch mode │              │ - Strategy viz          │  │
│  └──────┬───────┘              └───────────┬─────────────┘  │
└─────────┼──────────────────────────────────┼────────────────┘
          │                                  │
          └────────────────┬─────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────┐
│                    Router Layer                           │
│           ┌──────────────▼───────────────┐                │
│           │  SalesPromotionRouterRetriever  │              │
│           │  (router.py - Two-stage routing)│             │
│           └──────────┬────────┬───────────┘                │
│                      │        │                            │
│      ┌───────────────┘        └──────────────┐             │
│      │                                       │             │
│  ┌───▼─────────────────┐        ┌───────────▼──────────┐  │
│  │ Stage 1:            │        │ Stage 2:             │  │
│  │ Index Classification│        │ Strategy Selection   │  │
│  │ (index_classifier.py)│       │ - LLM-based         │  │
│  │                     │        │ - Heuristic fallback │  │
│  │ Output: "sales_promo"│       │ - Round-robin test   │  │
│  └─────────────────────┘        └──────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼─────────────────────────┐
│               Strategy Layer (7 Retrievers)                │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐              │
│  │ Vector   │  │ Hybrid   │  │ Metadata   │              │
│  │ (vector.py)│ (hybrid.py)│ (metadata.py)│              │
│  │ Fast     │  │ Thai+Vec │  │ Fast Index │              │
│  └──────────┘  └──────────┘  └────────────┘              │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐              │
│  │ Summary  │  │Recursive │  │Chunk Decouple│            │
│  │(summary.py)│(recursive.py)│(chunk_decoup.py)│         │
│  │Hierarchical│ │Multi-level│ │Context Pres.│            │
│  └──────────┘  └──────────┘  └────────────┘              │
│                                                            │
│  ┌────────────────────────┐                               │
│  │ Planner                │                               │
│  │ (planner.py)           │                               │
│  │ Multi-step reasoning   │                               │
│  └────────────────────────┘                               │
└────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼─────────────────────────┐
│                    Data Layer                              │
│                                                            │
│  ┌───────────────────┐    ┌──────────────────────────┐    │
│  │ Fast Metadata     │    │ Vector Index             │    │
│  │ Index Manager     │    │ (LlamaIndex)             │    │
│  │ (fast_metadata_   │    │ - 142 embeddings         │    │
│  │  index.py)        │    │ - 3072 dimensions        │    │
│  │                   │    │ - IndexNode + Summary    │    │
│  │ - Inverted indices│    └──────────────────────────┘    │
│  │ - B-tree indices  │                                     │
│  │ - <50ms filtering │    ┌──────────────────────────┐    │
│  └───────────────────┘    │ Cache Manager            │    │
│                           │ - Query cache            │    │
│                           │ - Result cache           │    │
│                           │ - TTL: 1 hour            │    │
│                           └──────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼─────────────────────────┐
│                 Storage Layer (Phase 2 Output)             │
│                                                            │
│  data/embedding/embeddings_sales_20251026_172927/         │
│  ├── batch_1/ (PKL, NPY, JSON)                            │
│  ├── batch_2/ (PKL, NPY, JSON)                            │
│  ├── batch_3/ (PKL, NPY, JSON)                            │
│  ├── batch_4/ (PKL, NPY, JSON)                            │
│  └── combined_statistics.json                             │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Component Relationships

**Router Orchestration:**
```python
# Initialization flow
router = SalesPromotionRouterRetriever(
    retrievers={
        "sales_promotions": {
            "vector": VectorRetrieverAdapter(index),
            "hybrid": HybridRetrieverAdapter(index),
            "metadata": MetadataRetrieverAdapter(index, enable_fast_filtering=True),
            "summary": SummaryRetrieverAdapter(doc_summary_index),
            "recursive": RecursiveRetrieverAdapter(index_nodes),
            "chunk_decoupling": ChunkDecouplingRetrieverAdapter(index),
            "planner": PlannerRetrieverAdapter(index, llm=llm)
        }
    },
    index_classifier=SalesPromotionIndexClassifier(),
    strategy_selector="llm",  # or "heuristic", "round_robin"
    enable_caching=True
)

# Query execution flow
results = router.retrieve("ส่วนลดสำหรับเอเย่นต์")
# → Index Classifier: "sales_promotions"
# → Strategy Selector: "metadata" (LLM decides)
# → MetadataRetrieverAdapter.retrieve()
# → Fast filtering: customer_type="เอเย่นต์"
# → Vector search on filtered subset
# → Return NodeWithScore results
```

### 4.3 Data Flow

**Query Processing Pipeline:**

```
User Query
    │
    ▼
┌───────────────────────────┐
│ 1. Query Preprocessing    │
│ - Thai text normalization │
│ - Keyword extraction      │
│ - Query embedding         │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 2. Cache Check            │
│ - Query hash lookup       │
│ - Return if cache hit     │
└───────────┬───────────────┘
            │ (cache miss)
            ▼
┌───────────────────────────┐
│ 3. Stage 1: Index         │
│    Classification         │
│ - Analyze query intent    │
│ - Select: "sales_promotions"│
│ - Confidence score        │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 4. Stage 2: Strategy      │
│    Selection              │
│ - LLM analyzes query      │
│ - Considers:              │
│   • Complexity            │
│   • Thai keywords         │
│   • Metadata hints        │
│   • Performance needs     │
│ - Selects best strategy   │
│ - Fallback: heuristics    │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 5. Strategy Execution     │
│ - Call retriever._retrieve()│
│ - Strategy-specific logic │
│ - Metadata enhancement    │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 6. Post-processing        │
│ - Filter by threshold     │
│ - Re-rank if needed       │
│ - Add metadata            │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 7. Response Synthesis     │
│ - Format results          │
│ - Generate summary        │
│ - Add confidence scores   │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 8. Cache Update           │
│ - Store query → results   │
│ - Set TTL: 1 hour         │
└───────────┬───────────────┘
            │
            ▼
    Return to User
```

---

## 5. Data Flow Specifications

### 5.1 Embedding Loading Flow

**Process:** Load Phase 2 outputs into LlamaIndex structures

```python
# 1. Load PKL files (complete objects)
def load_embeddings_from_pkl(embeddings_dir: Path) -> Dict[str, List]:
    """Load all batch PKL files."""
    indexnode_embeddings = []
    summary_embeddings = []

    for batch_dir in embeddings_dir.glob("batch_*/"):
        # Load IndexNode embeddings
        pkl_file = batch_dir / "indexnodes" / f"{batch_dir.name}_indexnodes_full.pkl"
        with open(pkl_file, 'rb') as f:
            batch_indexnodes = pickle.load(f)
            indexnode_embeddings.extend(batch_indexnodes)

        # Load Summary embeddings
        pkl_file = batch_dir / "summaries" / f"{batch_dir.name}_summaries_full.pkl"
        with open(pkl_file, 'rb') as f:
            batch_summaries = pickle.load(f)
            summary_embeddings.extend(batch_summaries)

    return {
        "indexnodes": indexnode_embeddings,  # 71 embeddings
        "summaries": summary_embeddings      # 71 embeddings
    }

# 2. Build LlamaIndex structures
def build_vector_index(embeddings: List[Dict]) -> VectorStoreIndex:
    """Build VectorStoreIndex from embeddings."""
    nodes = []
    for emb in embeddings:
        node = IndexNode(
            text=emb['text'],
            embedding=emb['embedding_vector'],  # 3072 dims
            metadata=emb['metadata']            # 29 fields
        )
        nodes.append(node)

    # Build index
    index = VectorStoreIndex(nodes)
    return index

# 3. Build DocumentSummaryIndex
def build_document_summary_index(embeddings: Dict) -> DocumentSummaryIndex:
    """Build DocumentSummaryIndex for summary retrieval."""
    # Use existing summaries from Phase 2
    # GPT-5 summaries already generated
    pass
```

---

## 6. Module Specifications

### 6.1 router.py

**Purpose:** Main router orchestrating two-stage retrieval process

**Key Class:** `SalesPromotionRouterRetriever(BaseRetriever)`

**Responsibilities:**
1. Accept queries from CLI/UI
2. Execute two-stage routing (index → strategy)
3. Call selected retriever
4. Manage caching
5. Log performance metrics
6. Handle errors and fallbacks

**Estimated Size:** ~300 lines

**Dependencies:**
- llama_index.core.base.base_retriever.BaseRetriever
- index_classifier.SalesPromotionIndexClassifier
- retrievers.base.BaseRetrieverAdapter
- OpenAI LLM for strategy selection

### 6.2 index_classifier.py

**Purpose:** Stage 1 routing - classify queries to appropriate indices

**Note:** For Phase 3, only one index exists ("sales_promotions"). This module provides architecture for future multi-index support.

**Key Class:** `SalesPromotionIndexClassifier`

**Estimated Size:** ~200 lines

**Dependencies:**
- OpenAI LLM (optional, for multi-index future)
- Minimal for Phase 3 (single index)

### 6.3 fast_metadata_index.py

**Purpose:** Sub-50ms metadata filtering using inverted and B-tree indices

**Key Class:** `FastMetadataIndexManager`

**Estimated Size:** ~250 lines

**Dependencies:**
- sortedcontainers.SortedList (for B-tree)
- collections.defaultdict
- llama_index.core.schema.BaseNode

### 6.4 cli.py

**Purpose:** Interactive command-line interface for querying promotions

**Modes:**
1. Interactive mode (REPL)
2. Single query mode
3. Batch query mode
4. Test queries mode
5. Benchmark mode

**Estimated Size:** ~400 lines

**Dependencies:**
- argparse
- router.SalesPromotionRouterRetriever
- Rich library for formatted output (optional)

### 6.5 streamlit_cli.py

**Purpose:** WhatsApp-style web interface for queries

**Features:**
1. Chat-style conversation interface
2. Strategy visualization
3. Confidence scores display
4. Metadata inspection
5. Export results to CSV/JSON
6. Query history

**Estimated Size:** ~300 lines

**Dependencies:**
- streamlit
- router.SalesPromotionRouterRetriever

### 6.6 Retriever Modules

**All retrievers inherit from:** `retrievers/base.py::BaseRetrieverAdapter`

**Estimated Sizes:**
- vector.py: ~80 lines (complete from 60)
- hybrid.py: ~200 lines (Thai keyword extraction, BM25, RRF)
- metadata.py: ~250 lines (fast indexing, LLM filter extraction)
- summary.py: ~150 lines (DocumentSummaryIndex integration)
- recursive.py: ~180 lines (multi-level retrieval)
- chunk_decoupling.py: ~140 lines (context preservation)
- planner.py: ~200 lines (multi-step query planning)

**Total for all 7 retrievers:** ~1,400 lines

---

## 7. Configuration & Environment

### 7.1 Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-proj-...

# Retrieval Configuration
TOP_K=5                          # Default number of results
SIMILARITY_THRESHOLD=0.7         # Minimum similarity score
ENABLE_FAST_FILTERING=true       # Enable fast metadata indexing
ENABLE_CACHING=true              # Enable query caching
CACHE_TTL_SECONDS=3600           # Cache expiration (1 hour)

# Strategy Selection
CLASSIFIER_MODE=llm              # "llm" or "embedding"
STRATEGY_SELECTOR=llm            # "llm", "heuristic", "round_robin"
LLM_STRATEGY_MODE=enhanced       # "simple" or "enhanced"

# Models (from Phase 2)
EMBEDDING_MODEL=text-embedding-3-large  # 3072 dimensions
LLM_MODEL=gpt-5                         # For summaries and routing

# Hybrid Search
HYBRID_ALPHA=0.7                 # Vector weight (1-alpha = keyword weight)
HYBRID_FUSION_MODE=rrf           # "rrf", "weighted", "borda"

# Metadata Filtering
AUTO_DETECT_FILTERS=true         # LLM-based filter extraction

# Performance
MAX_WORKERS=4                    # Parallel processing
REQUEST_DELAY=0.1                # Rate limiting (seconds)

# Debug
DEBUG_LOGGING=false              # Enable detailed logs
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
```

---

## 8. Output Specifications

### 8.1 CLI Output Format

**Interactive Mode:**
```
===============================================================
Sales Promotion Retrieval System - Interactive Mode
===============================================================
Type your query in Thai or English, or 'quit' to exit
Special commands:
  /help - Show help
  /stats - Show system stats
  /clear - Clear cache
  /strategy <name> - Force strategy

🔍 Query: ส่วนลดสำหรับเอเย่นต์

===============================================================
Results for: ส่วนลดสำหรับเอเย่นต์
Latency: 142.3ms | Results: 5 | Strategy: metadata
===============================================================

1. เบียร์ช้าง 320 ml
   Customer: เอเย่นต์
   Promotion: discount_percentage
   Discount: 15%
   Buy: 1 ลัง
   Free: 0
   Period: 1 พฤศจิกายน 2567 - 30 พฤศจิกายน 2567
   Active: ✅
   Score: 0.923
   Strategy: metadata
   Filters: {"customer_type": "เอเย่นต์"}

...

===============================================================
```

---

## 9. Quality Assurance & Validation

### 9.1 Unit Tests

**Test Coverage Requirements:**
- Router: 90%+
- Each retriever: 85%+
- Fast metadata indexing: 95%+
- CLI: 80%+

**Test Files:**
```
src-sale-promotion/test/
├── test_router.py
├── test_index_classifier.py
├── test_fast_metadata_index.py
├── test_retrievers/
│   ├── test_vector.py
│   ├── test_hybrid.py
│   ├── test_metadata.py
│   ├── test_summary.py
│   ├── test_recursive.py
│   ├── test_chunk_decoupling.py
│   └── test_planner.py
└── test_cli.py
```

---

## 10. Success Criteria

### 10.1 Functional Requirements

| Requirement | Target | Measurement Method |
|------------|--------|-------------------|
| **All 7 strategies implemented** | 7/7 complete | Code review + unit tests |
| **Router functional** | 100% | Integration tests |
| **CLI operational** | All modes working | Manual + automated tests |
| **Streamlit UI functional** | Full feature set | Manual testing |
| **Thai language support** | 100% queries | UTF-8 tests + manual |
| **Metadata filtering** | All 29 fields | Unit tests |

### 10.2 Performance Requirements

| Metric | Target | Acceptable | Measurement |
|--------|--------|-----------|-------------|
| **Vector Search** | <100ms | <150ms | Benchmark suite |
| **Hybrid Search** | <200ms | <300ms | Benchmark suite |
| **Metadata Filtering** | <50ms filter | <100ms | Fast index tests |
| **Metadata Total** | <150ms | <250ms | End-to-end tests |
| **Summary Retrieval** | <300ms | <500ms | Benchmark suite |
| **Recursive Retrieval** | <1s | <2s | Benchmark suite |
| **Chunk Decoupling** | <250ms | <400ms | Benchmark suite |
| **Query Planner** | <2s | <5s | Benchmark suite |
| **Cache Hit** | <10ms | <50ms | Cache tests |

---

## 11. Implementation Plan

### 11.1 Timeline Overview

**Total Duration:** 2 weeks (10 working days)

**Phases:**
1. **Week 1 (Days 1-5):** Core Infrastructure + Basic Retrievers
2. **Week 2 (Days 6-10):** Advanced Retrievers + UIs + Testing

### 11.2 Detailed Schedule

#### Week 1: Foundation

**Day 1-2: Core Infrastructure**
- ✅ Implement `router.py` (~300 lines)
- ✅ Implement `index_classifier.py` (~200 lines)
- ✅ Unit tests for router and classifier

**Day 3: Fast Metadata Indexing**
- ✅ Implement `fast_metadata_index.py` (~250 lines)
- ✅ Performance tests (<50ms target)
- ✅ Integration with metadata retriever

**Day 4: Basic Retrievers**
- ✅ Complete `vector.py` (enhance existing)
- ✅ Implement `hybrid.py` (~200 lines)
- ✅ Implement `metadata.py` (~250 lines)
- ✅ Unit tests

**Day 5: Advanced Retrievers (Part 1)**
- ✅ Implement `summary.py` (~150 lines)
- ✅ Implement `recursive.py` (~180 lines)
- ✅ Unit tests

#### Week 2: Advanced Features

**Day 6: Advanced Retrievers (Part 2)**
- ✅ Implement `chunk_decoupling.py` (~140 lines)
- ✅ Implement `planner.py` (~200 lines)
- ✅ Unit tests

**Day 7: CLI Implementation**
- ✅ Complete `cli.py` (~400 lines)
- ✅ CLI integration tests

**Day 8: Streamlit UI**
- ✅ Implement `streamlit_cli.py` (~300 lines)
- ✅ Manual testing

**Day 9: Integration & Testing**
- ✅ End-to-end integration tests
- ✅ Performance benchmarking
- ✅ Thai language tests
- ✅ Fix bugs and issues

**Day 10: Polish & Delivery**
- ✅ Code review and cleanup
- ✅ Final testing pass
- ✅ User acceptance testing
- ✅ Documentation finalization

### 11.3 Deliverables

**Code Deliverables:**
```
src-sale-promotion/retrieval/
├── router.py (300 lines) ✅
├── index_classifier.py (200 lines) ✅
├── fast_metadata_index.py (250 lines) ✅
├── cli.py (400 lines) ✅
├── streamlit_cli.py (300 lines) ✅
├── retrievers/
│   ├── vector.py (80 lines) ✅
│   ├── hybrid.py (200 lines) ✅
│   ├── metadata.py (250 lines) ✅
│   ├── summary.py (150 lines) ✅
│   ├── recursive.py (180 lines) ✅
│   ├── chunk_decoupling.py (140 lines) ✅
│   └── planner.py (200 lines) ✅

Total: ~2,650 lines of production code
```

---

## 12. Performance Requirements

### 12.1 Latency Targets

**Response Time Targets (End-to-End):**

| Strategy | Target (ms) | Acceptable (ms) | P95 (ms) |
|----------|------------|----------------|----------|
| Vector | <100 | <150 | <200 |
| Hybrid | <200 | <300 | <400 |
| Metadata | <150 | <250 | <350 |
| Summary | <300 | <500 | <700 |
| Recursive | <1000 | <2000 | <3000 |
| Chunk Decoupling | <250 | <400 | <500 |
| Planner | <2000 | <5000 | <7000 |

---

## 13. Error Handling & Recovery

### 13.1 Error Scenarios

**1. Embeddings Not Found**
- Display clear error message
- Suggest running Phase 2 first

**2. LLM API Errors**
- Fall back to heuristic routing
- Log warning and continue

**3. Empty Results**
- Provide suggestions to user
- Log warning

**4. Thai Encoding Errors**
- Fall back to basic string handling
- Log error

### 13.2 Graceful Degradation

**Priority Chain:**
```
1. Try LLM-based routing → Success ✅
2. If LLM fails → Heuristic routing ⚠️
3. If heuristic unsure → Default to hybrid 📊
4. If hybrid fails → Fall back to vector ⚡
5. If vector fails → Return error ❌
```

---

## 14. Appendices

### Appendix A: Thai Query Examples

**Customer Type Queries:**
```
เอเย่นต์ - Agent
ซับเอเย่นต์ - Sub-agent
ลูกค้าขายส่ง - Wholesale customer
ลูกค้าปลีก - Retail customer
```

**Product Category Queries:**
```
น้ำดื่ม - Drinking water
เบียร์ - Beer
สุรา - Spirits
เครื่องดื่ม - Beverage
```

**Promotion Type Queries:**
```
ซื้อ 1 แถม 1 - Buy 1 get 1 free
ส่วนลด 20% - 20% discount
แถมของ - Free gift
ลดราคา - Price reduction
```

**Complex Queries:**
```
โปรโมชันสำหรับเอเย่นต์ที่ให้ส่วนลดมากกว่า 20%
เปรียบเทียบโปรโมชันน้ำดื่มกับเบียร์
โปรโมชันที่จะหมดอายุภายใน 7 วัน
แคมเปญแบบซื้อ 2 แถม 1 สำหรับเบียร์ช้าง
```

### Appendix B: Metadata Field Reference

**Complete Metadata Schema (29 fields):**

**Customer Fields (5):**
- document_id, customer_type, customer_tier, is_wholesale, is_retail

**Product Fields (4):**
- product_name, product_category, product_brand, product_size

**Promotion Fields (8):**
- buy_quantity, buy_unit, free_item_name, free_quantity, free_unit, discount_percentage, promotion_type, additional_conditions

**Date Fields (8):**
- start_date, end_date, start_date_gregorian, end_date_gregorian, is_active, expires_soon, duration_days, keywords

**Additional Fields (4):**
- doc_id, doc_title, type, batch_number

---

**END OF DOCUMENT**

---

**Total Lines:** ~2,600 lines
**Status:** DRAFT - Ready for Review
**Next Phase:** Phase 4 - Production API & Deployment
