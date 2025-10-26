# Sales Promotion RAG Pipeline - Streamlined Methodology

**Purpose:** Implementation guide for building a RAG pipeline specifically optimized for sales promotion data.

**Based on:** Proven methodology from `src/METHODOLOGY.md` and `attached_assets/` documentation, adapted for atomic transactional promotion data.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Differences from General RAG](#key-differences-from-general-rag)
3. [Data Characteristics](#data-characteristics)
4. [Streamlined 4-Phase Approach](#streamlined-4-phase-approach)
5. [Sales-Specific Adaptations](#sales-specific-adaptations)
6. [Core Retrieval Strategies (3-4)](#core-retrieval-strategies)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Expected Query Patterns](#expected-query-patterns)
9. [Configuration Guide](#configuration-guide)
10. [Testing Strategy](#testing-strategy)

---

## Overview

This methodology adapts the battle-tested 7-strategy RAG approach (documented in `src/METHODOLOGY.md`) for **sales promotion data**, which has fundamentally different characteristics:

### What Makes Promotions Different?

**General RAG (src/):**
- Long documents (~1 page, 500-1000 tokens)
- Hierarchical structure (document → summary → chunks)
- Need for document-level reasoning
- 7 retrieval strategies

**Sales Promotions:**
- **Atomic records** (50-150 tokens per promotion)
- **Flat structure** (no hierarchy)
- **Highly structured** metadata
- **3-4 retrieval strategies** sufficient
- **Date-sensitive** (active/expired promotions)
- **Quantity-based logic** (buy X get Y)

### Success Criteria

✅ Fast metadata filtering for structured queries (date, customer type, product)
✅ Accurate Thai product name matching (exact + semantic)
✅ Active promotion filtering (Buddhist calendar support)
✅ Quantity-based recommendations
✅ Simple, maintainable codebase

---

## Key Differences from General RAG

| Aspect | General RAG (`src/`) | Sales Promotions |
|--------|---------------------|------------------|
| **Document Size** | ~500-1000 tokens | ~50-150 tokens |
| **Structure** | Hierarchical | Atomic (flat) |
| **Chunking** | Required (512/50) | ❌ Not needed |
| **Document Summaries** | Generated via LLM | ❌ Not needed |
| **IndexNodes** | Multi-level | ❌ Not needed |
| **Retrieval Strategies** | 7 strategies | ✅ 3-4 strategies |
| **Primary Strategy** | Hybrid search | ✅ Metadata filtering |
| **Metadata Importance** | Medium | ⭐ **CRITICAL** |
| **Date Filtering** | Optional | ⭐ **ESSENTIAL** |

**Bottom Line:** Sales promotions are **simpler but require precise metadata handling**.

---

## Data Characteristics

### Input Data: `data/promotion_list.csv`

**Schema:**
```csv
ประเภทลูกค้า,ระยะเวลาเริ่ม,ระยะเวลาสื้นสุด,สินค้า,ซื้อ,หน่วยซื้อ,รายการแถม,แถม,หน่วยแถม,เงื่อนไขเพิ่มเติม
```

**Example Record:**
```csv
เอเย่นต์,2568-10-01 00:00:00,2568-12-31 00:00:00,น้ำดื่มช้าง PET 600 CC,108,แพ็ค,น้ำดื่มช้าง PET 600 CC,42,แพ็ค,
```

### Key Features

1. **Thai Language**
   - Customer types: เอเย่นต์, ซับเอเย่นต์, รถขายปลีก, CVA, Retail TT, Super Sub
   - Products: น้ำดื่มช้าง, เบียร์ช้าง, สุราขาว, นิยมไทย, ซาโต้โคล่า
   - Units: แพ็ค, ลัง, ขวด, ถาด, กระป๋อง

2. **Buddhist Calendar**
   - Dates in B.E. (Buddhist Era): 2568 = 2025 CE
   - Need conversion for active promotion filtering

3. **Structured Promotion Rules**
   - Buy X get Y (ซื้อ/แถม)
   - Conditional promotions (เงื่อนไขเพิ่มเติม)
   - Time-bound offers

4. **Customer Segmentation**
   - Clear hierarchies (Agent → Sub-agent → Retail)
   - Different promotions per tier

---

## Streamlined 4-Phase Approach

### Phase 1: Data Preparation
**Reference:** `src/METHODOLOGY.md` Phase 1, `attached_assets/00_doc_prep_workflow.md`

**Adapt Field Mappings for Promotions:**

```python
from src_sale_promotion.data_processing import PromotionDataProcessor

field_mappings = {
    'temporal': {
        'start_date': 'ระยะเวลาเริ่ม',
        'end_date': 'ระยะเวลาสื้นสุด',
    },
    'customer_segment': {
        'customer_type': 'ประเภทลูกค้า',
    },
    'product': {
        'product_name': 'สินค้า',
        'free_item_name': 'รายการแถม',
    },
    'promotion_terms': {
        'buy_quantity': 'ซื้อ',
        'buy_unit': 'หน่วยซื้อ',
        'free_quantity': 'แถม',
        'free_unit': 'หน่วยแถม',
    },
    'conditions': {
        'additional_conditions': 'เงื่อนไขเพิ่มเติม',
    }
}
```

**Derived Metadata (Critical!):**

```python
derived_metadata = {
    # Date handling
    'is_active': is_within_date_range(start_date, end_date),
    'start_date_gregorian': convert_buddhist_to_gregorian(start_date),
    'end_date_gregorian': convert_buddhist_to_gregorian(end_date),
    'duration_days': calculate_duration(start_date, end_date),
    'expires_soon': is_expiring_within_days(end_date, 7),

    # Product categorization
    'product_category': extract_category(product_name),  # water, beer, spirits, food
    'product_size': extract_size(product_name),  # 600CC, 1500CC, 330มล
    'product_brand': extract_brand(product_name),  # ช้าง, นิยมไทย, ซาโต้

    # Promotion analysis
    'promotion_type': 'buy_x_get_y',
    'discount_percentage': calculate_discount_percentage(buy_qty, free_qty),
    'effective_price_per_unit': calculate_effective_price(buy_qty, free_qty),

    # Customer tier
    'customer_tier': normalize_customer_type(customer_type),  # tier_1, tier_2, tier_3
    'is_wholesale': customer_type in ['เอเย่นต์', 'ซับเอเย่นต์', 'Super Sub'],
    'is_retail': customer_type in ['รถขายปลีก', 'CVA', 'Retail TT'],

    # Searchability
    'search_text': generate_search_text(all_fields),  # Thai + transliterated
    'keywords': extract_keywords(product_name, customer_type),
}
```

**Output:**
- Markdown files with rich metadata (NO chunking needed)
- Configuration YAML for reproducibility
- Document list (each promotion = one document)

---

### Phase 2: Embedding Generation (Simplified)
**Reference:** `src/METHODOLOGY.md` Phase 2, `attached_assets/06_RAG_pipeline_50k_strategy.md`

**Key Simplification:** Skip DocumentSummaryIndex, skip IndexNodes

```python
from src_sale_promotion.docs_embedding import PromotionEmbeddingPipeline

# Simplified pipeline (no hierarchical index)
pipeline = PromotionEmbeddingPipeline(
    embed_model="text-embedding-3-small",
    batch_size=50,  # Process 50 promotions at a time
    vector_store=vector_store  # ChromaDB or pgvector
)

# Single-level embedding (no chunks, no summaries)
embeddings = pipeline.process_promotions(
    documents=promotion_documents,
    preserve_metadata=True  # CRITICAL
)
```

**What We Create:**
1. **Vector embeddings** of full promotion text
2. **Rich metadata** attached to each embedding
3. **Persistent storage** (ChromaDB/pgvector)

**What We Skip:**
- ❌ Document summaries (promotions are already atomic)
- ❌ Chunking (each promotion fits in context window)
- ❌ IndexNodes (no hierarchical retrieval needed)

**Batch Processing:**
```python
# For large datasets (1000+ promotions)
for batch in batch_iterator(promotions, batch_size=50):
    embeddings = generate_embeddings(batch)
    vector_store.add(embeddings)
    # Each batch persisted immediately
```

---

### Phase 3: Core Retrieval Strategies (3-4 Only)
**Reference:** `src/METHODOLOGY.md` Phase 3, `attached_assets/03_retrieval_pipeline_plan.md`

#### **Strategy 1: Metadata Filtering** ⭐⭐⭐ (PRIMARY)
**Script:** `src-sale-promotion/retrieval/retrievers/metadata.py`

**Why PRIMARY for promotions:**
- 80% of queries are structured: "โปรโมชั่นสำหรับเอเย่นต์", "ดีลเดือนนี้", "น้ำดื่มช้างโปรโมชั่น"
- Fast pre-filtering before vector search
- Essential for date range queries

**Implementation:**
```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

class PromotionMetadataRetriever:
    def retrieve(self, query: str, top_k: int = 10):
        # Auto-detect filters from query
        filters = self._infer_filters(query)

        # Examples:
        # "โปรโมชั่นเอเย่นต์" → filter: customer_type == "เอเย่นต์"
        # "ดีลเดือนนี้" → filter: is_active == True
        # "น้ำดื่มช้าง" → filter: product_category == "water"

        return self.index.as_retriever(
            filters=filters,
            similarity_top_k=top_k
        ).retrieve(query)

    def _infer_filters(self, query: str) -> MetadataFilters:
        """Use LLM to extract structured filters from Thai query."""
        # Extract: customer_type, date_range, product_category, is_active
        # Reference: src/14_metadata_filtering.py
```

**Common Filters:**
```python
# Active promotions only
filters = MetadataFilters(filters=[
    MetadataFilter(key="is_active", value=True)
])

# Customer-specific
filters = MetadataFilters(filters=[
    MetadataFilter(key="customer_type", value="เอเย่นต์"),
    MetadataFilter(key="is_active", value=True)
], condition="and")

# Product + date range
filters = MetadataFilters(filters=[
    MetadataFilter(key="product_category", value="water"),
    MetadataFilter(key="start_date_gregorian", value="2025-10-01", operator=">="),
    MetadataFilter(key="end_date_gregorian", value="2025-12-31", operator="<=")
], condition="and")
```

#### **Strategy 2: Hybrid Search** ⭐⭐ (ESSENTIAL)
**Script:** `src-sale-promotion/retrieval/retrievers/hybrid.py`

**Why ESSENTIAL for promotions:**
- Thai product names need exact matching
- "น้ำดื่มช้าง PET 600 CC" must match exactly
- Semantic search alone might miss specific SKUs

**Implementation:**
```python
from llama_index.core.retrievers import QueryFusionRetriever

class PromotionHybridRetriever:
    def __init__(self, vector_store):
        # Combine semantic + keyword search
        self.vector_retriever = vector_store.as_retriever(similarity_top_k=10)
        self.keyword_retriever = self._create_keyword_retriever()

        # Fusion retriever
        self.retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever, self.keyword_retriever],
            mode="reciprocal_rank_fusion",  # RRF works best
            similarity_top_k=10
        )

    def _create_keyword_retriever(self):
        """Create BM25-based keyword retriever for exact matching."""
        # Reference: src/16_hybrid_search.py
        # Use Thai tokenization for product names
```

**Keyword Indexing for Thai:**
```python
# Product name variations to index
product_keywords = {
    'น้ำดื่มช้าง PET 600 CC': [
        'น้ำดื่มช้าง', 'น้ำช้าง', 'Chang water',
        'PET 600', '600 CC', '600CC'
    ],
    'เบียร์ช้างคลาสิก 620 มล.': [
        'เบียร์ช้าง', 'ช้างคลาสิก', 'Chang Classic',
        'beer', '620', '620มล'
    ]
}
```

#### **Strategy 3: Vector Search** ⭐ (BASELINE)
**Script:** `src-sale-promotion/retrieval/retrievers/vector.py`

**Why BASELINE:**
- Simple, fast, reliable fallback
- Good for general semantic queries
- When metadata filters don't apply

**Implementation:**
```python
class PromotionVectorRetriever:
    def retrieve(self, query: str, top_k: int = 10):
        # Simple semantic similarity search
        return self.index.as_retriever(
            similarity_top_k=top_k
        ).retrieve(query)
```

#### **Strategy 4: Query Planning Agent** ⚠️ (OPTIONAL)
**Script:** `src-sale-promotion/retrieval/retrievers/planner.py`

**When to use:**
- Complex analytical queries
- Multi-step reasoning
- Comparative analysis

**Examples:**
- "เปรียบเทียบโปรโมชั่นเบียร์กับน้ำดื่มทั้งหมด" (compare beer vs water promos)
- "แสดงโปรโมชั่นที่ดีที่สุดสำหรับการซื้อ 100 แพ็ค" (best deal for 100 packs)
- "สรุปความแตกต่างระหว่างโปรโมชั่นเอเย่นต์กับซับเอเย่นต์" (agent vs sub-agent comparison)

**Implementation:**
```python
from llama_index.core.query_engine import SubQuestionQueryEngine

class PromotionPlannerRetriever:
    def retrieve(self, query: str):
        # Decompose complex query into sub-questions
        # Reference: src/17_query_planning_agent.py
        # Only activate for complex queries (detected by router)
```

**Strategies We Skip:**
- ❌ Document Summary Retrieval (no document hierarchy)
- ❌ Recursive Retrieval (no nested structure)
- ❌ Chunk Decoupling (no chunking)

---

### Phase 4: Router & Query Interface
**Reference:** `src/METHODOLOGY.md` Phase 4, `attached_assets/04_agentic_retrieval.md`

**Two-Stage Routing (Simplified):**

**Stage 1: Index Classification** (if multi-index)
```python
# Example: Separate indices for different promotion periods
indices = {
    'active_promotions': active_index,
    'archived_promotions': archived_index,
    'seasonal_promotions': seasonal_index
}

# Route based on date context in query
```

**Stage 2: Strategy Selection**
```python
class PromotionRouterRetriever:
    def __init__(self):
        self.strategy_priority = {
            "metadata": 1,    # Try metadata filtering first
            "hybrid": 2,      # Then hybrid for product matching
            "vector": 3,      # Fallback to vector
            "planner": 4      # Only for complex queries
        }

    def select_strategy(self, query: str) -> str:
        """LLM-based strategy selection with fallback heuristics."""

        # Heuristics (fast path)
        if self._has_structured_filters(query):
            return "metadata"  # "โปรโมชั่นเอเย่นต์เดือนนี้"

        if self._has_product_name(query):
            return "hybrid"    # "น้ำดื่มช้าง PET 600"

        if self._is_complex_query(query):
            return "planner"   # "เปรียบเทียบ...", "แสดงความแตกต่าง..."

        # LLM decision (fallback)
        return self._llm_select_strategy(query)

    def _has_structured_filters(self, query: str) -> bool:
        """Detect if query contains filterable entities."""
        customer_types = ['เอเย่นต์', 'ซับเอเย่นต์', 'รถขายปลีก']
        date_keywords = ['เดือนนี้', 'วันนี้', 'สัปดาห์นี้', 'ใกล้หมดอายุ']

        return (any(ct in query for ct in customer_types) or
                any(dk in query for dk in date_keywords))
```

**Strategy Descriptions for LLM:**
```python
strategy_descriptions = {
    "metadata": """
    กรองข้อมูลด้วยเงื่อนไขที่ชัดเจน เช่น ประเภทลูกค้า วันที่ สินค้า
    เหมาะกับ: คำถามที่มีเงื่อนไขชัดเจน เช่น "โปรโมชั่นสำหรับเอเย่นต์" "ดีลเดือนนี้"
    ความเร็ว: เร็วมาก (ค้นหาเฉพาะข้อมูลที่ตรงเงื่อนไข)
    """,

    "hybrid": """
    ค้นหาทั้งความหมายและคำที่ตรงกัน เหมาะสำหรับชื่อสินค้า
    เหมาะกับ: คำถามเกี่ยวกับสินค้าเฉพาะ เช่น "น้ำดื่มช้าง PET 600" "เบียร์ช้างคลาสิก"
    ความเร็ว: ปานกลาง (ค้นหา 2 วิธีแล้วรวมผล)
    """,

    "vector": """
    ค้นหาจากความหมายโดยรวม เหมาะกับคำถามทั่วไป
    เหมาะกับ: คำถามกว้างๆ เช่น "มีโปรโมชั่นอะไรบ้าง" "แสดงดีลทั้งหมด"
    ความเร็ว: เร็ว (ค้นหาอย่างเดียว)
    """,

    "planner": """
    วางแผนคำถามที่ซับซ้อน แยกเป็นขั้นตอน
    เหมาะกับ: คำถามวิเคราะห์ เช่น "เปรียบเทียบ..." "หาที่ดีที่สุด..." "คำนวณ..."
    ความเร็ว: ช้า (ต้องทำหลายขั้นตอน)
    """
}
```

---

## Sales-Specific Adaptations

### 1. Buddhist Calendar Conversion

```python
from datetime import datetime, timedelta

def convert_buddhist_to_gregorian(buddhist_date_str: str) -> datetime:
    """
    Convert Buddhist Era date to Gregorian (CE).
    Example: "2568-10-01" → "2025-10-01"
    """
    buddhist_year = int(buddhist_date_str.split('-')[0])
    gregorian_year = buddhist_year - 543

    return datetime.strptime(
        buddhist_date_str.replace(str(buddhist_year), str(gregorian_year)),
        "%Y-%m-%d %H:%M:%S"
    )

def is_promotion_active(start_date: datetime, end_date: datetime) -> bool:
    """Check if promotion is currently active."""
    now = datetime.now()
    return start_date <= now <= end_date

def is_expiring_soon(end_date: datetime, days: int = 7) -> bool:
    """Check if promotion expires within N days."""
    now = datetime.now()
    return timedelta(0) <= (end_date - now) <= timedelta(days=days)
```

### 2. Product Normalization

```python
# Handle product name variations and SKU extraction
class ProductNormalizer:
    def __init__(self):
        self.brand_aliases = {
            'ช้าง': ['Chang', 'chang', 'ช้าง', 'CHANG'],
            'นิยมไทย': ['Niyom Thai', 'นิยมไทย'],
            'ซาโต้': ['Sato', 'ซาโต้', 'SATO']
        }

        self.category_patterns = {
            'water': r'น้ำดื่ม|น้ำแร่',
            'beer': r'เบียร์|beer|Beer',
            'spirits': r'สุราขาว|เหล้า|whisky',
            'soda': r'โคล่า|โซดา|soda'
        }

    def extract_category(self, product_name: str) -> str:
        """Extract product category from Thai product name."""
        import re
        for category, pattern in self.category_patterns.items():
            if re.search(pattern, product_name):
                return category
        return 'other'

    def extract_size(self, product_name: str) -> str:
        """Extract size/volume from product name."""
        import re
        # Match patterns: "600 CC", "1500 CC", "330 มล.", "620 มล."
        size_match = re.search(r'(\d+)\s*(CC|มล\.|ml|ML)', product_name)
        return size_match.group(0) if size_match else None

    def normalize_product_name(self, product_name: str) -> str:
        """Normalize product name for better matching."""
        # Remove extra spaces, normalize case, handle variations
        return product_name.strip().replace('  ', ' ')
```

### 3. Customer Tier Hierarchy

```python
class CustomerTierManager:
    def __init__(self):
        self.tier_hierarchy = {
            'tier_1': ['เอเย่นต์', 'Super Sub'],  # Highest tier
            'tier_2': ['ซับเอเย่นต์', 'CVA'],
            'tier_3': ['รถขายปลีก', 'Retail TT', 'ซับซื้อตรง']
        }

    def get_tier(self, customer_type: str) -> str:
        """Get tier level for customer type."""
        for tier, types in self.tier_hierarchy.items():
            if customer_type in types:
                return tier
        return 'unknown'

    def is_wholesale(self, customer_type: str) -> bool:
        """Check if customer is wholesale tier."""
        return self.get_tier(customer_type) in ['tier_1', 'tier_2']
```

### 4. Quantity-Based Recommendations

```python
class QuantityRecommender:
    def find_best_deal(self, product: str, desired_quantity: int,
                      promotions: list) -> dict:
        """
        Find the best promotion for a given purchase quantity.

        Args:
            product: Product name (Thai)
            desired_quantity: How many units customer wants to buy
            promotions: List of applicable promotions

        Returns:
            Best promotion with effective discount
        """
        best_deal = None
        best_discount = 0

        for promo in promotions:
            # Check if quantity meets minimum
            if desired_quantity >= promo['buy_quantity']:
                # Calculate effective discount
                times_applicable = desired_quantity // promo['buy_quantity']
                total_free = times_applicable * promo['free_quantity']

                discount_pct = (total_free / desired_quantity) * 100

                if discount_pct > best_discount:
                    best_discount = discount_pct
                    best_deal = {
                        'promotion': promo,
                        'free_items': total_free,
                        'discount_percentage': discount_pct,
                        'times_applicable': times_applicable
                    }

        return best_deal
```

### 5. Thai Language Query Processing

```python
class ThaiQueryProcessor:
    def __init__(self):
        # Common Thai query patterns
        self.date_keywords = {
            'วันนี้': 'today',
            'เดือนนี้': 'this_month',
            'สัปดาห์นี้': 'this_week',
            'ใกล้หมดอายุ': 'expiring_soon',
            'ที่ใช้ได้': 'active'
        }

        self.quantity_keywords = {
            'ดีลที่ดีที่สุด': 'best_deal',
            'ส่วนลดสูงสุด': 'highest_discount',
            'แถมมากที่สุด': 'most_free_items'
        }

    def detect_intent(self, query: str) -> dict:
        """Detect query intent and extract entities."""
        intent = {
            'type': 'general',  # general, filter, comparison, calculation
            'date_filter': None,
            'customer_filter': None,
            'product_filter': None,
            'quantity_query': False
        }

        # Detect date filters
        for thai, eng in self.date_keywords.items():
            if thai in query:
                intent['date_filter'] = eng
                intent['type'] = 'filter'

        # Detect quantity-based queries
        if any(kw in query for kw in self.quantity_keywords.keys()):
            intent['quantity_query'] = True
            intent['type'] = 'calculation'

        # Detect comparison queries
        if 'เปรียบเทียบ' in query or 'แตกต่าง' in query:
            intent['type'] = 'comparison'

        return intent
```

---

## Implementation Roadmap

### **Phase 1: Foundation** (Week 1-2)

**Priority 1: Data Processing**
```bash
# Implement data preparation
src-sale-promotion/data_processing/
├── csv_processor.py          # CSV → Documents
├── field_mapper.py            # Field categorization
├── metadata_extractor.py      # Derive metadata
└── promotion_normalizer.py    # Thai text handling
```

**Deliverables:**
- ✅ Markdown files with rich metadata
- ✅ Buddhist → Gregorian date conversion
- ✅ Product categorization
- ✅ Customer tier assignment

**Priority 2: Embedding Generation**
```bash
# Implement simplified embedding pipeline
src-sale-promotion/docs_embedding/
├── batch_embedding.py         # Batch processing
└── vector_store_setup.py      # ChromaDB/pgvector
```

**Deliverables:**
- ✅ Single-level embeddings (no chunking)
- ✅ Metadata preserved in vector store
- ✅ Batch processing (50 promotions/batch)

---

### **Phase 2: Core Retrieval** (Week 3-4)

**Priority 1: Metadata Filtering**
```bash
src-sale-promotion/retrieval/retrievers/
└── metadata.py                # PRIMARY strategy
```

**Features:**
- Auto-detect filters from Thai queries
- Date range filtering (active promotions)
- Customer type filtering
- Product category filtering

**Priority 2: Hybrid Search**
```bash
src-sale-promotion/retrieval/retrievers/
└── hybrid.py                  # ESSENTIAL for Thai products
```

**Features:**
- Semantic + keyword fusion
- Thai tokenization for product names
- Reciprocal Rank Fusion (RRF)

**Priority 3: Vector Search**
```bash
src-sale-promotion/retrieval/retrievers/
└── vector.py                  # BASELINE fallback
```

**Features:**
- Simple semantic similarity
- Fast and reliable

**Test Each Strategy Independently:**
```python
# Test metadata filtering
python -m src_sale_promotion.retrieval.retrievers.metadata \
    --query "โปรโมชั่นสำหรับเอเย่นต์เดือนนี้"

# Test hybrid search
python -m src_sale_promotion.retrieval.retrievers.hybrid \
    --query "น้ำดื่มช้าง PET 600 CC"

# Test vector search
python -m src_sale_promotion.retrieval.retrievers.vector \
    --query "มีโปรโมชั่นอะไรบ้าง"
```

---

### **Phase 3: Router & CLI** (Week 5)

**Priority 1: Router Implementation**
```bash
src-sale-promotion/retrieval/
└── router.py                  # Strategy selection
```

**Features:**
- Heuristic-based fast path
- LLM fallback for complex queries
- Strategy priority: metadata → hybrid → vector

**Priority 2: CLI Interface**
```bash
src-sale-promotion/retrieval/
└── cli.py                     # Interactive query tool
```

**Features:**
```bash
# Interactive mode
python -m src_sale_promotion.retrieval.cli --interactive

# Single query
python -m src_sale_promotion.retrieval.cli \
    --query "โปรโมชั่นน้ำดื่มช้างสำหรับเอเย่นต์"

# Batch testing
python -m src_sale_promotion.retrieval.cli \
    --test-queries queries.txt
```

---

### **Phase 4: Advanced Features** (Week 6+, Optional)

**Optional: Query Planning Agent**
```bash
src-sale-promotion/retrieval/retrievers/
└── planner.py                 # For complex queries
```

**When to implement:**
- Users need analytical comparisons
- Multi-step reasoning required
- After core strategies validated

**Optional: Streamlit UI**
```bash
src-sale-promotion/retrieval/
└── streamlit_ui.py            # WhatsApp-style chat UI
```

**Features:**
- Thai language chat interface
- Real-time promotion lookup
- Quantity calculator
- Comparison tool

---

## Expected Query Patterns

### **Type 1: Simple Lookups** (80% of queries)
**Strategy:** Metadata Filtering

```
Query: "โปรโมชั่นสำหรับเอเย่นต์"
Filter: customer_type == "เอเย่นต์", is_active == True
Response: [List of active agent promotions]

Query: "ดีลเดือนตุลาคม"
Filter: start_date >= "2025-10-01", end_date <= "2025-10-31"
Response: [October promotions]

Query: "โปรโมชั่นที่ใกล้หมดอายุ"
Filter: expires_soon == True
Response: [Expiring promotions]
```

### **Type 2: Product-Specific** (15% of queries)
**Strategy:** Hybrid Search

```
Query: "น้ำดื่มช้าง PET 600 CC มีโปรโมชั่นอะไร"
Method: Keyword match "น้ำดื่มช้าง PET 600" + semantic search
Response: [All Chang water 600CC promotions]

Query: "เบียร์ช้างคลาสิก 620 มล."
Method: Exact product name match + variations
Response: [Chang Classic 620ml promotions]
```

### **Type 3: Analytical** (5% of queries)
**Strategy:** Query Planning Agent (if implemented)

```
Query: "เปรียบเทียบโปรโมชั่นน้ำดื่มกับเบียร์"
Steps:
  1. Retrieve all water promotions
  2. Retrieve all beer promotions
  3. Compare discount percentages
  4. Synthesize comparison
Response: [Structured comparison with insights]

Query: "หาโปรโมชั่นที่ดีที่สุดสำหรับซื้อน้ำดื่มช้าง 100 แพ็ค"
Steps:
  1. Filter Chang water promotions
  2. Calculate effective discount for 100 packs
  3. Rank by discount percentage
  4. Return best deal
Response: [Best promotion with calculations]
```

---

## Configuration Guide

### **Environment Variables**

```bash
# .env file
OPENAI_API_KEY=sk-...

# Embedding settings (same as src/)
EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=1024  # Not used (no chunking), but keep for compatibility
CHUNK_OVERLAP=50

# Vector store
VECTOR_STORE_TYPE=chromadb  # or pgvector
CHROMADB_PATH=./chromadb_promotions
PGVECTOR_CONNECTION=postgresql://...

# Retrieval settings
DEFAULT_TOP_K=10
METADATA_FILTER_MODE=auto  # auto-detect from query
HYBRID_FUSION_MODE=rrf  # reciprocal rank fusion

# Router settings
ROUTER_STRATEGY_SELECTOR=heuristic  # heuristic or llm
ROUTER_FALLBACK=vector

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/promotion_retrieval.log
```

### **Field Configuration**

```yaml
# config/promotion_fields.yaml
name: sale_promotions
description: Thai sales promotion data

field_mappings:
  # Temporal fields
  - csv_column: "ระยะเวลาเริ่ม"
    metadata_key: "start_date"
    field_type: "temporal"
    data_type: "datetime"
    required: true
    transform: "buddhist_to_gregorian"

  - csv_column: "ระยะเวลาสื้นสุด"
    metadata_key: "end_date"
    field_type: "temporal"
    data_type: "datetime"
    required: true
    transform: "buddhist_to_gregorian"

  # Customer segment
  - csv_column: "ประเภทลูกค้า"
    metadata_key: "customer_type"
    field_type: "customer_segment"
    data_type: "string"
    required: true
    values: ["เอเย่นต์", "ซับเอเย่นต์", "รถขายปลีก", "Super Sub", "CVA", "Retail TT", "ซับซื้อตรง"]

  # Product fields
  - csv_column: "สินค้า"
    metadata_key: "product_name"
    field_type: "product"
    data_type: "string"
    required: true
    transform: "normalize_product_name"

  # Promotion terms
  - csv_column: "ซื้อ"
    metadata_key: "buy_quantity"
    field_type: "promotion_terms"
    data_type: "numeric"
    required: true

  - csv_column: "หน่วยซื้อ"
    metadata_key: "buy_unit"
    field_type: "promotion_terms"
    data_type: "string"
    required: true
    values: ["แพ็ค", "ลัง", "ขวด", "ถาด", "กระป๋อง", "ลัง/ถาด"]

derived_metadata:
  - name: "is_active"
    type: "boolean"
    compute: "is_within_date_range(start_date, end_date)"

  - name: "product_category"
    type: "string"
    compute: "extract_category(product_name)"
    values: ["water", "beer", "spirits", "soda", "other"]

  - name: "customer_tier"
    type: "string"
    compute: "get_customer_tier(customer_type)"
    values: ["tier_1", "tier_2", "tier_3"]

  - name: "discount_percentage"
    type: "numeric"
    compute: "calculate_discount(buy_quantity, free_quantity)"
```

---

## Testing Strategy

### **Unit Tests**

```python
# tests/test_metadata_filtering.py
def test_customer_type_filter():
    """Test filtering by customer type."""
    query = "โปรโมชั่นสำหรับเอเย่นต์"
    filters = metadata_retriever.infer_filters(query)
    assert filters.get("customer_type") == "เอเย่นต์"

def test_date_filter():
    """Test active promotion filtering."""
    query = "ดีลที่ใช้ได้ในเดือนนี้"
    filters = metadata_retriever.infer_filters(query)
    assert filters.get("is_active") == True

# tests/test_product_normalization.py
def test_product_category_extraction():
    """Test product category extraction."""
    product = "น้ำดื่มช้าง PET 600 CC"
    category = normalizer.extract_category(product)
    assert category == "water"

def test_product_size_extraction():
    """Test size extraction from Thai product name."""
    product = "น้ำดื่มช้าง PET 1500 CC"
    size = normalizer.extract_size(product)
    assert size == "1500 CC"

# tests/test_date_conversion.py
def test_buddhist_to_gregorian():
    """Test Buddhist calendar conversion."""
    buddhist = "2568-10-01 00:00:00"
    gregorian = convert_buddhist_to_gregorian(buddhist)
    assert gregorian.year == 2025
```

### **Integration Tests**

```python
# tests/test_end_to_end.py
def test_full_retrieval_pipeline():
    """Test complete retrieval from query to results."""

    # Test metadata filtering
    query = "โปรโมชั่นน้ำดื่มช้างสำหรับเอเย่นต์"
    results = router.retrieve(query)

    assert len(results) > 0
    assert all(r.metadata['customer_type'] == 'เอเย่นต์' for r in results)
    assert all('น้ำดื่มช้าง' in r.text for r in results)

def test_hybrid_search():
    """Test hybrid search for exact product matching."""
    query = "น้ำดื่มช้าง PET 600 CC"
    results = hybrid_retriever.retrieve(query)

    # Should find exact matches
    assert any('PET 600 CC' in r.text for r in results)

def test_quantity_recommendation():
    """Test quantity-based recommendation."""
    product = "น้ำดื่มช้าง PET 600 CC"
    quantity = 100

    best_deal = recommender.find_best_deal(product, quantity)
    assert best_deal is not None
    assert best_deal['discount_percentage'] > 0
```

### **Evaluation Metrics**

```python
# tests/eval_promotion_rag.py
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision

# Test dataset (Thai queries with ground truth)
test_queries = [
    {
        "query": "โปรโมชั่นสำหรับเอเย่นต์เดือนตุลาคม",
        "ground_truth": "ซื้อน้ำดื่มช้าง 108 แพ็ค แถม 42 แพ็ค",
        "expected_filter": {"customer_type": "เอเย่นต์", "product_category": "water"}
    },
    # ... more test cases
]

# Evaluation
results = evaluate(
    dataset=test_dataset,
    metrics=[answer_relevancy, faithfulness, context_precision]
)

# Success criteria
assert results['answer_relevancy'] >= 0.80
assert results['faithfulness'] >= 0.85
assert results['context_precision'] >= 0.80
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Metadata Filter Latency** | < 50ms | Fast pre-filtering |
| **Hybrid Search Latency** | < 200ms | Semantic + keyword |
| **Vector Search Latency** | < 100ms | Simple similarity |
| **Query Planning Latency** | < 2s | Complex multi-step |
| **Answer Relevancy** | ≥ 0.80 | Ragas metric |
| **Faithfulness** | ≥ 0.85 | No hallucination |
| **Context Precision** | ≥ 0.80 | Correct metadata |

---

## Best Practices

### **DO:**
✅ Preserve all metadata from CSV (critical for filtering)
✅ Convert Buddhist dates to Gregorian for consistency
✅ Normalize Thai product names for better matching
✅ Use metadata filtering as primary strategy
✅ Test with real Thai queries from users
✅ Log all retrieval calls for analysis
✅ Start simple (3 strategies), add complexity as needed

### **DON'T:**
❌ Chunk promotions (they're atomic)
❌ Generate document summaries (unnecessary)
❌ Use all 7 strategies (overkill for this use case)
❌ Ignore date filtering (critical for active promotions)
❌ Skip Thai language testing (product names must match)
❌ Over-engineer initially (start with metadata + hybrid)

---

## References

### **Parent Methodology**
- `src/METHODOLOGY.md` - Complete 7-strategy approach
- `attached_assets/00_doc_prep_workflow.md` - Data preparation best practices
- `attached_assets/01_building_rag_pipeline.md` - Ingestion pipeline design
- `attached_assets/03_retrieval_pipeline_plan.md` - Implementation phases
- `attached_assets/04_agentic_retrieval.md` - Agentic retrieval PRD

### **Proven Applications**
- `src-iLand/` - Thai language implementation reference
- `src-iLand/retrieval/fast_metadata_index.py` - Sub-50ms metadata filtering
- `src/agentic_retriever/router.py` - Two-stage routing implementation

### **LlamaIndex Documentation**
- [Metadata Filtering](https://docs.llamaindex.ai/en/stable/examples/vector_stores/metadata_filter/)
- [Hybrid Search](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/)
- [Router Query Engine](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/)

---

**Last Updated:** 2025-10-26
**Version:** 1.0
**Status:** Implementation Guide - Ready for Development
**Next Steps:** Begin Phase 1 (Data Processing & Embedding Generation)
