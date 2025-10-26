# Product Requirements Document (PRD)
# Phase 1: Sales Promotion Data Preparation Workflow - v2.0

**Project:** Sales Promotion RAG Pipeline
**Phase:** 1 - Data Preparation
**Version:** 2.0 (Updated with Option 1c+2b)
**Date:** 2025-10-26
**Owner:** Data Engineering Team

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-10-26 | Initial PRD | Data Engineering Team |
| 2.0 | 2025-10-26 | Updated to Option 1c+2b (SimpleDocument + dual output) | Data Engineering Team |

**Key Changes in v2.0:**
- ✅ Use `SimpleDocument` class (not LlamaIndex Document) in Phase 1
- ✅ Dual output: JSONL (primary) + Markdown files (human review)
- ✅ Remove PKL file output (not needed)
- ✅ Add Phase 1→2 handoff specification
- ✅ Clarify Phase 2 converts SimpleDocument → LlamaIndex Documents

---

## 1. Executive Summary

Implement the data preparation workflow for the Sales Promotion RAG pipeline, transforming raw Thai promotional CSV data into **SimpleDocument objects** with natural language text and rich metadata, ready for embedding generation (Phase 2).

### Key Objectives
- ✅ Convert Thai CSV promotion data → **SimpleDocument objects** (lightweight, no LlamaIndex dependency in Phase 1)
- ✅ Extract and derive comprehensive metadata (15+ fields per document)
- ✅ Handle Buddhist calendar dates (2568 → 2025 CE)
- ✅ Normalize Thai product names and customer types
- ✅ Generate natural language text descriptions (NO chunking - atomic promotions)
- ✅ Enable fast metadata filtering for retrieval
- ✅ **Dual output format**: JSONL (Phase 2 consumption) + Markdown (human review)

### Critical Alignment with Proven Patterns

**Following:** `src-imap/data_processing/` (Proven Implementation)

This PRD implements the **proper document preparation workflow** using **Option 1c+2b**:

1. **CSV Row → SimpleDocument Object** (Phase 1)
   ```python
   # NO LlamaIndex imports in Phase 1!
   class SimpleDocument:
       def __init__(self, text: str, metadata: dict):
           self.text = text
           self.metadata = metadata
           self.id = metadata.get('document_id')
   ```

2. **Natural Language Text Generation**
   - Create human-readable description from structured data
   - 50-200 tokens per promotion
   - Embeddable content for semantic search

3. **NO Chunking** (Correctly identified - promotions are atomic)
   - Each promotion = 1 SimpleDocument
   - No need for TextSplitter

4. **Dual Output Formats**
   - **JSONL** (Primary): Machine-readable for Phase 2
   - **Markdown** (Secondary): Human-readable for QA/debugging
   - Phase 2 reads JSONL only (not markdown)

### Why This Matters

**Phase 2 (Embedding) workflow:**
```python
from llama_index.core import Document, VectorStoreIndex
import json

# Load Phase 1 output (JSONL format)
documents = []
with open("processed/documents/documents.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        # Phase 2 converts SimpleDocument → LlamaIndex Document
        doc = Document(
            text=data['text'],
            metadata=data['metadata'],
            doc_id=data['id']
        )
        documents.append(doc)

# Directly embed
index = VectorStoreIndex.from_documents(documents)
```

**Benefits of Option 1c+2b:**
- ✅ **Phase 1**: Lightweight (no LlamaIndex dependency)
- ✅ **JSONL**: Fast Phase 2 loading (no markdown parsing)
- ✅ **Markdown**: Human QA for Thai text validation
- ✅ **Loose coupling**: Phase 2 can change without touching Phase 1

---

## 2. Background & Context

### Current State
- **Input Data:** `src-sale-promotion/data/promotion_list.csv`
  - 71 Thai promotions (Buy X Get Y)
  - Buddhist calendar dates (B.E. 2568 = 2025 CE)
  - Thai language fields (customer types, products, units)
  - 10 columns of structured promotion data

### Problem Statement
Raw CSV data cannot be directly used for RAG retrieval because:
1. Buddhist calendar dates need conversion to Gregorian
2. Thai text needs normalization for searchability
3. Missing derived metadata (product categories, discount %, customer tiers)
4. No structured document format for embedding
5. No configuration for field mapping reproducibility

### Reference Methodology
- **Parent Methodology:** `src/METHODOLOGY.md` - Phase 1 (Data Preparation)
- **Application Guide:** `src-sale-promotion/METHODOLOGY.md` - Phase 1
- **Theoretical Docs:** `attached_assets/00_doc_prep_workflow.md`
- **Proven Implementation:** `src-imap/data_processing/` - **PRIMARY REFERENCE**

### Key Differences from General RAG
| Aspect | General RAG (src/) | Sales Promotions |
|--------|-------------------|------------------|
| Document Size | ~500-1000 tokens | 50-150 tokens |
| Chunking | Required | ❌ **NOT needed** |
| Summaries | LLM-generated | ❌ **NOT needed** |
| Structure | Hierarchical | Atomic (flat) |
| Primary Focus | Document content | **Metadata** |

---

## 3. Goals & Success Metrics

### Primary Goals
1. **Transform CSV → SimpleDocuments** with 100% data preservation
2. **Extract metadata** covering 15+ derived fields
3. **Handle Thai text** with proper UTF-8 encoding
4. **Buddhist calendar** conversion with validation
5. **Generate dual outputs** (JSONL + Markdown)

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Data Preservation** | 100% | All CSV fields preserved in output |
| **Metadata Completeness** | 15+ fields | Derived metadata generated per document |
| **Date Conversion Accuracy** | 100% | Buddhist → Gregorian validation |
| **Processing Speed** | > 50 docs/sec | 71 promotions in < 2 seconds |
| **Thai Encoding** | No errors | UTF-8 preserved in all outputs |
| **Reproducibility** | Config-driven | YAML config generates same output |
| **Output Formats** | 2 formats | JSONL + 71 markdown files |

### Non-Goals (Out of Scope for Phase 1)
- ❌ Embedding generation (Phase 2)
- ❌ Retrieval strategies (Phase 3)
- ❌ Query routing (Phase 4)
- ❌ UI/API development

---

## 4. Functional Requirements

### 4.1 CSV Processing

**FR-1.1: CSV Loading**
- Load `promotion_list.csv` with UTF-8 encoding
- Handle Thai characters in headers and data
- Support chunk processing for large files (1000+ rows)
- Validate CSV structure (10 expected columns)

**FR-1.2: Field Mapping**
- Map Thai CSV columns to internal metadata keys
- Support configuration via YAML file
- Validate required fields presence
- Handle optional fields gracefully

**Expected Field Mapping:**
```yaml
ประเภทลูกค้า → customer_type
ระยะเวลาเริ่ม → start_date (+ conversion)
ระยะเวลาสื้นสุด → end_date (+ conversion)
สินค้า → product_name
ซื้อ → buy_quantity
หน่วยซื้อ → buy_unit
รายการแถม → free_item_name
แถม → free_quantity
หน่วยแถม → free_unit
เงื่อนไขเพิ่มเติม → additional_conditions
```

### 4.2 Metadata Extraction

**FR-2.1: Direct Field Extraction**
Extract all CSV fields as base metadata:
- Customer type, dates, products, quantities, units, conditions

**FR-2.2: Derived Metadata - Date Handling**
- `is_active`: Check if current date within promotion period
- `start_date_gregorian`: Buddhist → Gregorian conversion
- `end_date_gregorian`: Buddhist → Gregorian conversion
- `duration_days`: Calculate promotion duration
- `expires_soon`: Flag promotions ending within 7 days

**FR-2.3: Derived Metadata - Product Categorization**
- `product_category`: Extract category (water, beer, spirits, soda)
- `product_size`: Extract size/volume (600 CC, 1500 CC, 330 มล.)
- `product_brand`: Extract brand (ช้าง, นิยมไทย, ซาโต้)

**Product Categorization Patterns (YAML Config):**
```yaml
product_patterns:
  water: ["น้ำดื่ม", "PET", "น้ำแร่"]
  beer: ["เบียร์", "beer", "ช้างคลาสสิก"]
  spirits: ["สุราขาว", "spirits", "แสงโสม"]
  soda: ["โคล่า", "cola", "น้ำอัดลม", "ซาโต้"]
```

**FR-2.4: Derived Metadata - Promotion Analysis**
- `promotion_type`: "buy_x_get_y" (all promotions)
- `discount_percentage`: Calculate effective discount
- `effective_price_ratio`: Calculate price after discount

**FR-2.5: Derived Metadata - Customer Tier**
- `customer_tier`: Map to tier_1, tier_2, tier_3
- `is_wholesale`: Boolean for wholesale customers
- `is_retail`: Boolean for retail customers

**Customer Tier Mapping:**
```
tier_1: เอเย่นต์, Super Sub
tier_2: ซับเอเย่นต์, CVA, ซับซื้อตรง
tier_3: รถขายปลีก, Retail TT
```

**FR-2.6: Derived Metadata - Searchability**
- `search_text`: Combined Thai + transliterated text
- `keywords`: Extracted searchable keywords

### 4.3 Thai Text Normalization

**FR-3.1: Product Name Normalization**
- Remove extra whitespace
- Handle list format: `"[product1, product2]"` → extract products
- Preserve Thai characters with UTF-8
- Extract SKU/size information

**FR-3.2: Customer Type Normalization**
- Standardize customer type names
- Map to internal tier system
- Preserve original Thai names

**FR-3.3: Unit Normalization**
- Standardize unit names (แพ็ค, ลัง, ขวด, ถาด, กระป๋อง)
- Handle compound units (ลัง/ถาด)

### 4.4 Buddhist Calendar Conversion

**FR-4.1: Date Conversion**
- Convert B.E. (Buddhist Era) → CE (Common Era)
- Formula: Gregorian Year = Buddhist Year - 543
- Example: 2568-10-01 → 2025-10-01

**FR-4.2: Date Validation**
- Validate date formats
- Check date ranges are logical (start < end)
- Handle timezone (assume Thailand UTC+7)

**FR-4.3: Active Promotion Detection**
- Compare current date against promotion period
- Mark active vs expired promotions
- Calculate days until expiration

### 4.5 SimpleDocument Generation

**FR-5.1: Natural Language Text Generation**
Each promotion generates a natural language text description that will be embedded:

```python
text = f"""
Promotion for {customer_type} customers: Buy {buy_quantity} {buy_unit} of {product_name} and get {free_quantity} {free_unit} of {free_item_name} FREE.

Promotion Period: Valid from {start_date_gregorian} to {end_date_gregorian} ({duration_days} days).
Status: {'Active ✓' if is_active else 'Expired ✗'}
{'⚠️ Expires soon (within 7 days)' if expires_soon else ''}

Promotion Terms:
- Purchase minimum: {buy_quantity} {buy_unit}
- Free items: {free_quantity} {free_unit}
- Effective discount: {discount_percentage}% off total quantity
- Customer tier: {customer_tier} ({'Wholesale' if is_wholesale else 'Retail'})

Product Information:
- Product: {product_name}
- Category: {product_category}
- Brand: {product_brand}
- Size: {product_size or 'N/A'}

{f'Additional Conditions: {additional_conditions}' if additional_conditions else 'No additional conditions.'}
""".strip()
```

**Example Output:**
```
Promotion for เอเย่นต์ customers: Buy 108 แพ็ค of น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด) and get 42 แพ็ค of น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด) FREE.

Promotion Period: Valid from 2025-10-01 to 2025-12-31 (92 days).
Status: Active ✓

Promotion Terms:
- Purchase minimum: 108 แพ็ค
- Free items: 42 แพ็ค
- Effective discount: 28.0% off total quantity
- Customer tier: tier_1 (Wholesale)

Product Information:
- Product: น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)
- Category: water
- Brand: ช้าง
- Size: 600 CC

No additional conditions.
```

**FR-5.2: SimpleDocument Object Creation**

Create lightweight `SimpleDocument` objects (NO LlamaIndex dependency in Phase 1):

```python
# NO import from llama_index!

class SimpleDocument:
    """
    Lightweight document class for Phase 1 data preparation.

    Phase 2 will convert this to LlamaIndex Document format.
    """
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.id = metadata.get('document_id')

    def to_dict(self) -> dict:
        """Serialize to dict for JSONL output"""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata
        }

# Example usage:
doc = SimpleDocument(
    text=generated_text,  # Natural language description from FR-5.1
    metadata={
        # Base fields
        "document_id": "PROMO_001",
        "customer_type": "เอเย่นต์",
        "start_date": "2568-10-01 00:00:00",
        "end_date": "2568-12-31 00:00:00",
        "product_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
        "buy_quantity": 108,
        "buy_unit": "แพ็ค",
        "free_item_name": "น้ำดื่มช้าง PET 600 CC (1 แพ็ค เท่ากับ 12 ขวด)",
        "free_quantity": 42,
        "free_unit": "แพ็ค",
        "additional_conditions": None,

        # Derived metadata (15+ fields)
        "customer_tier": "tier_1",
        "is_wholesale": True,
        "is_retail": False,
        "start_date_gregorian": "2025-10-01",
        "end_date_gregorian": "2025-12-31",
        "is_active": True,
        "expires_soon": False,
        "duration_days": 92,
        "product_category": "water",
        "product_brand": "ช้าง",
        "product_size": "600 CC",
        "discount_percentage": 28.0,
        "promotion_type": "buy_x_get_y",
        "keywords": ["น้ำดื่มช้าง", "PET", "600", "เอเย่นต์", "water", "chang"]
    }
)
```

**Why SimpleDocument (not LlamaIndex Document):**
- ✅ **Lightweight**: No external dependencies in Phase 1
- ✅ **Clean separation**: Phase 1 = data prep, Phase 2 = embedding
- ✅ **Flexible**: Can change Phase 2 embedding strategy without touching Phase 1
- ✅ **Proven pattern**: Following `src-imap/` implementation

**FR-5.3: Markdown File Generation (Human Review)**

Generate individual markdown files for human QA and debugging:
- **Purpose**: Thai text validation, metadata verification
- **Format**: `promotion_001.md` through `promotion_071.md`
- **NOT consumed by Phase 2** (only JSONL is consumed)

### 4.6 Configuration Management

**FR-6.1: YAML Configuration**
- Field mapping configuration
- Product category patterns
- Brand patterns
- Customer tier mapping
- Processing options

**FR-6.2: Config Validation**
- Validate YAML structure
- Check required fields present
- Validate pattern syntax

**FR-6.3: Config Reporting**
- Generate config report showing mappings used
- Document derived metadata computed
- Track fields processed vs skipped

### 4.7 Output Management

**FR-7.1: File Organization**
```
src-sale-promotion/data/processed/
├── documents/
│   ├── documents.jsonl              # ✅ PRIMARY OUTPUT (SimpleDocument as JSON)
│   └── markdown_files/              # ✅ SECONDARY OUTPUT (Human review)
│       ├── promotion_001.md
│       ├── promotion_002.md
│       ├── ...
│       └── promotion_071.md
├── config_used.yaml                 # Configuration snapshot
├── processing_report.json           # Statistics and validation
└── README.md                        # Processing summary
```

**Primary Output Format (documents.jsonl):**
- SimpleDocument objects serialized as JSON (one per line)
- Each line contains: `{id, text, metadata}`
- **This is what Phase 2 (Embedding) will consume**

**Example JSONL line:**
```json
{
  "id": "PROMO_001",
  "text": "Promotion for เอเย่นต์ customers: Buy 108 แพ็ค of น้ำดื่มช้าง...",
  "metadata": {
    "document_id": "PROMO_001",
    "customer_type": "เอเย่นต์",
    "customer_tier": "tier_1",
    "product_category": "water",
    "start_date_gregorian": "2025-10-01",
    "end_date_gregorian": "2025-12-31",
    "is_active": true,
    "discount_percentage": 28.0
  }
}
```

**Secondary Output Format (markdown_files/):**
- Individual markdown files: `promotion_001.md` to `promotion_071.md`
- Human-readable format for QA
- NOT consumed by Phase 2

**FR-7.2: Statistics Tracking**
- Total documents processed (target: 71)
- Total SimpleDocument objects created
- Fields extracted per document (base: 10, derived: 15+)
- Date conversions performed
- Text generation statistics (avg length, min/max)
- Validation errors (if any)
- Processing time
- Memory usage

**FR-7.3: Document Validation**
- Verify all SimpleDocuments have `text` field (non-empty)
- Verify all SimpleDocuments have `metadata` dict with 15+ fields
- Verify all SimpleDocuments have unique `id`
- Validate Thai UTF-8 encoding in text
- Validate metadata types (dates as strings, numbers as floats, etc.)

### 4.8 Phase 1→2 Handoff Specification

**CRITICAL: Data Contract Between Phases**

**Phase 1 Output:**
- Format: JSONL (newline-delimited JSON)
- File: `src-sale-promotion/data/processed/documents/documents.jsonl`
- Content: SimpleDocument objects serialized as JSON

**JSONL Format Specification:**
```json
{
  "id": "PROMO_001",
  "text": "Natural language promotion description (50-200 tokens)",
  "metadata": {
    "document_id": "PROMO_001",
    "customer_type": "เอเย่นต์",
    "customer_tier": "tier_1",
    "product_name": "น้ำดื่มช้าง PET 600 CC",
    "product_category": "water",
    "product_brand": "ช้าง",
    "product_size": "600 CC",
    "buy_quantity": 108,
    "buy_unit": "แพ็ค",
    "free_item_name": "น้ำดื่มช้าง PET 600 CC",
    "free_quantity": 42,
    "free_unit": "แพ็ค",
    "start_date": "2568-10-01 00:00:00",
    "end_date": "2568-12-31 00:00:00",
    "start_date_gregorian": "2025-10-01",
    "end_date_gregorian": "2025-12-31",
    "is_active": true,
    "expires_soon": false,
    "duration_days": 92,
    "discount_percentage": 28.0,
    "promotion_type": "buy_x_get_y",
    "is_wholesale": true,
    "is_retail": false,
    "keywords": ["น้ำดื่มช้าง", "PET", "600", "เอเย่นต์", "water"],
    "additional_conditions": null
  }
}
```

**Phase 2 (Embedding) Consumption:**

```python
from llama_index.core import Document, VectorStoreIndex
import json

def load_phase1_documents(jsonl_path: str) -> List[Document]:
    """
    Load SimpleDocuments from Phase 1 JSONL and convert to LlamaIndex Documents.

    Args:
        jsonl_path: Path to documents.jsonl from Phase 1

    Returns:
        List of LlamaIndex Document objects ready for embedding
    """
    documents = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse SimpleDocument JSON
                data = json.loads(line)

                # Convert to LlamaIndex Document
                doc = Document(
                    text=data['text'],
                    metadata=data['metadata'],
                    doc_id=data['id']
                )

                documents.append(doc)

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except KeyError as e:
                print(f"Missing field in line {line_num}: {e}")

    return documents

# Usage in Phase 2:
documents = load_phase1_documents('data/processed/documents/documents.jsonl')
print(f"Loaded {len(documents)} documents from Phase 1")

# Create embeddings
index = VectorStoreIndex.from_documents(documents)
```

**Markdown Files (NOT for Phase 2):**
- Location: `data/processed/documents/markdown_files/promotion_NNN.md`
- Purpose: Human QA, Thai text validation, debugging
- **NOT consumed by Phase 2** (only JSONL is consumed)
- Format: Human-readable markdown with sections

**Data Contract Guarantees:**
1. ✅ Each JSONL line is valid JSON
2. ✅ Each object has `id`, `text`, `metadata` fields
3. ✅ `text` is non-empty string (50-200 tokens)
4. ✅ `metadata` has 15+ fields
5. ✅ Thai UTF-8 encoding preserved
6. ✅ All 71 promotions included

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| Processing Speed | > 50 docs/sec | 71 promotions < 2 seconds |
| Memory Usage | < 100 MB | Small dataset, should be minimal |
| CPU Usage | < 50% single core | No heavy computation |
| Startup Time | < 1 second | Fast iteration during development |

### 5.2 Reliability

- **Data Preservation**: 100% of CSV data must appear in output
- **Encoding**: Zero Thai character corruption
- **Date Conversion**: 100% accuracy (validated against ground truth)
- **Error Handling**: Graceful failure with clear error messages

### 5.3 Maintainability

- **Code Quality**: Type hints, docstrings, clear naming
- **Configuration**: All mappings in YAML (no hardcoded values)
- **Logging**: INFO level for progress, DEBUG for details
- **Testing**: Unit tests for all transformation functions

### 5.4 Usability

**Command-Line Interface:**
```bash
# Basic usage
python -m src_sale_promotion.data_processing.run_data_prep

# With custom config
python -m src_sale_promotion.data_processing.run_data_prep \
  --config config/custom_fields.yaml

# Dry run (validate only)
python -m src_sale_promotion.data_processing.run_data_prep --dry-run

# Verbose output
python -m src_sale_promotion.data_processing.run_data_prep --verbose
```

---

## 6. Technical Architecture

### 6.1 Component Structure

```
src-sale-promotion/data_processing/
├── __init__.py
├── config/
│   └── promotion_fields.yaml       # Field mapping config
├── processors/
│   ├── __init__.py
│   ├── csv_processor.py            # CSV loading & parsing
│   ├── metadata_extractor.py       # Metadata derivation
│   ├── thai_normalizer.py          # Thai text handling
│   └── text_generator.py           # Natural language text generation
├── utils/
│   ├── __init__.py
│   ├── date_utils.py               # Buddhist calendar conversion
│   ├── product_utils.py            # Product categorization
│   └── customer_utils.py           # Customer tier mapping
├── models.py                        # Data classes (PromotionDocument, SimpleDocument)
└── run_data_prep.py                # Main orchestrator
```

**Implementation Strategy:**
- Copy and modify modules from `src-imap/data_processing/` as templates
- Adapt for promotion-specific requirements
- Not direct reuse - will customize for sales data

### 6.2 Data Flow

```
CSV File (promotion_list.csv)
    ↓
[CSV Processor]
  - Load with UTF-8
  - Validate structure
  - Parse into rows
    ↓
[Field Mapper]
  - Map Thai columns → metadata keys
  - Extract base fields (10 fields)
    ↓
[Metadata Extractor]
  - Convert dates (Buddhist → Gregorian)
  - Categorize products (water, beer, spirits)
  - Calculate discounts (effective %)
  - Assign customer tiers (tier_1/2/3)
  - Derive 15+ metadata fields
    ↓
[Thai Normalizer]
  - Normalize product names
  - Extract keywords
  - Generate search text
    ↓
[Text Generator]
  - Create natural language descriptions
  - Format promotion details (50-200 tokens)
    ↓
[Document Creator]
  - Create PromotionDocument objects
  - Convert to SimpleDocument objects
    (text + metadata + id)
    ↓
[Output Manager]
  - Save as JSONL (documents.jsonl)           # Primary output
  - Save as Markdown files (promotion_NNN.md) # Secondary output
  - Generate statistics
    ↓
Output: 71 SimpleDocument objects in 2 formats
        Ready for Phase 2 (Embedding)

[Phase 2: Load JSONL] → Convert to LlamaIndex Documents → Embed
```

### 6.3 Key Classes

**SimpleDocument (Lightweight Document):**
```python
from typing import Dict, Any

class SimpleDocument:
    """
    Lightweight document class for Phase 1 data preparation.

    NO LlamaIndex dependency - Phase 2 will convert to LlamaIndex Documents.
    """

    def __init__(self, text: str, metadata: Dict[str, Any]):
        """
        Initialize SimpleDocument.

        Args:
            text: Natural language description (50-200 tokens)
            metadata: Dictionary with 15+ fields
        """
        self.text = text
        self.metadata = metadata
        self.id = metadata.get('document_id')

    def to_dict(self) -> dict:
        """
        Serialize to dict for JSONL output.

        Returns:
            Dict with id, text, metadata fields
        """
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata
        }

    def to_markdown(self) -> str:
        """
        Generate markdown representation for human review.

        Returns:
            Markdown formatted string
        """
        lines = [
            f"# Promotion: {self.id}",
            "",
            "## Description",
            "",
            self.text,
            "",
            "## Metadata",
            ""
        ]

        for key, value in self.metadata.items():
            lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)
```

**PromotionDocument (Dataclass):**
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class PromotionDocument:
    document_id: str

    # Base fields from CSV
    customer_type: str
    start_date: str  # Buddhist calendar
    end_date: str    # Buddhist calendar
    product_name: str
    buy_quantity: int
    buy_unit: str
    free_item_name: str
    free_quantity: int
    free_unit: str
    additional_conditions: Optional[str]

    # Derived metadata
    customer_tier: str
    is_wholesale: bool
    is_retail: bool
    start_date_gregorian: datetime
    end_date_gregorian: datetime
    is_active: bool
    expires_soon: bool
    duration_days: int
    product_category: str
    product_brand: str
    product_size: Optional[str]
    discount_percentage: float
    promotion_type: str
    keywords: List[str]

    def to_simple_document(self) -> SimpleDocument:
        """
        Convert to SimpleDocument for Phase 1 output.

        Phase 2 will convert SimpleDocument → LlamaIndex Document.

        Returns:
            SimpleDocument object with text + metadata
        """
        text = self._generate_text()
        metadata = self.to_metadata_dict()

        return SimpleDocument(text=text, metadata=metadata)

    def _generate_text(self) -> str:
        """
        Generate natural language text description.

        Returns:
            Human-readable promotion description (50-200 tokens)
        """
        text = f"""
Promotion for {self.customer_type} customers: Buy {self.buy_quantity} {self.buy_unit} of {self.product_name} and get {self.free_quantity} {self.free_unit} of {self.free_item_name} FREE.

Promotion Period: Valid from {self.start_date_gregorian.strftime('%Y-%m-%d')} to {self.end_date_gregorian.strftime('%Y-%m-%d')} ({self.duration_days} days).
Status: {'Active ✓' if self.is_active else 'Expired ✗'}
{('⚠️ Expires soon (within 7 days)' if self.expires_soon else '')}

Promotion Terms:
- Purchase minimum: {self.buy_quantity} {self.buy_unit}
- Free items: {self.free_quantity} {self.free_unit}
- Effective discount: {self.discount_percentage}% off total quantity
- Customer tier: {self.customer_tier} ({'Wholesale' if self.is_wholesale else 'Retail'})

Product Information:
- Product: {self.product_name}
- Category: {self.product_category}
- Brand: {self.product_brand}
- Size: {self.product_size or 'N/A'}

{(f'Additional Conditions: {self.additional_conditions}' if self.additional_conditions else 'No additional conditions.')}
        """.strip()

        return text

    def to_metadata_dict(self) -> dict:
        """
        Generate complete metadata dictionary.

        Returns:
            All fields as dict for SimpleDocument metadata
        """
        return {
            # Base fields
            "document_id": self.document_id,
            "customer_type": self.customer_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "product_name": self.product_name,
            "buy_quantity": self.buy_quantity,
            "buy_unit": self.buy_unit,
            "free_item_name": self.free_item_name,
            "free_quantity": self.free_quantity,
            "free_unit": self.free_unit,
            "additional_conditions": self.additional_conditions,

            # Derived metadata
            "customer_tier": self.customer_tier,
            "is_wholesale": self.is_wholesale,
            "is_retail": self.is_retail,
            "start_date_gregorian": self.start_date_gregorian.strftime('%Y-%m-%d'),
            "end_date_gregorian": self.end_date_gregorian.strftime('%Y-%m-%d'),
            "is_active": self.is_active,
            "expires_soon": self.expires_soon,
            "duration_days": self.duration_days,
            "product_category": self.product_category,
            "product_brand": self.product_brand,
            "product_size": self.product_size,
            "discount_percentage": self.discount_percentage,
            "promotion_type": self.promotion_type,
            "keywords": self.keywords
        }
```

**DataProcessor (Main Orchestrator):**
```python
from typing import List

class PromotionDataProcessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.csv_processor = CSVProcessor(self.config)
        self.metadata_extractor = MetadataExtractor(self.config)
        self.thai_normalizer = ThaiNormalizer(self.config)

    def process(self, csv_path: str, output_dir: str) -> ProcessingResult:
        """
        Main processing pipeline: CSV → SimpleDocuments.

        Returns:
            ProcessingResult with statistics and SimpleDocument list
        """
        # 1. Load CSV
        rows = self.csv_processor.load_csv(csv_path)

        # 2. Process each row
        promotion_docs = []
        for row in rows:
            # 2a. Extract base metadata
            base_metadata = self.csv_processor.extract_fields(row)

            # 2b. Derive metadata (dates, categories, tiers, etc.)
            derived_metadata = self.metadata_extractor.derive_metadata(base_metadata)

            # 2c. Normalize Thai text
            normalized = self.thai_normalizer.normalize(base_metadata, derived_metadata)

            # 2d. Create PromotionDocument
            promo_doc = PromotionDocument(**base_metadata, **derived_metadata, **normalized)

            promotion_docs.append(promo_doc)

        # 3. Convert to SimpleDocuments (NO LlamaIndex!)
        simple_documents: List[SimpleDocument] = [
            doc.to_simple_document() for doc in promotion_docs
        ]

        # 4. Save outputs (dual format)
        self._save_documents_jsonl(simple_documents, output_dir)
        self._save_markdown_files(simple_documents, output_dir)
        self._save_config_snapshot(output_dir)

        # 5. Generate statistics
        stats = self._generate_statistics(promotion_docs, simple_documents)

        # 6. Return result
        return ProcessingResult(
            documents=simple_documents,
            statistics=stats,
            config_used=self.config
        )

    def _save_documents_jsonl(self, documents: List[SimpleDocument], output_dir: str):
        """
        Save SimpleDocuments as JSONL (primary output for Phase 2).

        Format: One JSON object per line (id, text, metadata)
        """
        import json

        jsonl_path = os.path.join(output_dir, 'documents', 'documents.jsonl')
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                json_line = json.dumps(doc.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')

    def _save_markdown_files(self, documents: List[SimpleDocument], output_dir: str):
        """
        Save individual markdown files (secondary output for human review).

        Format: promotion_001.md through promotion_071.md
        """
        markdown_dir = os.path.join(output_dir, 'documents', 'markdown_files')
        os.makedirs(markdown_dir, exist_ok=True)

        for idx, doc in enumerate(documents, 1):
            filename = f"promotion_{str(idx).zfill(3)}.md"
            filepath = os.path.join(markdown_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(doc.to_markdown())
```

### 6.4 Configuration Schema

**promotion_fields.yaml:**
```yaml
name: thai_sales_promotions
version: 2.0
description: Sales promotion data processing configuration (Option 1c+2b)

field_mappings:
  - csv_column: "ประเภทลูกค้า"
    metadata_key: "customer_type"
    field_type: "categorical"
    required: true

  - csv_column: "ระยะเวลาเริ่ม"
    metadata_key: "start_date"
    field_type: "temporal"
    transform: "buddhist_to_gregorian"
    required: true

  - csv_column: "ระยะเวลาสื้นสุด"
    metadata_key: "end_date"
    field_type: "temporal"
    transform: "buddhist_to_gregorian"
    required: true

  # ... more mappings

derived_metadata:
  - name: "product_category"
    compute: "extract_category(product_name)"
    values: ["water", "beer", "spirits", "soda", "other"]

  - name: "customer_tier"
    compute: "map_customer_tier(customer_type)"
    values: ["tier_1", "tier_2", "tier_3"]

  # ... more derived fields

product_patterns:
  water: ["น้ำดื่ม", "PET", "น้ำแร่"]
  beer: ["เบียร์", "beer", "ช้างคลาสสิก"]
  spirits: ["สุราขาว", "spirits", "แสงโสม"]
  soda: ["โคล่า", "cola", "น้ำอัดลม", "ซาโต้"]

customer_tiers:
  tier_1: ["เอเย่นต์", "Super Sub"]
  tier_2: ["ซับเอเย่นต์", "CVA", "ซับซื้อตรง"]
  tier_3: ["รถขายปลีก", "Retail TT"]
```

---

## 7. Implementation Plan

### Phase 1A: Core Infrastructure (Week 1, Days 1-2)

**Tasks:**
1. Create directory structure
2. Implement configuration loading (YAML)
3. Implement CSV processor (UTF-8 handling)
4. Create data models (PromotionDocument, SimpleDocument)

**Deliverables:**
- ✅ Project structure created
- ✅ Config loading with validation
- ✅ CSV parser with Thai support
- ✅ Data models defined

**Copy-Modify from src-imap:**
- `config_manager.py` (adapt for promotion config)
- `csv_analyzer.py` (adapt for promotion CSV structure)
- `models.py` (add SimpleDocument, PromotionDocument)

### Phase 1B: Metadata Extraction (Week 1, Days 3-4)

**Tasks:**
1. Implement Buddhist calendar conversion
2. Implement product categorization
3. Implement customer tier mapping
4. Implement discount calculation

**Deliverables:**
- ✅ Date utilities (with tests)
- ✅ Product utilities (with pattern matching)
- ✅ Customer utilities (tier mapping)
- ✅ Promotion analysis (discount %)

**Copy-Modify from src-imap:**
- `document_processor.py` (adapt metadata extraction logic)

### Phase 1C: Text Generation & Output (Week 1, Day 5 - Week 2, Day 1)

**Tasks:**
1. Implement natural language text generation
2. Implement SimpleDocument creation (`to_simple_document()`)
3. Implement JSONL serialization
4. Implement markdown file generation

**Deliverables:**
- ✅ Text generator (natural language descriptions)
- ✅ SimpleDocument creator
- ✅ JSONL writer (primary output)
- ✅ Markdown file writer (secondary output)

**Copy-Modify from src-imap:**
- `file_output.py` (adapt for dual output format)

### Phase 1D: Testing & Validation (Week 2, Days 2-5)

**Tasks:**
1. Unit tests for all utilities
2. Integration test with sample data (5 promotions first)
3. Full dataset test (71 promotions)
4. Validation against ground truth
5. Performance benchmarking

**Deliverables:**
- ✅ Complete test suite (>80% coverage)
- ✅ Validation report
- ✅ Performance metrics
- ✅ Documentation

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Test Coverage:**
- Date conversion (Buddhist → Gregorian)
- Product categorization (pattern matching)
- Customer tier mapping
- Discount calculation
- Thai text normalization
- SimpleDocument creation

**Example Test Cases:**
```python
def test_buddhist_to_gregorian():
    assert convert_buddhist("2568-10-01") == "2025-10-01"
    assert convert_buddhist("2568-12-31") == "2025-12-31"

def test_product_category():
    assert extract_category("น้ำดื่มช้าง PET 600") == "water"
    assert extract_category("เบียร์ช้างคลาสสิก") == "beer"

def test_customer_tier():
    assert get_tier("เอเย่นต์") == "tier_1"
    assert get_tier("รถขายปลีก") == "tier_3"

def test_discount_calculation():
    assert calculate_discount(108, 42) == 28.0  # 42/(108+42) * 100

def test_simple_document_creation():
    doc = SimpleDocument(text="Test", metadata={"id": "001"})
    assert doc.to_dict()['id'] == "001"
    assert doc.to_dict()['text'] == "Test"
```

### 8.2 Integration Tests

**Test Scenarios:**
1. Process single CSV row → validate SimpleDocument output
2. Process 5 sample promotions
3. Process entire CSV file (71 rows)
4. Validate JSONL format
5. Validate 71 markdown files generated
6. Check metadata completeness
7. Verify Thai encoding preserved

### 8.3 Validation Tests

**Validation Checks:**
- All 71 promotions processed
- No data loss from CSV
- All derived fields computed
- Dates converted correctly
- Thai characters intact
- No duplicate document IDs
- JSONL parseable by Phase 2
- Markdown files human-readable

---

## 9. Success Criteria

### Definition of Done

Phase 1 is complete when:

✅ **All 71 promotions** processed successfully
✅ **71 SimpleDocument objects** created with text + metadata
✅ **documents.jsonl** file with all SimpleDocuments (JSON format)
✅ **71 markdown files** (promotion_001.md to promotion_071.md)
✅ **All SimpleDocuments** have natural language text field (50-200 tokens each)
✅ **All SimpleDocuments** have 15+ metadata fields
✅ **100% Thai encoding** preserved in text and metadata (no corruption)
✅ **100% date conversion** accuracy (Buddhist → Gregorian validated)
✅ **Configuration-driven** (YAML controls all mappings)
✅ **Test coverage** > 80% for all modules
✅ **Processing time** < 2 seconds for full dataset
✅ **Documentation** complete (docstrings + README)
✅ **CLI interface** functional and user-friendly
✅ **Phase 2 ready** - JSONL can be loaded and converted to LlamaIndex Documents

### Acceptance Tests

```bash
# Test 1: Process full dataset
python -m src_sale_promotion.data_processing.run_data_prep
# Expected: 71 SimpleDocument objects in < 2 seconds, no errors

# Test 2: Validate outputs
ls data/processed/documents/
# Expected:
#   - documents.jsonl (71 lines)
#   - markdown_files/ (71 .md files)
#   - README.md, processing_report.json

# Test 3: Validate JSONL format
python -c "
import json

count = 0
with open('data/processed/documents/documents.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        assert 'id' in data
        assert 'text' in data
        assert 'metadata' in data
        assert len(data['metadata']) >= 15
        count += 1

print(f'✅ Validated {count} SimpleDocuments in JSONL')
"
# Expected: ✅ Validated 71 SimpleDocuments in JSONL

# Test 4: Test Phase 2 compatibility
python -c "
from llama_index.core import Document
import json

# Simulate Phase 2 loading
documents = []
with open('data/processed/documents/documents.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        doc = Document(
            text=data['text'],
            metadata=data['metadata'],
            doc_id=data['id']
        )
        documents.append(doc)

print(f'✅ Loaded {len(documents)} LlamaIndex Documents')
print(f'✅ Sample text length: {len(documents[0].text)} chars')
print(f'✅ Sample metadata fields: {len(documents[0].metadata)}')
"
# Expected: All assertions pass, ready for Phase 2

# Test 5: Run test suite
pytest src-sale-promotion/test/test_data_processing.py -v
# Expected: All tests pass, coverage > 80%
```

---

## 10. Dependencies & Prerequisites

### Required

- Python 3.9+
- pandas (CSV processing)
- pyyaml (configuration)
- python-dateutil (date handling)
- **NO llama-index-core** (Phase 1 is lightweight!)

### Optional

- pytest (testing)
- pytest-cov (coverage)
- black (code formatting)
- mypy (type checking)

### Data

- `src-sale-promotion/data/promotion_list.csv` (71 rows)
- UTF-8 encoding verified

---

## 11. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Thai encoding corruption | High | Low | UTF-8 throughout, test early |
| Date conversion errors | High | Low | Extensive testing, validation |
| Product pattern mismatch | Medium | Medium | Fallback to "other" category |
| Performance issues | Low | Low | Small dataset, profile if needed |
| Config complexity | Medium | Medium | Clear examples, validation |
| JSONL format issues | Medium | Low | Validate with Phase 2 test |

---

## 12. Future Enhancements (Phase 2+)

Not in scope for Phase 1, but documented for future:

- [ ] Support multiple CSV files
- [ ] Incremental processing (only new promotions)
- [ ] Advanced product matching (ML-based)
- [ ] Multi-language support (English + Thai)
- [ ] Web-based config editor
- [ ] Real-time validation UI

---

## 13. References

### Methodology Documentation
- `src/METHODOLOGY.md` - General RAG methodology
- `src-sale-promotion/METHODOLOGY.md` - Sales-specific guide
- `attached_assets/00_doc_prep_workflow.md` - Document prep best practices

### Proven Implementations
- **`src-imap/data_processing/`** - PRIMARY REFERENCE (Thai language processing, modular architecture)
- `src/02_prep_doc_for_embedding.py` - General purpose reference

### External Resources
- Thai language processing: https://pythainlp.github.io/
- Python YAML: https://pyyaml.org/
- Python dateutil: https://dateutil.readthedocs.io/

---

## 14. Approval & Sign-off

**Prepared by:** Data Engineering Team
**Date:** 2025-10-26
**Version:** 2.0 (Option 1c+2b)
**Status:** ✅ Ready for Review

**Approvers:**
- [ ] Technical Lead
- [ ] Product Owner
- [ ] Engineering Manager

---

**Next Steps After Approval:**
1. Create feature branch: `feature/sale-promotion-phase1-data-prep-v2`
2. Copy and modify modules from `src-imap/data_processing/` as templates
3. Implement following the plan in Section 7
4. Test with 5 sample promotions first, then full dataset
5. Code review before merge to main

---

## Appendix A: v1.0 → v2.0 Changes Summary

**Major Changes:**
1. ✅ Use SimpleDocument (not LlamaIndex Document) in Phase 1
2. ✅ Dual output: JSONL (primary) + Markdown (secondary)
3. ✅ Remove PKL file output
4. ✅ Add Phase 1→2 handoff specification
5. ✅ Update all code examples to use SimpleDocument
6. ✅ Clarify markdown files for human review only
7. ✅ Document copy-modify strategy from src-imap

**Key Benefits:**
- Lightweight Phase 1 (no LlamaIndex dependency)
- Faster Phase 2 loading (JSONL instead of markdown parsing)
- Better separation of concerns (data prep vs embedding)
- Proven pattern (following src-imap implementation)
