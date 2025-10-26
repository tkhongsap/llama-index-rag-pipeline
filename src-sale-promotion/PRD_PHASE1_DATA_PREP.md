# Product Requirements Document (PRD)
# Phase 1: Sales Promotion Data Preparation Workflow

**Project:** Sales Promotion RAG Pipeline
**Phase:** 1 - Data Preparation
**Version:** 1.0
**Date:** 2025-10-26
**Owner:** Data Engineering Team

---

## 1. Executive Summary

Implement the data preparation workflow for the Sales Promotion RAG pipeline, transforming raw Thai promotional CSV data into structured markdown documents with rich metadata, ready for embedding generation.

### Key Objectives
- ✅ Convert Thai CSV promotion data → Structured documents
- ✅ Extract and derive comprehensive metadata
- ✅ Handle Buddhist calendar dates (2568 → 2025 CE)
- ✅ Normalize Thai product names and customer types
- ✅ Generate markdown documents with NO chunking (atomic promotions)
- ✅ Enable fast metadata filtering for retrieval

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
- **Proven Implementation:** `src/02_prep_doc_for_embedding.py`

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
1. **Transform CSV → Documents** with 100% data preservation
2. **Extract metadata** covering 15+ derived fields
3. **Handle Thai text** with proper UTF-8 encoding
4. **Buddhist calendar** conversion with validation
5. **Generate markdown** with structured format

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Data Preservation** | 100% | All CSV fields preserved in output |
| **Metadata Completeness** | 15+ fields | Derived metadata generated per document |
| **Date Conversion Accuracy** | 100% | Buddhist → Gregorian validation |
| **Processing Speed** | > 50 docs/sec | 71 promotions in < 2 seconds |
| **Thai Encoding** | No errors | UTF-8 preserved in all outputs |
| **Reproducibility** | Config-driven | YAML config generates same output |

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

### 4.5 Document Generation

**FR-5.1: Markdown Format**
Each promotion generates ONE markdown document:

```markdown
# Promotion: Buy น้ำดื่มช้าง PET 600 CC Get Free Items

## Promotion Details
- **Customer Type**: เอเย่นต์ (Tier 1 - Wholesale)
- **Period**: October 1, 2025 - December 31, 2025 (92 days)
- **Status**: Active ✓

## Terms
- **Buy**: 108 แพ็ค of น้ำดื่มช้าง PET 600 CC (1 แพ็ค = 12 ขวด)
- **Get Free**: 42 แพ็ค of น้ำดื่มช้าง PET 600 CC

## Promotion Analysis
- **Effective Discount**: 28.0%
- **Effective Price Ratio**: 0.72 (pay for 72% of total quantity)
- **Promotion Type**: Buy X Get Y

## Product Information
- **Product**: น้ำดื่มช้าง PET 600 CC
- **Category**: Water
- **Brand**: ช้าง (Chang)
- **Size**: 600 CC
- **Free Item**: น้ำดื่มช้าง PET 600 CC (same product)

## Additional Conditions
[None specified]

---
*Document ID: PROMO_001*
*Generated: 2025-10-26*
```

**FR-5.2: Metadata JSON**
Generate companion JSON with all metadata:
```json
{
  "document_id": "PROMO_001",
  "customer_type": "เอเย่นต์",
  "customer_tier": "tier_1",
  "is_wholesale": true,
  "is_retail": false,
  "start_date": "2568-10-01 00:00:00",
  "end_date": "2568-12-31 00:00:00",
  "start_date_gregorian": "2025-10-01",
  "end_date_gregorian": "2025-12-31",
  "is_active": true,
  "expires_soon": false,
  "duration_days": 92,
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
  "keywords": ["น้ำดื่มช้าง", "PET", "600", "เอเย่นต์", "water", "chang"]
}
```

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
├── markdown_files/
│   ├── PROMO_001.md
│   ├── PROMO_002.md
│   └── ... (71 files)
├── metadata/
│   ├── PROMO_001_metadata.json
│   ├── PROMO_002_metadata.json
│   └── ... (71 files)
├── documents.jsonl          # All documents in JSONL format
├── config_used.yaml         # Configuration used for this run
├── processing_report.json   # Statistics and validation
└── README.md                # Processing summary
```

**FR-7.2: Statistics Tracking**
- Total documents processed
- Fields extracted (base + derived)
- Date conversions performed
- Validation errors (if any)
- Processing time

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
│   └── document_generator.py       # Markdown generation
├── utils/
│   ├── __init__.py
│   ├── date_utils.py               # Buddhist calendar conversion
│   ├── product_utils.py            # Product categorization
│   └── customer_utils.py           # Customer tier mapping
├── models.py                        # Data classes
└── run_data_prep.py                # Main orchestrator
```

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
  - Extract base fields
    ↓
[Metadata Extractor]
  - Convert dates (Buddhist → Gregorian)
  - Categorize products
  - Calculate discounts
  - Assign customer tiers
    ↓
[Thai Normalizer]
  - Normalize product names
  - Extract keywords
  - Generate search text
    ↓
[Document Generator]
  - Create markdown documents
  - Generate metadata JSON
  - Save to files
    ↓
Output Files (Markdown + JSON + Statistics)
```

### 6.3 Key Classes

**PromotionDocument (Dataclass):**
```python
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

    def to_markdown(self) -> str:
        """Generate markdown representation."""

    def to_metadata_dict(self) -> dict:
        """Generate metadata dictionary."""
```

**DataProcessor (Main Orchestrator):**
```python
class PromotionDataProcessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.csv_processor = CSVProcessor(self.config)
        self.metadata_extractor = MetadataExtractor(self.config)
        self.thai_normalizer = ThaiNormalizer(self.config)
        self.doc_generator = DocumentGenerator(self.config)

    def process(self, csv_path: str, output_dir: str) -> ProcessingResult:
        """Main processing pipeline."""
        # 1. Load CSV
        # 2. Extract metadata
        # 3. Normalize Thai text
        # 4. Generate documents
        # 5. Save outputs
        # 6. Return statistics
```

### 6.4 Configuration Schema

**promotion_fields.yaml:**
```yaml
name: thai_sales_promotions
version: 1.0

field_mappings:
  - csv_column: "ระยะเวลาเริ่ม"
    metadata_key: "start_date"
    field_type: "temporal"
    transform: "buddhist_to_gregorian"
  # ... more mappings

derived_metadata:
  - name: "product_category"
    compute: "extract_category(product_name)"
    values: ["water", "beer", "spirits", "soda"]
  # ... more derived fields

product_patterns:
  water: ["น้ำดื่ม", "PET"]
  beer: ["เบียร์", "beer"]
  # ... more patterns

customer_tiers:
  tier_1: ["เอเย่นต์", "Super Sub"]
  # ... more tiers
```

---

## 7. Implementation Plan

### Phase 1A: Core Infrastructure (Week 1, Days 1-2)

**Tasks:**
1. Create directory structure
2. Implement configuration loading (YAML)
3. Implement CSV processor (UTF-8 handling)
4. Create data models (PromotionDocument)

**Deliverables:**
- ✅ Project structure created
- ✅ Config loading with validation
- ✅ CSV parser with Thai support
- ✅ Data models defined

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

### Phase 1C: Thai Normalization (Week 1, Day 5)

**Tasks:**
1. Implement product name normalization
2. Implement keyword extraction
3. Implement search text generation

**Deliverables:**
- ✅ Thai normalizer module
- ✅ Keyword extraction logic
- ✅ Search text generator

### Phase 1D: Document Generation (Week 2, Days 1-2)

**Tasks:**
1. Implement markdown template
2. Implement metadata JSON generation
3. Implement file output management

**Deliverables:**
- ✅ Markdown generator
- ✅ JSON metadata export
- ✅ File organization

### Phase 1E: Testing & Validation (Week 2, Days 3-5)

**Tasks:**
1. Unit tests for all utilities
2. Integration test with sample data
3. Validation against ground truth
4. Performance benchmarking

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
```

### 8.2 Integration Tests

**Test Scenarios:**
1. Process single CSV row → validate output
2. Process entire CSV file (71 rows)
3. Validate all 71 markdown files generated
4. Check metadata completeness
5. Verify Thai encoding preserved

### 8.3 Validation Tests

**Validation Checks:**
- All 71 promotions processed
- No data loss from CSV
- All derived fields computed
- Dates converted correctly
- Thai characters intact
- No duplicate document IDs

---

## 9. Success Criteria

### Definition of Done

Phase 1 is complete when:

✅ **All 71 promotions** processed successfully
✅ **71 markdown files** generated with structured content
✅ **71 metadata JSON** files with 15+ fields each
✅ **100% Thai encoding** preserved (no corruption)
✅ **100% date conversion** accuracy (validated)
✅ **Configuration-driven** (YAML controls all mappings)
✅ **Test coverage** > 80% for all modules
✅ **Processing time** < 2 seconds for full dataset
✅ **Documentation** complete (docstrings + README)
✅ **CLI interface** functional and user-friendly

### Acceptance Tests

```bash
# Test 1: Process full dataset
python -m src_sale_promotion.data_processing.run_data_prep
# Expected: 71 files in < 2 seconds, no errors

# Test 2: Validate outputs
python -m src_sale_promotion.data_processing.validate_output
# Expected: All validation checks pass

# Test 3: Run test suite
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
- `src/02_prep_doc_for_embedding.py` - Reference implementation
- `src-iLand/data_processing/` - Thai language processing example

### External Resources
- LlamaIndex Document preparation: https://docs.llamaindex.ai/en/stable/
- Thai language processing: https://pythainlp.github.io/

---

## 14. Approval & Sign-off

**Prepared by:** AI Engineering Team
**Date:** 2025-10-26
**Status:** ✅ Ready for Review

**Approvers:**
- [ ] Technical Lead
- [ ] Product Owner
- [ ] Engineering Manager

---

**Next Steps After Approval:**
1. Create feature branch: `feature/sale-promotion-phase1-data-prep`
2. Begin implementation following the plan in Section 7
3. Daily standups to track progress
4. Code review before merge to main

