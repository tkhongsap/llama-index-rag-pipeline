# Phase 4 PRD: CLI Implementation & Test Set

**Project:** Sales Promotion RAG Pipeline
**Phase:** 4 - CLI Implementation & Test Set Creation
**Version:** 1.0
**Date:** 2025-10-26
**Status:** Draft
**Author:** Development Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Objectives & Success Criteria](#2-objectives--success-criteria)
3. [CLI Requirements](#3-cli-requirements)
4. [Test Set Specifications](#4-test-set-specifications)
5. [Module Architecture](#5-module-architecture)
6. [CLI Design Specifications](#6-cli-design-specifications)
7. [Test Framework Design](#7-test-framework-design)
8. [Performance Targets](#8-performance-targets)
9. [Phase Dependencies](#9-phase-dependencies)
10. [Implementation Plan](#10-implementation-plan)
11. [Example Usage](#11-example-usage)
12. [References & Appendix](#12-references--appendix)

---

## 1. Executive Summary

### 1.1 Overview

Phase 4 delivers a **simple, production-ready CLI** for querying 71 Thai sales promotion documents using the 7 retrieval strategies implemented in Phase 3. The phase also creates a **comprehensive 10-question test set** to validate end-to-end system quality.

**Core Focus:**
- ✅ Minimalist CLI: Just type questions, get answers
- ✅ 10 diverse test questions from real promotion data
- ✅ Automated test runner with quality metrics
- ✅ Full Thai/English language support

### 1.2 User Personas

**Primary Users:**
1. **Business Analysts** - Query promotions for sales analysis
2. **Sales Team** - Find relevant promotions for customers
3. **QA Engineers** - Validate system quality and accuracy
4. **Developers** - Test and debug retrieval strategies

**User Goals:**
- Query promotions naturally in Thai or English
- Get accurate, relevant results quickly
- Understand which retrieval strategy was used
- Validate system performance with standardized tests

### 1.3 Phase Context

**Phase 1 (Complete):** CSV → Markdown → JSONL (71 promotions)
**Phase 2 (Complete):** JSONL → Embeddings (142 embeddings, 3072 dims)
**Phase 3 (Complete):** 7 Retrieval Strategies + Two-Stage Router
**Phase 4 (This Phase):** CLI + 10-Question Test Set

### 1.4 Key Deliverables

| # | Deliverable | Description | Lines of Code |
|---|-------------|-------------|---------------|
| 1 | `cli.py` | Main CLI interface | ~300 |
| 2 | `test_set.py` | 10 test questions + expected outputs | ~250 |
| 3 | `test_runner.py` | Automated test execution | ~200 |
| 4 | `README_CLI.md` | CLI usage documentation | ~300 |
| 5 | Test Reports | Automated test results | Generated |

**Total:** ~1,050 lines of code + documentation

### 1.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **CLI Functionality** | 100% working | All features operational |
| **Test Pass Rate** | ≥ 8/10 (80%) | Automated test execution |
| **Strategy Selection Accuracy** | ≥ 80% | Correct strategy chosen |
| **Result Relevance** | ≥ 85% | Top-3 results relevant |
| **Response Time** | < 20s | Query → answer (with GPT-5) |
| **Thai Text Rendering** | 100% | UTF-8 display correct |

---

## 2. Objectives & Success Criteria

### 2.1 Primary Objectives

**Objective 1: Simple CLI Interface**
- **Goal:** Enable users to query promotions by simply typing questions
- **Design Principle:** Minimize complexity, maximize usability
- **Target UX:** "Type question → Get answer" (2 steps)

**Objective 2: Comprehensive Test Set**
- **Goal:** Create 10 diverse test questions covering all retrieval strategies
- **Coverage:** Customer types, product categories, discounts, dates, complex queries
- **Validation:** Automated quality metrics for system validation

**Objective 3: Quality Assurance**
- **Goal:** Validate end-to-end RAG pipeline quality
- **Metrics:** Strategy selection, retrieval accuracy, response quality
- **Automation:** Repeatable test execution with reports

### 2.2 Functional Success Criteria

#### CLI Requirements ✅

| Requirement | Success Criteria |
|-------------|------------------|
| **Interactive Mode** | User can type questions continuously until exit |
| **Single Query Mode** | User can run one query and exit |
| **Strategy Display** | Show which strategy was selected and why |
| **Result Formatting** | Clean, readable output with Thai UTF-8 support |
| **Error Handling** | Graceful failure with helpful error messages |
| **Session History** | Track queries within a session (optional) |

#### Test Set Requirements ✅

| Requirement | Success Criteria |
|-------------|------------------|
| **10 Test Questions** | Cover all 7 retrieval strategies |
| **Diversity** | Mix of customer types, products, filters |
| **Language Mix** | Both Thai and English queries |
| **Expected Outputs** | Each test has expected strategy + results |
| **Validation Logic** | Automated pass/fail determination |
| **Performance Benchmarks** | Measure response time per query |

#### Test Runner Requirements ✅

| Requirement | Success Criteria |
|-------------|------------------|
| **Automated Execution** | Run all 10 tests without intervention |
| **Quality Metrics** | Calculate accuracy, relevance, performance |
| **Report Generation** | JSON results + Markdown summary |
| **Pass/Fail Status** | Clear indication of test outcomes |
| **Debugging Info** | Detailed logs for failed tests |

### 2.3 Quality Success Criteria

#### Retrieval Quality ✅

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| **Strategy Selection Accuracy** | ≥ 80% | Compare selected vs expected strategy |
| **Top-1 Relevance** | ≥ 70% | Most relevant result in top-1 |
| **Top-3 Relevance** | ≥ 85% | Relevant result in top-3 |
| **Top-5 Relevance** | ≥ 95% | Relevant result in top-5 |
| **Zero Results** | < 5% | Queries returning no results |

#### Performance Quality ✅

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| **Average Response Time** | < 20s | Time from query to response |
| **Pure Retrieval Time** | < 500ms | Retrieval without GPT-5 synthesis |
| **Cache Hit Rate** | > 30% | After warmup period |
| **Strategy Routing Time** | < 2s | Two-stage routing decision |

#### User Experience Quality ✅

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| **Thai Text Display** | 100% correct | Visual inspection + encoding check |
| **Output Clarity** | Subjective high | User feedback |
| **Error Messages** | Clear & actionable | Error scenario testing |
| **CLI Responsiveness** | No hangs | Timeout testing |

---

## 3. CLI Requirements

### 3.1 CLI Design Principles

**Principle 1: Simplicity First**
- No complex configuration required
- Start CLI → Type question → Get answer
- Sensible defaults for all parameters

**Principle 2: Progressive Disclosure**
- Basic mode: Just query and response
- Verbose mode: Show strategy selection, metadata, timing
- Debug mode: Full retrieval details

**Principle 3: Thai-First Design**
- Native UTF-8 support
- Thai keyword extraction
- Thai output formatting

### 3.2 CLI Modes

#### Mode 1: Interactive Mode (Primary)

**Description:** Continuous query-response loop until user exits

**Usage:**
```bash
python -m src_sale_promotion.retrieval.cli --interactive
```

**Flow:**
```
1. Load embeddings and indices
2. Show welcome message
3. LOOP:
   - Prompt: "Query: "
   - User types question
   - Process query (router → retriever → GPT-5)
   - Display results
   - Show strategy used + metadata
4. Exit on 'exit', 'quit', or Ctrl+C
```

**Example Session:**
```
=== Sales Promotion Query System ===
Loaded 142 embeddings (71 promotions)
Type 'exit' to quit

Query: โปรโมชันสำหรับเอเย่นต์
Strategy: metadata (confidence: 0.94)

Found 25 promotions for เอเย่นต์ customers:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC (28% discount)
2. PROMO_025 - น้ำดื่มช้าง PET 1500 CC (28% discount)
3. PROMO_042 - แสงโสม สปอนเซอร์ (36.85% discount)
...

Query: exit
Goodbye!
```

#### Mode 2: Single Query Mode

**Description:** Run one query and exit

**Usage:**
```bash
python -m src_sale_promotion.retrieval.cli --query "โปรโมชันน้ำดื่มช้าง"
```

**Flow:**
```
1. Load embeddings and indices
2. Process query
3. Display results
4. Exit
```

**Output:**
```
Query: โปรโมชันน้ำดื่มช้าง
Strategy: vector (confidence: 0.96)

Found 39 water promotions:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC for เอเย่นต์ (28% discount)
2. PROMO_008 - น้ำดื่มช้าง PET 600 CC for ซับเอเย่นต์ (33.3% discount)
3. PROMO_015 - น้ำดื่มช้าง PET 600 CC for ซับเอเย่นต์ (28% discount)
...

Response time: 8.4s
```

#### Mode 3: Test Mode

**Description:** Run predefined test questions

**Usage:**
```bash
python -m src_sale_promotion.retrieval.cli --run-tests
```

**Flow:**
```
1. Load embeddings and indices
2. Load test set (10 questions)
3. FOR EACH test:
   - Run query
   - Validate results
   - Record metrics
4. Generate test report
5. Display summary
```

**Output:**
```
=== Running Test Set (10 questions) ===

Test 1/10: โปรโมชันสำหรับเอเย่นต์
  Strategy: metadata (expected: metadata) ✓
  Results: 25 found
  Relevance: 95% ✓
  Time: 8.2s
  Status: PASS

Test 2/10: โปรโมชันน้ำดื่มช้าง
  Strategy: vector (expected: vector) ✓
  Results: 39 found
  Relevance: 100% ✓
  Time: 6.8s
  Status: PASS

...

=== Test Summary ===
Total: 10
Passed: 9 (90%)
Failed: 1 (10%)
Avg Response Time: 8.5s
Strategy Accuracy: 90%
Relevance (Top-3): 92%

Report saved: test_results/test_results_20251026_143052.json
```

### 3.3 CLI Arguments

```bash
python -m src_sale_promotion.retrieval.cli [OPTIONS]
```

**Arguments:**

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--interactive` | `-i` | flag | False | Start interactive mode |
| `--query` | `-q` | string | None | Single query mode |
| `--run-tests` | `-t` | flag | False | Run test set |
| `--verbose` | `-v` | flag | False | Show detailed output |
| `--debug` | `-d` | flag | False | Show debug information |
| `--load-embeddings` | `-l` | path | latest | Embedding directory |
| `--top-k` | `-k` | int | 5 | Number of results |
| `--strategy` | `-s` | string | auto | Force strategy (vector/metadata/hybrid/etc.) |
| `--no-synthesis` | | flag | False | Skip GPT-5 response synthesis |
| `--cache` | | flag | True | Enable query caching |

**Examples:**

```bash
# Interactive mode
python -m src_sale_promotion.retrieval.cli -i

# Single query
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์"

# Single query with verbose output
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" -v

# Force specific strategy
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" -s metadata

# Run test set
python -m src_sale_promotion.retrieval.cli -t

# Debug mode
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" -d
```

### 3.4 Output Formatting

#### Basic Output Format

```
Query: [user query]
Strategy: [strategy_name] (confidence: [0.00-1.00])

[GPT-5 generated response in Thai or English]

Promotions Found: [count]

[Formatted list of promotions with key details]

---
Response Time: [X.X]s
Strategy: [strategy_name]
Retrieved: [N] documents
Cache: [HIT/MISS]
```

#### Verbose Output Format

```
=== Query Analysis ===
Query: [user query]
Language: [Thai/English/Mixed]
Detected Filters: [filter_dict]
Query Length: [N] characters

=== Routing Decision ===
Stage 1 - Index: sales_promotions
Stage 2 - Strategy: metadata
Confidence: 0.94
Reasoning: Query contains explicit customer type filter (เอเย่นต์)

=== Retrieval Results ===
Retrieved: 25 documents
Filtered: 46 → 25 (by metadata)
Retrieval Time: 0.03s

=== Top Results ===
1. PROMO_001 (score: 0.92)
   - Customer: เอเย่นต์
   - Product: น้ำดื่มช้าง PET 600 CC
   - Discount: 28%

2. PROMO_025 (score: 0.89)
   - Customer: เอเย่นต์
   - Product: น้ำดื่มช้าง PET 1500 CC
   - Discount: 28%

=== GPT-5 Response Synthesis ===
Synthesis Time: 8.1s
Model: gpt-5
Response Length: 456 characters

=== Final Response ===
[GPT-5 response]

=== Performance Metrics ===
Total Time: 8.4s
  - Routing: 1.2s
  - Retrieval: 0.03s
  - Synthesis: 8.1s
Cache: MISS
```

#### Debug Output Format

Includes all verbose output plus:
- Full node metadata for retrieved documents
- Embedding vectors (first 5 dims)
- Filter application details
- Cache key hashes
- Error stack traces

### 3.5 Error Handling

#### Error Categories

**1. Input Errors**
- Empty query
- Invalid strategy name
- Invalid arguments

**2. System Errors**
- Embeddings not loaded
- Index not built
- API key missing

**3. Processing Errors**
- Routing failure
- Retrieval failure
- Synthesis failure

**4. Network Errors**
- OpenAI API timeout
- Rate limit exceeded

#### Error Messages

```python
# Empty query
"Error: Query cannot be empty. Please enter a question."

# Embeddings not found
"Error: Embeddings not found at [path]. Please run Phase 2 first or specify correct path with --load-embeddings."

# API key missing
"Error: OPENAI_API_KEY not found. Please set in .env file."

# Strategy not found
"Error: Strategy 'xyz' not recognized. Available: vector, metadata, hybrid, summary, recursive, chunk_decoupling, planner"

# Retrieval failure
"Error: Retrieval failed. [error_message]
Falling back to vector strategy..."

# API timeout
"Error: OpenAI API timeout after 30s. Please try again."
```

### 3.6 CLI Implementation Structure

**File:** `src-sale-promotion/retrieval/cli.py` (~300 lines)

```python
# CLI Module Structure

## 1. Imports & Configuration (30 lines)
import argparse
import sys
from pathlib import Path
from typing import Optional
# ... other imports

## 2. CLI Class (200 lines)
class SalesPromotionCLI:
    """Main CLI interface for sales promotion queries"""

    def __init__(self, config: dict):
        """Initialize CLI with configuration"""

    def load_system(self):
        """Load embeddings, build indices, initialize router"""

    def interactive_mode(self):
        """Run interactive query loop"""

    def single_query_mode(self, query: str):
        """Process single query and exit"""

    def test_mode(self):
        """Run test set"""

    def process_query(self, query: str):
        """Process a single query through the pipeline"""

    def format_output(self, result: dict, verbose: bool):
        """Format and display results"""

    def handle_error(self, error: Exception):
        """Handle and display errors gracefully"""

## 3. Helper Functions (40 lines)
def parse_arguments():
    """Parse CLI arguments"""

def load_config():
    """Load configuration from environment"""

def display_welcome():
    """Show welcome message"""

## 4. Main Entry Point (30 lines)
def main():
    """CLI entry point"""
    args = parse_arguments()
    config = load_config()

    cli = SalesPromotionCLI(config)
    cli.load_system()

    if args.interactive:
        cli.interactive_mode()
    elif args.query:
        cli.single_query_mode(args.query)
    elif args.run_tests:
        cli.test_mode()
    else:
        print("No mode specified. Use --help for options.")

if __name__ == "__main__":
    main()
```

---

## 4. Test Set Specifications

### 4.1 Test Set Overview

**Goal:** Create 10 diverse test questions that validate all retrieval strategies and cover common query patterns.

**Coverage Matrix:**

| Test # | Query Type | Strategy | Customer Type | Product | Language |
|--------|-----------|----------|---------------|---------|----------|
| 1 | Customer filter | metadata | เอเย่นต์ | All | Thai |
| 2 | Product search | vector | All | Water | Thai |
| 3 | Discount filter | metadata | All | All | Thai |
| 4 | Active status | metadata | All | All | Thai |
| 5 | Multi-criteria | hybrid | ซับเอเย่นต์ | Beer | Thai |
| 6 | Free gift | vector | All | All | Thai |
| 7 | Brand search | vector | All | ช้าง brand | Thai |
| 8 | Size comparison | summary | All | Water | English |
| 9 | Expired promos | metadata | All | All | Thai |
| 10 | Best value | planner | Wholesale | All | English |

### 4.2 Test Question Definitions

#### Test 1: Customer Type Filtering

**Query (Thai):** `"โปรโมชันสำหรับเอเย่นต์"`

**English Translation:** "Promotions for agents"

**Expected Strategy:** `metadata`

**Expected Filter Detection:**
```python
{
    "customer_type": "เอเย่นต์"
}
```

**Expected Results:**
- Document count: 25 promotions
- All results must have `customer_type == "เอเย่นต์"`
- Top results should include PROMO_001, PROMO_025, PROMO_042

**Validation Criteria:**
- ✅ Strategy selected: metadata
- ✅ Results count: 20-30
- ✅ All results have correct customer_type
- ✅ Top-3 relevance: > 90%

**Performance Target:** < 10s

---

#### Test 2: Product Category Search

**Query (Thai):** `"โปรโมชันน้ำดื่มช้าง"`

**English Translation:** "Chang drinking water promotions"

**Expected Strategy:** `vector` or `hybrid`

**Expected Results:**
- Document count: 30-40 promotions
- All results should be water category
- Brand should be "ช้าง" (Chang)
- Top results: PROMO_001, PROMO_008, PROMO_015, PROMO_025

**Validation Criteria:**
- ✅ Strategy selected: vector or hybrid
- ✅ Results count: 30-40
- ✅ Top-5 all water category
- ✅ Top-3 relevance: > 95%

**Performance Target:** < 12s

---

#### Test 3: Discount Range Filtering

**Query (Thai):** `"โปรโมชันที่ให้ส่วนลดมากกว่า 25%"`

**English Translation:** "Promotions with discount greater than 25%"

**Expected Strategy:** `metadata`

**Expected Filter Detection:**
```python
{
    "discount_percentage": {"$gt": 25}
}
```

**Expected Results:**
- Document count: 25-35 promotions
- All results must have `discount_percentage > 25`
- Should include high-discount promotions like PROMO_008 (33.3%), PROMO_042 (36.85%)

**Validation Criteria:**
- ✅ Strategy selected: metadata
- ✅ Results count: 25-35
- ✅ All results have discount > 25%
- ✅ Sorted by discount descending

**Performance Target:** < 10s

---

#### Test 4: Active Promotions

**Query (Thai):** `"โปรโมชันที่กำลังใช้งานอยู่"`

**English Translation:** "Currently active promotions"

**Expected Strategy:** `metadata`

**Expected Filter Detection:**
```python
{
    "is_active": True
}
```

**Expected Results:**
- Document count: 40-50 promotions (depends on current date)
- All results must have `is_active == True`
- Should exclude expired promotions like PROMO_055, PROMO_060

**Validation Criteria:**
- ✅ Strategy selected: metadata
- ✅ Results count: > 30
- ✅ All results have is_active == True
- ✅ No expired promotions in results

**Performance Target:** < 10s

---

#### Test 5: Multi-Criteria Query

**Query (Thai):** `"โปรโมชันเบียร์สำหรับซับเอเย่นต์"`

**English Translation:** "Beer promotions for sub-agents"

**Expected Strategy:** `hybrid` or `metadata`

**Expected Filter Detection:**
```python
{
    "customer_type": "ซับเอเย่นต์",
    "product_category": "beer"
}
```

**Expected Results:**
- Document count: 5-10 promotions
- All results: customer_type = "ซับเอเย่นต์" AND product_category = "beer"

**Validation Criteria:**
- ✅ Strategy selected: hybrid or metadata
- ✅ Results count: 5-15
- ✅ All results match both filters
- ✅ Top-3 relevance: > 85%

**Performance Target:** < 15s

---

#### Test 6: Free Gift Queries

**Query (Thai):** `"โปรโมชันที่ให้ของแถม"`

**English Translation:** "Promotions with free gifts"

**Expected Strategy:** `vector`

**Expected Results:**
- Document count: 60-70 promotions (most have free gifts)
- All results should have `promotion_type == "buy_x_get_y"`
- Should highlight promotions with significant free quantities

**Validation Criteria:**
- ✅ Strategy selected: vector
- ✅ Results count: > 50
- ✅ All results are buy_x_get_y type
- ✅ Top-3 relevance: > 80%

**Performance Target:** < 12s

---

#### Test 7: Brand-Specific Search

**Query (Thai):** `"โปรโมชันสินค้าแบรนด์ช้าง"`

**English Translation:** "Chang brand product promotions"

**Expected Strategy:** `vector` or `hybrid`

**Expected Results:**
- Document count: 50-60 promotions
- All results should have `product_brand == "ช้าง"`
- Should include both water and beer products

**Validation Criteria:**
- ✅ Strategy selected: vector or hybrid
- ✅ Results count: 50-60
- ✅ Top-5 all Chang brand
- ✅ Top-3 relevance: > 90%

**Performance Target:** < 12s

---

#### Test 8: Size Comparison Query

**Query (English):** `"Compare promotions for 600 CC vs 1500 CC water bottles"`

**Thai Alternative:** `"เปรียบเทียบโปรโมชันขนาด 600 CC กับ 1500 CC"`

**Expected Strategy:** `summary` or `planner`

**Expected Results:**
- Should retrieve promotions for both sizes
- Should provide comparative analysis
- Should mention PROMO_001 (600 CC) and PROMO_025 (1500 CC)

**Validation Criteria:**
- ✅ Strategy selected: summary or planner
- ✅ Results include both sizes
- ✅ Response provides comparison
- ✅ Mentions discount differences

**Performance Target:** < 20s

---

#### Test 9: Expired Promotions

**Query (Thai):** `"โปรโมชันที่หมดอายุแล้ว"`

**English Translation:** "Expired promotions"

**Expected Strategy:** `metadata`

**Expected Filter Detection:**
```python
{
    "is_active": False
}
```

**Expected Results:**
- Document count: 20-30 promotions
- All results must have `is_active == False`
- Should include PROMO_055, PROMO_060

**Validation Criteria:**
- ✅ Strategy selected: metadata
- ✅ Results count: 20-30
- ✅ All results have is_active == False
- ✅ No active promotions in results

**Performance Target:** < 10s

---

#### Test 10: Best Value Analysis

**Query (English):** `"Which promotion offers the best value for wholesale customers?"`

**Thai Alternative:** `"โปรโมชันไหนคุ้มที่สุดสำหรับลูกค้าขายส่ง"`

**Expected Strategy:** `planner`

**Expected Results:**
- Should analyze multiple promotions
- Should consider discount percentage, quantity, customer type
- Should provide reasoning for "best value"
- May mention PROMO_008 (33.3% discount) or PROMO_042 (36.85%)

**Validation Criteria:**
- ✅ Strategy selected: planner
- ✅ Response includes analysis
- ✅ Mentions discount percentage
- ✅ Considers wholesale customer context

**Performance Target:** < 25s

---

### 4.3 Test Set Data Structure

**File:** `src-sale-promotion/retrieval/test_set.py` (~250 lines)

```python
# Test Set Structure

from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TestQuestion:
    """Single test question definition"""

    test_id: int
    query: str
    query_english: str
    language: str  # "Thai" or "English"

    # Expected routing
    expected_strategy: str
    alternative_strategies: List[str]  # Acceptable alternatives

    # Expected filters (for metadata strategy)
    expected_filters: Optional[Dict] = None

    # Expected results
    expected_count_min: int = 1
    expected_count_max: int = 100
    expected_doc_ids: List[str] = None  # Must appear in results

    # Validation criteria
    validation_rules: Dict = None

    # Performance
    timeout: int = 30  # seconds

# 10 Test Questions
TEST_SET: List[TestQuestion] = [
    # Test 1: Customer Type Filtering
    TestQuestion(
        test_id=1,
        query="โปรโมชันสำหรับเอเย่นต์",
        query_english="Promotions for agents",
        language="Thai",
        expected_strategy="metadata",
        alternative_strategies=["vector"],
        expected_filters={"customer_type": "เอเย่นต์"},
        expected_count_min=20,
        expected_count_max=30,
        expected_doc_ids=["PROMO_001", "PROMO_025"],
        validation_rules={
            "all_results_match_customer_type": True,
            "top_3_relevance_threshold": 0.90
        },
        timeout=15
    ),

    # Test 2: Product Category Search
    TestQuestion(
        test_id=2,
        query="โปรโมชันน้ำดื่มช้าง",
        query_english="Chang drinking water promotions",
        language="Thai",
        expected_strategy="vector",
        alternative_strategies=["hybrid"],
        expected_count_min=30,
        expected_count_max=45,
        expected_doc_ids=["PROMO_001", "PROMO_008", "PROMO_015"],
        validation_rules={
            "top_5_all_water_category": True,
            "top_3_relevance_threshold": 0.95
        },
        timeout=15
    ),

    # ... (Tests 3-10 defined similarly)
]

def get_test_set() -> List[TestQuestion]:
    """Return complete test set"""
    return TEST_SET

def get_test_by_id(test_id: int) -> TestQuestion:
    """Get specific test by ID"""
    return next((t for t in TEST_SET if t.test_id == test_id), None)
```

### 4.4 Test Validation Logic

**File:** `src-sale-promotion/retrieval/test_runner.py` (~200 lines)

```python
# Test Runner Module

from dataclasses import dataclass
from typing import List, Dict
import time
import json
from pathlib import Path

@dataclass
class TestResult:
    """Result of a single test execution"""

    test_id: int
    query: str

    # Execution results
    selected_strategy: str
    strategy_match: bool

    results_count: int
    results_count_valid: bool

    response_time: float
    response_time_valid: bool

    # Validation results
    validation_passed: Dict[str, bool]
    overall_pass: bool

    # Retrieved documents
    retrieved_doc_ids: List[str]
    top_3_scores: List[float]

    # Error info
    error: Optional[str] = None

class TestRunner:
    """Automated test execution and validation"""

    def __init__(self, cli):
        """Initialize with CLI instance"""
        self.cli = cli

    def run_all_tests(self) -> List[TestResult]:
        """Run all 10 tests"""
        results = []

        for test in TEST_SET:
            print(f"Running Test {test.test_id}/10: {test.query}")
            result = self.run_single_test(test)
            results.append(result)

        return results

    def run_single_test(self, test: TestQuestion) -> TestResult:
        """Run a single test and validate"""
        start_time = time.time()

        try:
            # Execute query
            result = self.cli.process_query(test.query)
            response_time = time.time() - start_time

            # Validate strategy selection
            strategy_match = (
                result['strategy'] == test.expected_strategy or
                result['strategy'] in test.alternative_strategies
            )

            # Validate result count
            results_count = len(result['nodes'])
            results_count_valid = (
                test.expected_count_min <= results_count <= test.expected_count_max
            )

            # Validate response time
            response_time_valid = response_time <= test.timeout

            # Run custom validation rules
            validation_passed = self.validate_rules(test, result)

            # Overall pass/fail
            overall_pass = (
                strategy_match and
                results_count_valid and
                response_time_valid and
                all(validation_passed.values())
            )

            return TestResult(
                test_id=test.test_id,
                query=test.query,
                selected_strategy=result['strategy'],
                strategy_match=strategy_match,
                results_count=results_count,
                results_count_valid=results_count_valid,
                response_time=response_time,
                response_time_valid=response_time_valid,
                validation_passed=validation_passed,
                overall_pass=overall_pass,
                retrieved_doc_ids=[n.node_id for n in result['nodes']],
                top_3_scores=[n.score for n in result['nodes'][:3]]
            )

        except Exception as e:
            return TestResult(
                test_id=test.test_id,
                query=test.query,
                selected_strategy="ERROR",
                strategy_match=False,
                results_count=0,
                results_count_valid=False,
                response_time=time.time() - start_time,
                response_time_valid=False,
                validation_passed={},
                overall_pass=False,
                retrieved_doc_ids=[],
                top_3_scores=[],
                error=str(e)
            )

    def validate_rules(self, test: TestQuestion, result: dict) -> Dict[str, bool]:
        """Validate custom rules for test"""
        validation = {}

        if not test.validation_rules:
            return validation

        # Rule: all_results_match_customer_type
        if "all_results_match_customer_type" in test.validation_rules:
            expected_type = test.expected_filters.get("customer_type")
            validation["all_results_match_customer_type"] = all(
                n.metadata.get("customer_type") == expected_type
                for n in result['nodes']
            )

        # Rule: top_3_relevance_threshold
        if "top_3_relevance_threshold" in test.validation_rules:
            threshold = test.validation_rules["top_3_relevance_threshold"]
            top_3_scores = [n.score for n in result['nodes'][:3]]
            validation["top_3_relevance_threshold"] = (
                len(top_3_scores) > 0 and
                min(top_3_scores) >= threshold
            )

        # ... other validation rules

        return validation

    def generate_report(self, results: List[TestResult]):
        """Generate JSON and Markdown reports"""

        # Calculate summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.overall_pass)
        failed = total - passed
        pass_rate = (passed / total) * 100

        avg_response_time = sum(r.response_time for r in results) / total

        strategy_accuracy = sum(
            1 for r in results if r.strategy_match
        ) / total * 100

        # Save JSON report
        json_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "pass_rate_percent": pass_rate,
                "avg_response_time": avg_response_time,
                "strategy_accuracy_percent": strategy_accuracy
            },
            "test_results": [
                {
                    "test_id": r.test_id,
                    "query": r.query,
                    "status": "PASS" if r.overall_pass else "FAIL",
                    "strategy": r.selected_strategy,
                    "strategy_match": r.strategy_match,
                    "results_count": r.results_count,
                    "response_time": r.response_time,
                    "validation": r.validation_passed,
                    "error": r.error
                }
                for r in results
            ]
        }

        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"test_results_{timestamp}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)

        # Generate Markdown summary
        md_report = self.generate_markdown_summary(results, json_report)
        md_path = output_dir / f"test_summary_{timestamp}.md"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)

        print(f"\nReports saved:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        return json_report

    def generate_markdown_summary(self, results: List[TestResult], json_report: dict) -> str:
        """Generate Markdown summary report"""

        summary = json_report["summary"]

        md = f"""# Test Run Summary

**Date:** {json_report["timestamp"]}

## Overall Results

| Metric | Value |
|--------|-------|
| **Total Tests** | {summary["total_tests"]} |
| **Passed** | {summary["passed"]} ✅ |
| **Failed** | {summary["failed"]} ❌ |
| **Pass Rate** | {summary["pass_rate_percent"]:.1f}% |
| **Avg Response Time** | {summary["avg_response_time"]:.2f}s |
| **Strategy Accuracy** | {summary["strategy_accuracy_percent"]:.1f}% |

## Test Results

"""

        for r in results:
            status_icon = "✅" if r.overall_pass else "❌"
            md += f"\n### Test {r.test_id}: {status_icon}\n\n"
            md += f"**Query:** {r.query}\n\n"
            md += f"- **Strategy:** {r.selected_strategy} "
            md += f"({'✓' if r.strategy_match else '✗'})\n"
            md += f"- **Results:** {r.results_count} documents\n"
            md += f"- **Response Time:** {r.response_time:.2f}s\n"

            if r.error:
                md += f"- **Error:** {r.error}\n"

            if r.validation_passed:
                md += f"- **Validation:**\n"
                for rule, passed in r.validation_passed.items():
                    md += f"  - {rule}: {'✓' if passed else '✗'}\n"

            md += f"- **Status:** {'PASS' if r.overall_pass else 'FAIL'}\n"

        return md
```

---

## 5. Module Architecture

### 5.1 Directory Structure

```
src-sale-promotion/retrieval/
├── __init__.py                      # Package exports
├── config.py                        # Configuration management
├── loader.py                        # Embedding loader
├── fast_metadata_index.py           # Fast metadata filtering
├── index_classifier.py              # Stage 1 routing
├── strategy_selector.py             # Stage 2 routing
├── cache.py                         # Query caching
├── router.py                        # Two-stage router
├── query_engine.py                  # Production query engine
├── cli.py                           # CLI interface (NEW - Phase 4)
├── test_set.py                      # Test question definitions (NEW - Phase 4)
├── test_runner.py                   # Test execution (NEW - Phase 4)
├── README.md                        # Module documentation
├── README_CLI.md                    # CLI usage guide (NEW - Phase 4)
├── retrievers/
│   ├── __init__.py
│   ├── base.py
│   ├── vector.py
│   ├── summary.py
│   ├── recursive.py
│   ├── metadata.py
│   ├── hybrid.py
│   ├── chunk_decoupling.py
│   └── planner.py
└── test_results/                    # Test output directory (NEW - Phase 4)
    ├── test_results_20251026_143052.json
    └── test_summary_20251026_143052.md
```

### 5.2 New Files in Phase 4

#### cli.py (~300 lines)

**Purpose:** Main CLI interface for querying promotions

**Key Classes:**
- `SalesPromotionCLI` - Main CLI controller

**Key Methods:**
- `load_system()` - Initialize embeddings, indices, router
- `interactive_mode()` - Run continuous query loop
- `single_query_mode()` - Process single query
- `test_mode()` - Run test set
- `process_query()` - Core query processing
- `format_output()` - Format and display results

**Dependencies:**
- Phase 3: `router.py`, `query_engine.py`, all retrievers
- Phase 2: `loader.py` for embeddings
- Standard library: `argparse`, `sys`, `pathlib`

#### test_set.py (~250 lines)

**Purpose:** Define 10 test questions with expected outputs

**Key Classes:**
- `TestQuestion` - Single test definition dataclass

**Key Data:**
- `TEST_SET` - List of 10 TestQuestion objects

**Key Methods:**
- `get_test_set()` - Return all tests
- `get_test_by_id()` - Get specific test

**Dependencies:**
- Standard library: `dataclasses`, `typing`

#### test_runner.py (~200 lines)

**Purpose:** Automated test execution and validation

**Key Classes:**
- `TestResult` - Single test result dataclass
- `TestRunner` - Test execution controller

**Key Methods:**
- `run_all_tests()` - Execute all 10 tests
- `run_single_test()` - Execute and validate one test
- `validate_rules()` - Apply validation rules
- `generate_report()` - Create JSON + Markdown reports
- `generate_markdown_summary()` - Format Markdown output

**Dependencies:**
- `cli.py` - CLI instance for query execution
- `test_set.py` - Test definitions
- Standard library: `time`, `json`, `pathlib`

#### README_CLI.md (~300 lines)

**Purpose:** Comprehensive CLI usage documentation

**Sections:**
1. Quick Start
2. CLI Modes (Interactive, Single Query, Test)
3. Command-Line Arguments
4. Example Usage
5. Output Formats
6. Test Set Guide
7. Troubleshooting
8. Advanced Usage

### 5.3 Integration with Phase 3

**Phase 3 Components Used:**

1. **EmbeddingLoader** (`loader.py`)
   - Load Phase 2 embeddings
   - Build VectorStoreIndex
   - Prepare metadata

2. **FastMetadataIndex** (`fast_metadata_index.py`)
   - Enable sub-50ms filtering
   - Build inverted indices
   - Support metadata queries

3. **PromotionRouter** (`router.py`)
   - Two-stage routing
   - Strategy selection
   - Confidence scoring

4. **PromotionQueryEngine** (`query_engine.py`)
   - End-to-end query processing
   - GPT-5 response synthesis
   - Result formatting

5. **All 7 Retrievers** (`retrievers/`)
   - Vector, Hybrid, Metadata, Summary, Recursive, Chunk Decoupling, Planner
   - Strategy-specific retrieval
   - Metadata preservation

**Integration Flow:**

```
CLI (cli.py)
  ↓
Query Input
  ↓
PromotionQueryEngine (query_engine.py)
  ↓
PromotionRouter (router.py)
  ↓
IndexClassifier (index_classifier.py) → "sales_promotions"
  ↓
StrategySelector (strategy_selector.py) → "metadata" (confidence: 0.94)
  ↓
MetadataRetrieverAdapter (retrievers/metadata.py)
  ↓
FastMetadataIndex (fast_metadata_index.py) → Filter 71 docs → 25 docs
  ↓
VectorStoreIndex → Top-5 results
  ↓
GPT-5 Response Synthesis
  ↓
CLI Output Formatting
  ↓
Display to User
```

---

## 6. CLI Design Specifications

### 6.1 Interactive Mode Design

**User Flow:**

```
┌─────────────────────────────────────┐
│ 1. Start CLI                         │
│    python -m ... --interactive       │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 2. System Initialization             │
│    - Load embeddings                 │
│    - Build indices                   │
│    - Initialize router               │
│    [Progress indicators shown]       │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 3. Welcome Message                   │
│    === Sales Promotion Query ===     │
│    Loaded: 142 embeddings            │
│    Strategies: 7 available           │
│    Type 'exit' to quit               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 4. Query Loop                        │
│    Query: [user input]               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 5. Query Processing                  │
│    - Route query                     │
│    - Execute retrieval               │
│    - Synthesize response             │
│    [Spinner shown during wait]       │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 6. Display Results                   │
│    Strategy: metadata (0.94)         │
│    [GPT-5 response]                  │
│    [Top-N promotion list]            │
│    Response time: 8.4s               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 7. Continue or Exit?                 │
│    - Continue: Go to step 4          │
│    - Exit: Show goodbye              │
└─────────────────────────────────────┘
```

**Screen Layout (Interactive Mode):**

```
═══════════════════════════════════════════════════════════════
  Sales Promotion Query System v0.4.0
═══════════════════════════════════════════════════════════════

Loaded: 142 embeddings (71 promotions)
Strategies: vector, metadata, hybrid, summary, recursive, chunk_decoupling, planner
Cache: Enabled (100 queries, 1h TTL)

Type your question in Thai or English. Type 'exit' to quit.

───────────────────────────────────────────────────────────────
Query: โปรโมชันสำหรับเอเย่นต์

[Processing... Strategy selection → Retrieval → Synthesis]

Strategy: metadata (confidence: 0.94)
Reasoning: Query contains explicit customer type filter

───────────────────────────────────────────────────────────────
Response:

พบโปรโมชัน 25 รายการสำหรับลูกค้าประเภทเอเย่นต์:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC
   ส่วนลด: 28% | ระยะเวลา: 92 วัน | สถานะ: ใช้งานอยู่

2. PROMO_025 - น้ำดื่มช้าง PET 1500 CC
   ส่วนลด: 28% | ระยะเวลา: 92 วัน | สถานะ: ใช้งานอยู่

3. PROMO_042 - แสงโสม สปอนเซอร์
   ส่วนลด: 36.85% | ระยะเวลา: 7 วัน | สถานะ: ใช้งานอยู่

[... more results ...]

───────────────────────────────────────────────────────────────
Performance:
  Total time: 8.4s
  Retrieval: 0.03s (metadata filtering)
  Synthesis: 8.1s (GPT-5)
  Cache: MISS

───────────────────────────────────────────────────────────────
Query: exit

Goodbye! Query statistics:
  Total queries: 1
  Cache hit rate: 0%
  Avg response time: 8.4s

═══════════════════════════════════════════════════════════════
```

### 6.2 Single Query Mode Design

**Usage:**
```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันน้ำดื่มช้าง"
```

**Output:**
```
Query: โปรโมชันน้ำดื่มช้าง
Strategy: vector (confidence: 0.96)

พบโปรโมชันน้ำดื่ม 39 รายการ:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC (เอเย่นต์)
   ซื้อ 108 แพ็ค แถม 42 แพ็ค | ส่วนลด 28%

2. PROMO_008 - น้ำดื่มช้าง PET 600 CC (ซับเอเย่นต์)
   ซื้อ 1200 แพ็ค แถม 600 แพ็ค | ส่วนลด 33.3%

3. PROMO_015 - น้ำดื่มช้าง PET 600 CC (ซับเอเย่นต์)
   ซื้อ 108 แพ็ค แถม 42 แพ็ค | ส่วนลด 28%

[... 5 total shown, 34 more available ...]

Response time: 8.4s
```

**Verbose Mode:**
```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันน้ำดื่มช้าง" -v
```

**Output:**
```
═══════════════════════════════════════════════════════════════
Query Analysis
═══════════════════════════════════════════════════════════════

Query: โปรโมชันน้ำดื่มช้าง
Language: Thai
Query Length: 19 characters
Detected Keywords: ['โปรโมชัน', 'น้ำดื่ม', 'ช้าง']

───────────────────────────────────────────────────────────────
Routing Decision
───────────────────────────────────────────────────────────────

Stage 1 - Index Classification: sales_promotions
Stage 2 - Strategy Selection: vector
Confidence: 0.96

Reasoning:
- Semantic query without explicit filters
- Thai keywords: "น้ำดื่ม" (water), "ช้าง" (brand)
- Best suited for vector similarity search

───────────────────────────────────────────────────────────────
Retrieval Execution
───────────────────────────────────────────────────────────────

Strategy: VectorRetrieverAdapter
Embedding: text-embedding-3-large (3072 dims)
Top-K: 5
Similarity Metric: cosine

Retrieved Documents: 39
Top-5 Scores: [0.92, 0.91, 0.90, 0.89, 0.88]
Retrieval Time: 0.08s

Top 5 Results:
1. PROMO_001 (score: 0.92)
   - document_id: PROMO_001
   - customer_type: เอเย่นต์
   - product: น้ำดื่มช้าง PET 600 CC
   - discount: 28%

2. PROMO_008 (score: 0.91)
   - document_id: PROMO_008
   - customer_type: ซับเอเย่นต์
   - product: น้ำดื่มช้าง PET 600 CC
   - discount: 33.3%

[... continued ...]

───────────────────────────────────────────────────────────────
Response Synthesis
───────────────────────────────────────────────────────────────

Model: gpt-5
Synthesis Time: 8.1s
Response Length: 456 characters

───────────────────────────────────────────────────────────────
Final Response
───────────────────────────────────────────────────────────────

พบโปรโมชันน้ำดื่มช้าง 39 รายการ:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC (เอเย่นต์)
   ซื้อ 108 แพ็ค แถม 42 แพ็ค | ส่วนลด 28%
   ระยะเวลา: 2025-10-01 ถึง 2025-12-31 (92 วัน)
   สถานะ: ใช้งานอยู่ ✓

[... continued ...]

═══════════════════════════════════════════════════════════════
Performance Metrics
═══════════════════════════════════════════════════════════════

Total Time: 8.4s
  - Query Analysis: 0.2s
  - Routing: 1.0s
  - Retrieval: 0.08s
  - Synthesis: 8.1s

Cache: MISS
Strategy: vector
Documents Retrieved: 39
Documents Displayed: 5

═══════════════════════════════════════════════════════════════
```

### 6.3 Test Mode Design

**Usage:**
```bash
python -m src_sale_promotion.retrieval.cli --run-tests
```

**Output:**
```
═══════════════════════════════════════════════════════════════
  Running Test Set - Sales Promotion RAG Pipeline
═══════════════════════════════════════════════════════════════

Loading system...
  ✓ Embeddings loaded (142)
  ✓ Indices built
  ✓ Router initialized

Loading test set... 10 questions loaded

───────────────────────────────────────────────────────────────
Test 1/10: Customer Type Filtering
───────────────────────────────────────────────────────────────

Query: โปรโมชันสำหรับเอเย่นต์
Expected Strategy: metadata

[Processing... 8.2s]

✅ PASS

  Strategy: metadata (expected: metadata) ✓
  Results: 25 found (expected: 20-30) ✓
  Response Time: 8.2s (limit: 15s) ✓

  Validation:
    ✓ all_results_match_customer_type: True
    ✓ top_3_relevance_threshold: 0.92 > 0.90

  Top 3 Results:
    1. PROMO_001 (score: 0.92)
    2. PROMO_025 (score: 0.89)
    3. PROMO_042 (score: 0.87)

───────────────────────────────────────────────────────────────
Test 2/10: Product Category Search
───────────────────────────────────────────────────────────────

Query: โปรโมชันน้ำดื่มช้าง
Expected Strategy: vector

[Processing... 6.8s]

✅ PASS

  Strategy: vector (expected: vector) ✓
  Results: 39 found (expected: 30-45) ✓
  Response Time: 6.8s (limit: 15s) ✓

  Validation:
    ✓ top_5_all_water_category: True
    ✓ top_3_relevance_threshold: 0.96 > 0.95

  Top 3 Results:
    1. PROMO_001 (score: 0.92)
    2. PROMO_008 (score: 0.91)
    3. PROMO_015 (score: 0.90)

───────────────────────────────────────────────────────────────
[... Tests 3-9 ...]
───────────────────────────────────────────────────────────────

Test 10/10: Best Value Analysis
───────────────────────────────────────────────────────────────

Query: Which promotion offers the best value for wholesale customers?
Expected Strategy: planner

[Processing... 22.5s]

❌ FAIL

  Strategy: summary (expected: planner) ✗
  Results: 5 found (expected: 1-100) ✓
  Response Time: 22.5s (limit: 25s) ✓

  Validation:
    ✓ response_includes_analysis: True
    ✗ mentions_discount_percentage: False

  Top 3 Results:
    1. PROMO_008 (score: 0.85)
    2. PROMO_042 (score: 0.83)
    3. PROMO_001 (score: 0.80)

═══════════════════════════════════════════════════════════════
Test Summary
═══════════════════════════════════════════════════════════════

Total Tests: 10
Passed: 9 (90%) ✅
Failed: 1 (10%) ❌

Performance:
  Avg Response Time: 10.2s
  Min: 6.8s (Test 2)
  Max: 22.5s (Test 10)

Quality Metrics:
  Strategy Selection Accuracy: 90% (9/10)
  Result Count Accuracy: 100% (10/10)
  Validation Pass Rate: 95% (19/20 rules)

Failed Tests:
  - Test 10: Best Value Analysis
    Issue: Strategy selection (summary instead of planner)

Reports Generated:
  ✓ test_results/test_results_20251026_143052.json
  ✓ test_results/test_summary_20251026_143052.md

═══════════════════════════════════════════════════════════════
```

### 6.4 Progress Indicators

**Loading Indicators:**

```python
# During system initialization
print("Loading embeddings...", end="", flush=True)
# ... loading code ...
print(" ✓")

print("Building indices...", end="", flush=True)
# ... building code ...
print(" ✓")

print("Initializing router...", end="", flush=True)
# ... init code ...
print(" ✓")
```

**Query Processing Spinner:**

```python
# During query processing (interactive mode)
import itertools
import sys
import time
import threading

class Spinner:
    def __init__(self, message="Processing"):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.message = message
        self.running = False
        self.thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(f"\r{self.message}... {next(self.spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

# Usage
spinner = Spinner("Processing query")
spinner.start()
# ... process query ...
spinner.stop()
```

---

## 7. Test Framework Design

### 7.1 Test Execution Flow

```
┌─────────────────────────────────────┐
│ 1. Load Test Set                     │
│    - Read test_set.py                │
│    - Validate 10 questions           │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 2. Initialize System                 │
│    - Load embeddings                 │
│    - Build indices                   │
│    - Initialize CLI                  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 3. FOR EACH Test (1-10):             │
│                                      │
│    a. Display test info              │
│    b. Execute query                  │
│    c. Measure performance            │
│    d. Validate strategy              │
│    e. Validate result count          │
│    f. Run validation rules           │
│    g. Determine PASS/FAIL            │
│    h. Record result                  │
│                                      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 4. Calculate Summary Statistics      │
│    - Pass rate                       │
│    - Avg response time               │
│    - Strategy accuracy               │
│    - Relevance metrics               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 5. Generate Reports                  │
│    - JSON detailed results           │
│    - Markdown summary                │
│    - Console output                  │
└─────────────────────────────────────┘
```

### 7.2 Validation Rules Engine

**Rule Types:**

1. **Strategy Matching Rules**
   - Exact match: `selected_strategy == expected_strategy`
   - Alternative match: `selected_strategy in alternative_strategies`

2. **Result Count Rules**
   - Range validation: `min <= count <= max`
   - Threshold validation: `count >= threshold`

3. **Content Validation Rules**
   - All results match filter: Check metadata for all results
   - Top-N category match: Verify top results belong to expected category
   - Required documents present: Expected doc IDs in results

4. **Quality Validation Rules**
   - Top-3 relevance threshold: Scores above minimum
   - Top-1 relevance: Best result meets criteria
   - Zero results: No empty result sets

5. **Performance Rules**
   - Response time: Within timeout
   - Retrieval time: Pure retrieval latency
   - Synthesis time: GPT-5 generation time

**Rule Implementation:**

```python
class ValidationEngine:
    """Validation rules engine"""

    @staticmethod
    def validate_strategy_match(result, test):
        """Validate strategy selection"""
        selected = result['strategy']
        expected = test.expected_strategy
        alternatives = test.alternative_strategies or []

        return selected == expected or selected in alternatives

    @staticmethod
    def validate_result_count(result, test):
        """Validate result count in range"""
        count = len(result['nodes'])
        return test.expected_count_min <= count <= test.expected_count_max

    @staticmethod
    def validate_all_results_match_filter(result, test, filter_key, filter_value):
        """Validate all results match metadata filter"""
        for node in result['nodes']:
            if node.metadata.get(filter_key) != filter_value:
                return False
        return True

    @staticmethod
    def validate_top_n_category(result, test, n, expected_category):
        """Validate top N results belong to expected category"""
        top_n = result['nodes'][:n]
        for node in top_n:
            if node.metadata.get('product_category') != expected_category:
                return False
        return True

    @staticmethod
    def validate_relevance_threshold(result, test, n, threshold):
        """Validate top N relevance scores above threshold"""
        top_n_scores = [node.score for node in result['nodes'][:n]]
        if not top_n_scores:
            return False
        return min(top_n_scores) >= threshold

    @staticmethod
    def validate_required_docs_present(result, test):
        """Validate expected document IDs are in results"""
        if not test.expected_doc_ids:
            return True

        result_ids = [node.node_id for node in result['nodes']]
        for expected_id in test.expected_doc_ids:
            if expected_id not in result_ids:
                return False
        return True

    @staticmethod
    def validate_response_time(response_time, timeout):
        """Validate response within timeout"""
        return response_time <= timeout
```

### 7.3 Report Formats

#### JSON Report Format

**File:** `test_results/test_results_YYYYMMDD_HHMMSS.json`

```json
{
  "timestamp": "2025-10-26 14:30:52",
  "test_run_id": "run_20251026_143052",
  "system_info": {
    "phase": "Phase 4",
    "total_promotions": 71,
    "total_embeddings": 142,
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-5",
    "strategies_available": 7
  },
  "summary": {
    "total_tests": 10,
    "passed": 9,
    "failed": 1,
    "pass_rate_percent": 90.0,
    "avg_response_time": 10.2,
    "min_response_time": 6.8,
    "max_response_time": 22.5,
    "strategy_accuracy_percent": 90.0,
    "result_count_accuracy_percent": 100.0,
    "validation_pass_rate_percent": 95.0
  },
  "test_results": [
    {
      "test_id": 1,
      "query": "โปรโมชันสำหรับเอเย่นต์",
      "query_english": "Promotions for agents",
      "language": "Thai",
      "status": "PASS",
      "selected_strategy": "metadata",
      "expected_strategy": "metadata",
      "strategy_match": true,
      "results_count": 25,
      "expected_count_min": 20,
      "expected_count_max": 30,
      "results_count_valid": true,
      "response_time": 8.2,
      "timeout": 15,
      "response_time_valid": true,
      "validation_rules": {
        "all_results_match_customer_type": {
          "passed": true,
          "details": "All 25 results have customer_type == เอเย่นต์"
        },
        "top_3_relevance_threshold": {
          "passed": true,
          "threshold": 0.90,
          "actual_min": 0.92,
          "details": "Top-3 scores: [0.92, 0.89, 0.87]"
        }
      },
      "retrieved_doc_ids": ["PROMO_001", "PROMO_025", "PROMO_042", "..."],
      "top_3_scores": [0.92, 0.89, 0.87],
      "error": null
    },
    {
      "test_id": 2,
      "query": "โปรโมชันน้ำดื่มช้าง",
      "query_english": "Chang drinking water promotions",
      "language": "Thai",
      "status": "PASS",
      "selected_strategy": "vector",
      "expected_strategy": "vector",
      "strategy_match": true,
      "results_count": 39,
      "expected_count_min": 30,
      "expected_count_max": 45,
      "results_count_valid": true,
      "response_time": 6.8,
      "timeout": 15,
      "response_time_valid": true,
      "validation_rules": {
        "top_5_all_water_category": {
          "passed": true,
          "details": "All top-5 have product_category == water"
        },
        "top_3_relevance_threshold": {
          "passed": true,
          "threshold": 0.95,
          "actual_min": 0.96,
          "details": "Top-3 scores: [0.92, 0.91, 0.90]"
        }
      },
      "retrieved_doc_ids": ["PROMO_001", "PROMO_008", "PROMO_015", "..."],
      "top_3_scores": [0.92, 0.91, 0.90],
      "error": null
    }
  ],
  "failed_tests": [
    {
      "test_id": 10,
      "reason": "Strategy mismatch: expected 'planner', got 'summary'",
      "recommendation": "Review query complexity detection in strategy selector"
    }
  ]
}
```

#### Markdown Report Format

**File:** `test_results/test_summary_YYYYMMDD_HHMMSS.md`

```markdown
# Test Run Summary

**Date:** 2025-10-26 14:30:52
**Test Run ID:** run_20251026_143052
**Phase:** 4 - CLI Implementation & Test Set

---

## System Configuration

| Component | Value |
|-----------|-------|
| **Total Promotions** | 71 |
| **Total Embeddings** | 142 |
| **Embedding Model** | text-embedding-3-large (3072 dims) |
| **LLM Model** | gpt-5 |
| **Strategies Available** | 7 (vector, metadata, hybrid, summary, recursive, chunk_decoupling, planner) |

---

## Overall Results

| Metric | Value |
|--------|-------|
| **Total Tests** | 10 |
| **Passed** | 9 ✅ |
| **Failed** | 1 ❌ |
| **Pass Rate** | 90.0% |
| **Avg Response Time** | 10.2s |
| **Min Response Time** | 6.8s (Test 2) |
| **Max Response Time** | 22.5s (Test 10) |
| **Strategy Accuracy** | 90.0% (9/10) |
| **Result Count Accuracy** | 100.0% (10/10) |
| **Validation Pass Rate** | 95.0% (19/20 rules) |

---

## Test Results

### Test 1: Customer Type Filtering ✅

**Query:** โปรโมชันสำหรับเอเย่นต์
**Expected Strategy:** metadata

- **Strategy:** metadata (expected: metadata) ✓
- **Results:** 25 documents (expected: 20-30) ✓
- **Response Time:** 8.2s (limit: 15s) ✓
- **Validation:**
  - ✓ all_results_match_customer_type: All 25 results have customer_type == เอเย่นต์
  - ✓ top_3_relevance_threshold: 0.92 > 0.90
- **Top 3 Results:**
  1. PROMO_001 (score: 0.92)
  2. PROMO_025 (score: 0.89)
  3. PROMO_042 (score: 0.87)
- **Status:** PASS

---

### Test 2: Product Category Search ✅

**Query:** โปรโมชันน้ำดื่มช้าง
**Expected Strategy:** vector

- **Strategy:** vector (expected: vector) ✓
- **Results:** 39 documents (expected: 30-45) ✓
- **Response Time:** 6.8s (limit: 15s) ✓
- **Validation:**
  - ✓ top_5_all_water_category: All top-5 have product_category == water
  - ✓ top_3_relevance_threshold: 0.96 > 0.95
- **Top 3 Results:**
  1. PROMO_001 (score: 0.92)
  2. PROMO_008 (score: 0.91)
  3. PROMO_015 (score: 0.90)
- **Status:** PASS

---

[... Tests 3-9 ...]

---

### Test 10: Best Value Analysis ❌

**Query:** Which promotion offers the best value for wholesale customers?
**Expected Strategy:** planner

- **Strategy:** summary (expected: planner) ✗
- **Results:** 5 documents (expected: 1-100) ✓
- **Response Time:** 22.5s (limit: 25s) ✓
- **Validation:**
  - ✓ response_includes_analysis: True
  - ✗ mentions_discount_percentage: False
- **Top 3 Results:**
  1. PROMO_008 (score: 0.85)
  2. PROMO_042 (score: 0.83)
  3. PROMO_001 (score: 0.80)
- **Status:** FAIL
- **Failure Reason:** Strategy mismatch (summary instead of planner)

---

## Failed Tests Analysis

### Test 10: Best Value Analysis

**Issue:** Strategy selection error

**Expected:** planner (for complex analytical queries)
**Actual:** summary

**Root Cause:** Query complexity detection may not be identifying "best value" as requiring multi-step planning.

**Recommendation:**
- Review `StrategySelector` heuristics for analytical queries
- Add explicit patterns for "best", "compare", "analyze" keywords
- Consider lowering threshold for planner selection

---

## Performance Analysis

### Response Time Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| < 10s | 6 | 60% |
| 10-15s | 3 | 30% |
| 15-25s | 1 | 10% |
| > 25s | 0 | 0% |

**Fastest Test:** Test 2 (6.8s) - Product Category Search (vector)
**Slowest Test:** Test 10 (22.5s) - Best Value Analysis (summary)

### Strategy Usage

| Strategy | Times Selected | Expected Times | Accuracy |
|----------|----------------|----------------|----------|
| vector | 3 | 3 | 100% |
| metadata | 5 | 5 | 100% |
| hybrid | 1 | 1 | 100% |
| summary | 1 | 0 | 0% (should be planner) |
| planner | 0 | 1 | 0% (selected summary) |
| recursive | 0 | 0 | N/A |
| chunk_decoupling | 0 | 0 | N/A |

---

## Recommendations

### High Priority

1. **Fix Strategy Selection for Analytical Queries**
   - Test 10 failed due to incorrect strategy (summary vs planner)
   - Add explicit patterns for analytical keywords
   - Review LLM prompts in `StrategySelector`

### Medium Priority

2. **Optimize Response Time**
   - Current avg: 10.2s (target: < 10s)
   - GPT-5 synthesis is main bottleneck (~8-10s)
   - Consider response streaming for better UX

3. **Expand Test Coverage**
   - Add tests for recursive retriever
   - Add tests for chunk_decoupling retriever
   - Add more complex multi-criteria tests

### Low Priority

4. **Cache Warmup**
   - All tests show CACHE: MISS
   - Consider pre-warming cache with common queries
   - Monitor cache hit rate after deployment

---

## Conclusion

**Overall Assessment:** ✅ **PASS (90%)**

The Phase 4 CLI implementation successfully meets quality targets:
- ✅ 90% test pass rate (target: ≥ 80%)
- ✅ 90% strategy accuracy (target: ≥ 80%)
- ⚠️ 10.2s avg response time (target: < 10s, close)

**Ready for Production:** Yes, with minor improvements to planner strategy selection.

---

**Report Generated:** 2025-10-26 14:30:52
**Next Review:** After strategy selector improvements
```

---

## 8. Performance Targets

### 8.1 Response Time Targets

| Component | Target | Acceptable | Measurement |
|-----------|--------|------------|-------------|
| **Total Query Time** | < 10s | < 20s | Query input → response display |
| **System Initialization** | < 5s | < 10s | CLI start → ready for queries |
| **Query Routing** | < 2s | < 3s | Two-stage routing decision |
| **Pure Retrieval** | < 500ms | < 1s | Retrieval without synthesis |
| **GPT-5 Synthesis** | < 10s | < 20s | Response generation |
| **Output Formatting** | < 100ms | < 500ms | Result formatting + display |

**Breakdown Example:**

```
Total Query Time: 8.4s
  ├─ Input Processing: 0.05s
  ├─ Query Routing: 1.2s
  │   ├─ Index Classification: 0.3s
  │   └─ Strategy Selection: 0.9s (GPT-5 call)
  ├─ Retrieval Execution: 0.03s
  │   ├─ Metadata Filtering: 0.02s (fast index)
  │   └─ Vector Search: 0.01s
  ├─ Response Synthesis: 8.1s (GPT-5)
  └─ Output Formatting: 0.02s
```

### 8.2 Quality Targets

#### Retrieval Quality

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Strategy Selection Accuracy** | ≥ 90% | ≥ 80% | Correct strategy / total queries |
| **Top-1 Relevance** | ≥ 80% | ≥ 70% | Relevant result in position 1 |
| **Top-3 Relevance** | ≥ 90% | ≥ 85% | Relevant result in top-3 |
| **Top-5 Relevance** | ≥ 95% | ≥ 90% | Relevant result in top-5 |
| **Zero Results Rate** | < 2% | < 5% | Queries with no results |

#### Test Set Quality

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Test Pass Rate** | ≥ 90% | ≥ 80% | Passed tests / total tests |
| **Result Count Accuracy** | 100% | ≥ 90% | Results in expected range |
| **Validation Rule Pass** | ≥ 95% | ≥ 85% | Rules passed / total rules |

### 8.3 System Performance Targets

#### Scalability

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| **Max Concurrent Users** | 10 | 5 | CLI instances |
| **Queries per Minute** | 60 | 30 | With caching |
| **Cache Hit Rate** | > 40% | > 30% | After warmup |
| **Memory Usage** | < 2 GB | < 4 GB | Per CLI instance |

#### Reliability

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| **Uptime** | 99.9% | 99% | System availability |
| **Error Rate** | < 1% | < 2% | Failed queries |
| **Recovery Time** | < 5s | < 10s | After transient error |

### 8.4 User Experience Targets

#### Usability

| Metric | Target | Method |
|--------|--------|--------|
| **Time to First Query** | < 30s | From CLI start to first result |
| **Learning Curve** | < 5 min | Time to understand basic usage |
| **Error Message Clarity** | 100% actionable | User can fix issue from message |

#### Output Quality

| Metric | Target | Method |
|--------|--------|--------|
| **Thai Text Rendering** | 100% correct | Visual inspection + encoding check |
| **Output Readability** | High | Subjective user feedback |
| **Information Completeness** | 100% | All relevant details shown |

---

## 9. Phase Dependencies

### 9.1 Prerequisites

**Phase 1 (Data Preparation) - COMPLETE ✅**

| Component | Status | Output |
|-----------|--------|--------|
| CSV Parsing | ✅ Done | 71 promotions parsed |
| Markdown Generation | ✅ Done | 71 markdown files |
| JSONL Conversion | ✅ Done | `documents.jsonl` (71 docs) |
| Metadata Extraction | ✅ Done | 25 metadata fields |

**Phase 2 (Embedding & Indexing) - COMPLETE ✅**

| Component | Status | Output |
|-----------|--------|--------|
| Document Loading | ✅ Done | 71 LlamaIndex Documents |
| GPT-5 Summaries | ✅ Done | 71 summaries generated |
| Embeddings | ✅ Done | 142 embeddings (3072 dims) |
| Multi-Format Output | ✅ Done | JSON, PKL, NPY files |

**Phase 3 (Retrieval Strategies) - COMPLETE ✅**

| Component | Status | Output |
|-----------|--------|--------|
| 7 Retrieval Strategies | ✅ Done | All strategies operational |
| Two-Stage Router | ✅ Done | Index + strategy routing |
| Fast Metadata Index | ✅ Done | Sub-50ms filtering |
| Query Caching | ✅ Done | LRU cache with TTL |
| GPT-5 Integration | ✅ Done | Strategy selection + synthesis |

### 9.2 Phase 4 Dependencies

**Phase 4 Components:**

```
cli.py
  ├─ Depends on: Phase 3 router, query_engine, all retrievers
  ├─ Depends on: Phase 2 loader, embeddings
  └─ Depends on: Phase 1 documents.jsonl

test_set.py
  ├─ Depends on: Phase 1 promotion data (for test design)
  └─ Depends on: Phase 3 strategy definitions

test_runner.py
  ├─ Depends on: cli.py
  ├─ Depends on: test_set.py
  └─ Depends on: Phase 3 query_engine
```

**File Dependencies:**

```
Phase 4 NEW FILES:
├── cli.py
│   ├── imports: router.py (Phase 3)
│   ├── imports: query_engine.py (Phase 3)
│   ├── imports: loader.py (Phase 3)
│   └── imports: all retrievers (Phase 3)
│
├── test_set.py
│   └── No dependencies (pure data)
│
├── test_runner.py
│   ├── imports: cli.py (Phase 4)
│   ├── imports: test_set.py (Phase 4)
│   └── imports: query_engine.py (Phase 3)
│
└── README_CLI.md
    └── No dependencies (documentation)
```

### 9.3 External Dependencies

**Python Packages:**

```python
# Core dependencies (from Phase 3)
llama-index-core==0.10.0
llama-index-llms-openai==0.1.0
llama-index-embeddings-openai==0.1.0

# Additional for CLI (new)
# (No new packages needed - using stdlib)
argparse  # stdlib
sys       # stdlib
pathlib   # stdlib
time      # stdlib
json      # stdlib
threading # stdlib (for spinner)
```

**Environment Variables:**

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (Phase 3 defaults)
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-5
CHUNK_SIZE=2048
BATCH_SIZE=20
```

---

## 10. Implementation Plan

### 10.1 Timeline Overview

**Total Duration:** 1 week (5 working days)

**Milestones:**

| Day | Milestone | Deliverables |
|-----|-----------|--------------|
| 1 | CLI Implementation | `cli.py` complete, basic modes working |
| 2 | Test Set Creation | `test_set.py` with all 10 questions |
| 3 | Test Runner Implementation | `test_runner.py`, automated execution |
| 4 | Testing & Refinement | Run tests, fix issues, optimize |
| 5 | Documentation & Handoff | README_CLI.md, implementation summary |

### 10.2 Day-by-Day Plan

#### Day 1: CLI Implementation

**Tasks:**

1. **Setup Module Structure** (1 hour)
   - Create `cli.py` file
   - Define `SalesPromotionCLI` class
   - Import Phase 3 dependencies

2. **Implement Argument Parsing** (1 hour)
   - Define all CLI arguments
   - Create `parse_arguments()` function
   - Add validation logic

3. **Implement System Initialization** (2 hours)
   - `load_system()` method
   - Load embeddings from Phase 2
   - Build fast metadata index
   - Initialize router and query engine
   - Add progress indicators

4. **Implement Interactive Mode** (2 hours)
   - `interactive_mode()` method
   - Query input loop
   - Process queries
   - Display results
   - Handle exit

5. **Implement Single Query Mode** (1 hour)
   - `single_query_mode()` method
   - Process single query
   - Format output
   - Exit cleanly

6. **Implement Output Formatting** (1 hour)
   - `format_output()` method
   - Basic format (clean)
   - Verbose format (detailed)
   - Debug format (full)
   - Thai UTF-8 handling

**Acceptance Criteria:**
- ✅ CLI starts and loads system
- ✅ Interactive mode works
- ✅ Single query mode works
- ✅ Output formatted correctly
- ✅ Thai text displays correctly

---

#### Day 2: Test Set Creation

**Tasks:**

1. **Analyze Promotion Data** (1 hour)
   - Review 71 promotion markdown files
   - Identify query patterns
   - Determine expected strategies

2. **Design 10 Test Questions** (2 hours)
   - Define Test 1-5 (basic queries)
   - Define Test 6-10 (complex queries)
   - Specify expected strategies
   - Define validation rules

3. **Implement Test Set Module** (2 hours)
   - Create `test_set.py`
   - Define `TestQuestion` dataclass
   - Create `TEST_SET` list
   - Implement helper functions

4. **Validate Test Definitions** (1 hour)
   - Review all 10 tests
   - Ensure diversity
   - Check expected values
   - Verify validation rules

5. **Manual Test Execution** (2 hours)
   - Run each test manually via CLI
   - Verify expected strategies
   - Check result counts
   - Document actual results

**Acceptance Criteria:**
- ✅ 10 diverse test questions defined
- ✅ All tests cover different strategies
- ✅ Expected values are realistic
- ✅ Validation rules are implementable
- ✅ Manual tests show >80% pass rate

---

#### Day 3: Test Runner Implementation

**Tasks:**

1. **Implement TestResult Dataclass** (1 hour)
   - Define `TestResult` structure
   - Add all required fields
   - Implement serialization

2. **Implement TestRunner Class** (3 hours)
   - Create `TestRunner` class
   - `run_all_tests()` method
   - `run_single_test()` method
   - `validate_rules()` method
   - Progress display

3. **Implement Report Generation** (2 hours)
   - `generate_report()` method
   - JSON output format
   - Markdown output format
   - Summary statistics

4. **Implement Validation Rules** (2 hours)
   - Strategy matching
   - Result count validation
   - Content validation (filters, categories)
   - Quality validation (relevance scores)
   - Performance validation (timing)

**Acceptance Criteria:**
- ✅ Test runner executes all 10 tests
- ✅ Validation rules work correctly
- ✅ JSON report generated
- ✅ Markdown summary generated
- ✅ Console output is clear

---

#### Day 4: Testing & Refinement

**Tasks:**

1. **Run Complete Test Suite** (1 hour)
   - Execute `--run-tests` mode
   - Review all test results
   - Identify failures

2. **Fix Test Failures** (3 hours)
   - Adjust expected values if needed
   - Fix validation rule bugs
   - Improve strategy selection (if needed)
   - Re-run tests

3. **Performance Optimization** (2 hours)
   - Measure response times
   - Identify bottlenecks
   - Optimize if needed
   - Cache warmup

4. **User Experience Refinement** (2 hours)
   - Improve output formatting
   - Better error messages
   - Progress indicators
   - Help text

**Acceptance Criteria:**
- ✅ Test pass rate ≥ 80%
- ✅ Average response time < 20s
- ✅ Strategy accuracy ≥ 80%
- ✅ CLI is user-friendly
- ✅ All error cases handled

---

#### Day 5: Documentation & Handoff

**Tasks:**

1. **Create CLI README** (2 hours)
   - Write `README_CLI.md`
   - Quick start guide
   - All CLI modes explained
   - Example usage
   - Troubleshooting

2. **Create Implementation Summary** (2 hours)
   - Write `PHASE4_IMPLEMENTATION_SUMMARY.md`
   - Document what was built
   - Test results summary
   - Lessons learned
   - Next steps

3. **Code Review & Cleanup** (2 hours)
   - Review all new code
   - Add docstrings
   - Clean up comments
   - Format code

4. **Final Testing** (1 hour)
   - Run test suite one more time
   - Verify all features work
   - Check documentation accuracy

5. **Git Commit & PR** (1 hour)
   - Commit all changes
   - Create pull request
   - Add PR description
   - Tag as v0.4.0

**Acceptance Criteria:**
- ✅ README_CLI.md complete
- ✅ Implementation summary complete
- ✅ All code reviewed and clean
- ✅ Final test suite passes
- ✅ PR created and ready for review

### 10.3 Risk Management

**Risks & Mitigations:**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Test pass rate < 80%** | Medium | High | Build in buffer for fixing tests on Day 4 |
| **GPT-5 API rate limits** | Low | Medium | Implement retry logic, delays between tests |
| **Thai text rendering issues** | Low | Medium | Test on multiple terminals early |
| **Response time > 20s** | Medium | Medium | Profile early, optimize or adjust targets |
| **Complex validation rules** | Medium | Low | Start simple, add complexity incrementally |

---

## 11. Example Usage

### 11.1 Quick Start

```bash
# Navigate to retrieval directory
cd src-sale-promotion/retrieval

# Ensure Phase 3 is complete
ls -la  # Should see router.py, query_engine.py, etc.

# Install dependencies (if needed)
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai

# Set OpenAI API key
export OPENAI_API_KEY=sk-...

# Run CLI in interactive mode
python -m src_sale_promotion.retrieval.cli --interactive
```

### 11.2 Example Queries

**Example 1: Simple Customer Filter**

```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันสำหรับเอเย่นต์"
```

**Output:**
```
Query: โปรโมชันสำหรับเอเย่นต์
Strategy: metadata (confidence: 0.94)

พบโปรโมชัน 25 รายการสำหรับลูกค้าประเภทเอเย่นต์:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC
   ส่วนลด: 28% | ระยะเวลา: 92 วัน | สถานะ: ใช้งานอยู่

2. PROMO_025 - น้ำดื่มช้าง PET 1500 CC
   ส่วนลด: 28% | ระยะเวลา: 92 วัน | สถานะ: ใช้งานอยู่

3. PROMO_042 - แสงโสม สปอนเซอร์
   ส่วนลด: 36.85% | ระยะเวลา: 7 วัน | สถานะ: ใช้งานอยู่

[... 22 more promotions ...]

Response time: 8.4s
```

---

**Example 2: Product Search with Verbose**

```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันน้ำดื่มช้าง" -v
```

**Output:**
```
═══════════════════════════════════════════════════════════════
Query Analysis
═══════════════════════════════════════════════════════════════

Query: โปรโมชันน้ำดื่มช้าง
Language: Thai
Detected Keywords: ['โปรโมชัน', 'น้ำดื่ม', 'ช้าง']

───────────────────────────────────────────────────────────────
Routing Decision
───────────────────────────────────────────────────────────────

Stage 1: sales_promotions
Stage 2: vector (confidence: 0.96)

Reasoning: Semantic query for product category + brand

───────────────────────────────────────────────────────────────
Retrieval Results
───────────────────────────────────────────────────────────────

Retrieved: 39 documents
Top-5 Scores: [0.92, 0.91, 0.90, 0.89, 0.88]
Retrieval Time: 0.08s

───────────────────────────────────────────────────────────────
Final Response
───────────────────────────────────────────────────────────────

พบโปรโมชันน้ำดื่มช้าง 39 รายการ:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC (เอเย่นต์)
   ซื้อ 108 แพ็ค แถม 42 แพ็ค | ส่วนลด 28%

2. PROMO_008 - น้ำดื่มช้าง PET 600 CC (ซับเอเย่นต์)
   ซื้อ 1200 แพ็ค แถม 600 แพ็ค | ส่วนลด 33.3%

[... more results ...]

═══════════════════════════════════════════════════════════════
Performance
═══════════════════════════════════════════════════════════════

Total: 8.4s (Routing: 1.0s | Retrieval: 0.08s | Synthesis: 8.1s)
Cache: MISS
═══════════════════════════════════════════════════════════════
```

---

**Example 3: Complex Multi-Criteria Query**

```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์สำหรับซับเอเย่นต์ที่มีส่วนลดมากกว่า 10%"
```

**Output:**
```
Query: โปรโมชันเบียร์สำหรับซับเอเย่นต์ที่มีส่วนลดมากกว่า 10%
Strategy: metadata (confidence: 0.91)

Auto-detected filters:
  - customer_type: ซับเอเย่นต์
  - product_category: beer
  - discount_percentage: > 10%

พบโปรโมชัน 8 รายการที่ตรงตามเงื่อนไข:

1. PROMO_XXX - เบียร์ช้าง 330 ML (ซับเอเย่นต์)
   ส่วนลด: 15.2% | สถานะ: ใช้งานอยู่

[... more results ...]

Response time: 9.1s
```

---

**Example 4: Interactive Session**

```bash
python -m src_sale_promotion.retrieval.cli -i
```

**Output:**
```
═══════════════════════════════════════════════════════════════
  Sales Promotion Query System v0.4.0
═══════════════════════════════════════════════════════════════

Loaded: 142 embeddings (71 promotions)
Type 'exit' to quit

───────────────────────────────────────────────────────────────
Query: โปรโมชันที่หมดอายุแล้ว

Strategy: metadata (confidence: 0.95)

พบโปรโมชันที่หมดอายุแล้ว 21 รายการ:

1. PROMO_055 - เบียร์ทุก SKU (เอเย่นต์)
   หมดอายุ: 2025-10-14 (หมดอายุแล้ว)

2. PROMO_060 - เบียร์ทุก SKU (Super Sub)
   หมดอายุ: 2025-10-08 (หมดอายุแล้ว)

[... more results ...]

Response time: 8.7s

───────────────────────────────────────────────────────────────
Query: โปรโมชันที่กำลังใช้งานอยู่

Strategy: metadata (confidence: 0.96)

พบโปรโมชันที่กำลังใช้งานอยู่ 50 รายการ:

1. PROMO_001 - น้ำดื่มช้าง PET 600 CC (เอเย่นต์)
   สิ้นสุด: 2025-12-31 (อีก 66 วัน)

[... more results ...]

Response time: 7.2s (Cache: MISS)

───────────────────────────────────────────────────────────────
Query: exit

Goodbye! Session summary:
  Queries: 2
  Avg time: 7.95s
  Cache hits: 0

═══════════════════════════════════════════════════════════════
```

---

**Example 5: Run Test Suite**

```bash
python -m src_sale_promotion.retrieval.cli --run-tests
```

**Output:**
```
═══════════════════════════════════════════════════════════════
  Running Test Set - Sales Promotion RAG Pipeline
═══════════════════════════════════════════════════════════════

Loading system... ✓

Running 10 tests...

───────────────────────────────────────────────────────────────
Test 1/10: Customer Type Filtering

Query: โปรโมชันสำหรับเอเย่นต์
[Processing... 8.2s]

✅ PASS
  Strategy: metadata ✓
  Results: 25 (expected: 20-30) ✓
  Time: 8.2s ✓
  Validation: all passed ✓

───────────────────────────────────────────────────────────────
[... Tests 2-9 ...]
───────────────────────────────────────────────────────────────

Test 10/10: Best Value Analysis

Query: Which promotion offers the best value for wholesale customers?
[Processing... 22.5s]

❌ FAIL
  Strategy: summary (expected: planner) ✗
  Results: 5 ✓
  Time: 22.5s ✓

═══════════════════════════════════════════════════════════════
Test Summary
═══════════════════════════════════════════════════════════════

Total: 10
Passed: 9 (90%)
Failed: 1 (10%)

Avg Response Time: 10.2s
Strategy Accuracy: 90%

Reports saved:
  test_results/test_results_20251026_143052.json
  test_results/test_summary_20251026_143052.md

═══════════════════════════════════════════════════════════════
```

### 11.3 Advanced Usage

**Force Specific Strategy:**

```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" -s hybrid
```

**Skip GPT-5 Synthesis:**

```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" --no-synthesis
```

**Custom Top-K:**

```bash
python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" -k 10
```

**Load Specific Embeddings:**

```bash
python -m src_sale_promotion.retrieval.cli -i -l ../data/embedding/embeddings_sales_20251026
```

---

## 12. References & Appendix

### 12.1 Related Documents

**Phase PRDs:**

1. **Phase 1 PRD:** `src-sale-promotion/prd/01_PRD_PHASE1_DATA_PREP_v2.md`
   - Data preparation specifications
   - CSV parsing, markdown generation
   - JSONL output format

2. **Phase 2 PRD:** `src-sale-promotion/prd/02_PRD_PHASE2_EMBEDDING_INDEXING.md`
   - Embedding generation specifications
   - DocumentSummaryIndex pattern
   - GPT-5 + text-embedding-3-large

3. **Phase 3 PRD:** `src-sale-promotion/prd/03_PRD_PHASE3_RETRIEVAL_STRATEGIES.md`
   - 7 retrieval strategies
   - Two-stage routing architecture
   - Fast metadata indexing

**Implementation Summaries:**

1. **Phase 2 Summary:** `src-sale-promotion/prd/PHASE2_IMPLEMENTATION_SUMMARY.md`
2. **Phase 3 Summary:** `src-sale-promotion/prd/PHASE3_IMPLEMENTATION_SUMMARY.md`

**Module Documentation:**

1. **Phase 2 README:** `src-sale-promotion/docs_embedding/README.md`
2. **Phase 3 README:** `src-sale-promotion/retrieval/README.md`

### 12.2 Key Phase 3 Files Used

**Core Modules:**

```
src-sale-promotion/retrieval/
├── router.py                        # Two-stage router
├── query_engine.py                  # Main query engine
├── loader.py                        # Embedding loader
├── fast_metadata_index.py           # Fast metadata filtering
├── index_classifier.py              # Stage 1 routing
├── strategy_selector.py             # Stage 2 routing
└── retrievers/                      # 7 retrieval strategies
    ├── vector.py
    ├── metadata.py
    ├── hybrid.py
    ├── summary.py
    ├── recursive.py
    ├── chunk_decoupling.py
    └── planner.py
```

### 12.3 Data Files

**Phase 1 Output:**
- `src-sale-promotion/data/processed/documents/documents.jsonl` (71 promotions)
- `src-sale-promotion/data/processed/documents/markdown_files/*.md` (71 files)

**Phase 2 Output:**
- `src-sale-promotion/data/embedding/embeddings_sales_20251026_172927/`
  - 142 embeddings (71 IndexNode + 71 Summary)
  - Multi-format: JSON, PKL, NPY

### 12.4 External Resources

**LlamaIndex Documentation:**
- https://docs.llamaindex.ai/
- https://docs.llamaindex.ai/en/stable/module_guides/querying/

**OpenAI API:**
- https://platform.openai.com/docs/models/gpt-5
- https://platform.openai.com/docs/guides/embeddings

**Python CLI Best Practices:**
- https://docs.python.org/3/library/argparse.html
- https://click.palletsprojects.com/ (not used, but reference)

### 12.5 Glossary

**Terms:**

- **CLI:** Command-Line Interface
- **RAG:** Retrieval-Augmented Generation
- **PRD:** Product Requirements Document
- **LLM:** Large Language Model
- **GPT-5:** OpenAI's latest language model
- **Embedding:** Vector representation of text (3072 dimensions)
- **IndexNode:** Document-level navigation node in LlamaIndex
- **DocumentSummaryIndex:** Hierarchical index with LLM summaries
- **Two-Stage Routing:** Index classification → Strategy selection
- **Fast Metadata Index:** In-memory inverted/B-tree indices for sub-50ms filtering

**Thai Terms:**

- **เอเย่นต์ (Agent):** Direct wholesale customer
- **ซับเอเย่นต์ (Sub-agent):** Secondary wholesale distributor
- **รถขายปลีก (Retail Truck):** Mobile retail seller
- **Super Sub:** Large-volume sub-distributor
- **น้ำดื่ม (Water):** Drinking water category
- **เบียร์ (Beer):** Beer category
- **ช้าง (Chang):** Chang brand (ThaiBev's flagship brand)

### 12.6 Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-26 | Initial PRD creation |

---

**End of Phase 4 PRD**

**Document Version:** 1.0
**Last Updated:** 2025-10-26
**Status:** Draft - Ready for Review
**Next Phase:** Phase 4 Implementation (1 week)

---

**Approval Signatures:**

- [ ] Product Owner: _________________
- [ ] Technical Lead: _________________
- [ ] QA Lead: _________________

**Approved Date:** _________________
