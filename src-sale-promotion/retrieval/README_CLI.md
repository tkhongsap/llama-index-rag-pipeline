# Sales Promotion CLI - User Guide

Simple command-line interface for querying Thai sales promotion documents.

## Quick Start

```bash
cd src-sale-promotion/retrieval

# Interactive mode
python3 cli.py --interactive

# Single query
python3 cli.py --query "โปรโมชันเบียร์"

# Run test suite
python3 cli.py --run-tests
```

---

## Installation

### Prerequisites

1. **Phase 1-3 Complete:**
   - ✅ 71 promotions processed
   - ✅ 142 embeddings generated (3072 dims)
   - ✅ 7 retrieval strategies implemented

2. **Environment Setup:**
   ```bash
   # Virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure API key
   echo "OPENAI_API_KEY=sk-your-key" > .env
   ```

---

## CLI Modes

### 1. Interactive Mode

Continuous query loop for multiple questions.

```bash
python3 cli.py --interactive
```

**Features:**
- Type questions continuously
- Session statistics on exit
- Cache benefits across queries
- Type 'exit' or 'quit' to stop

**Example Session:**
```
======================================================================
  Sales Promotion Query System v4.0
======================================================================

Initializing system...
  [1/5] Loading embeddings... ✓
  [2/5] Validating embeddings... ✓
  [3/5] Building vector index... ✓
  [4/5] Building metadata index... ✓
  [5/5] Initializing query engine... ✓

✅ System ready!
   - Embeddings: 142 (71 promotions)
   - Strategies: 7 available
   - Models: GPT-4o-mini + text-embedding-3-large
======================================================================

Interactive Mode
Type your question in Thai or English. Type 'exit' or 'quit' to stop.

Query: โปรโมชันน้ำดื่มช้าง

Strategy: hybrid (confidence: 0.86)

[Thai response with promotions...]

Promotions Found: 5
Response Time: 13.3s

----------------------------------------------------------------------

Query: exit

======================================================================
Session Summary:
  Queries: 1
  Session time: 45.2s
  Cache hit rate: 0.0%
======================================================================

👋 Goodbye!
```

---

### 2. Single Query Mode

Run one query and exit.

```bash
python3 cli.py --query "โปรโมชันเบียร์"
```

**Use Cases:**
- Quick one-off queries
- Scripting/automation
- Testing specific queries

**Example:**
```
Query: โปรโมชันเบียร์
Strategy: vector (confidence: 0.95)

[Response...]

Promotions Found: 3
Response Time: 12.4s
```

---

### 3. Test Mode

Run automated 10-question test suite.

```bash
python3 cli.py --run-tests
```

**Features:**
- 10 comprehensive test questions
- Automated validation
- JSON + Markdown reports
- Performance benchmarking

**Example Output:**
```
======================================================================
  Running Automated Test Suite
======================================================================

Running 10 Tests
======================================================================

Test 1/10: Promotions for agents
Query: โปรโมชันสำหรับเอเย่นต์

✅ PASS
  Strategy: metadata (expected: metadata) ✓
  Results: 5 documents
  Time: 21.4s

[... Tests 2-10 ...]

======================================================================
Reports Generated:
  JSON: test_results/test_results_20251026_173245.json
  Markdown: test_results/test_summary_20251026_173245.md
======================================================================

======================================================================
Test Results: 8/10 passed (80%)
======================================================================
```

---

## Command-Line Arguments

### Modes

| Argument | Short | Description |
|----------|-------|-------------|
| `--interactive` | `-i` | Start interactive mode |
| `--query TEXT` | `-q` | Run single query |
| `--run-tests` | `-t` | Run test suite |

### Options

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--verbose` | `-v` | flag | False | Show detailed metrics |
| `--debug` | `-d` | flag | False | Show debug info |
| `--top-k` | `-k` | int | 5 | Number of results |
| `--strategy` | `-s` | string | auto | Force strategy |
| `--no-synthesis` | | flag | False | Skip GPT response |
| `--no-stream` | | flag | False | Disable streaming |
| `--cache` | | flag | True | Enable caching |

---

## Usage Examples

### Example 1: Simple Query
```bash
python3 cli.py -q "โปรโมชันน้ำดื่มช้าง"
```

### Example 2: Verbose Output
```bash
python3 cli.py -q "โปรโมชันน้ำดื่มช้าง" -v
```

**Output includes:**
- Routing decision details
- Performance breakdown
- Top-3 relevance scores
- Cache status

### Example 3: Force Strategy
```bash
python3 cli.py -q "โปรโมชันเบียร์" -s metadata
```

Forces metadata strategy instead of automatic selection.

### Example 4: More Results
```bash
python3 cli.py -q "โปรโมชันน้ำดื่มช้าง" -k 10
```

Returns top-10 results instead of default 5.

### Example 5: Fast Mode (No GPT Synthesis)
```bash
python3 cli.py -q "โปรโมชันเบียร์" --no-synthesis
```

Shows raw results without GPT synthesis (sub-second response).

### Example 6: Disable Streaming
```bash
python3 cli.py -q "โปรโมชันเบียร์" --no-stream
```

Wait for complete response instead of streaming tokens.

---

## Output Formats

### Basic Output
```
Query: [query]
Strategy: [strategy] (confidence: [0.XX])

[GPT-4o-mini streaming response]

Promotions Found: [count]
Response Time: [X.X]s
```

### Verbose Output
```
Query: [query]
Strategy: [strategy] (confidence: [0.XX])

[GPT-4o-mini streaming response]

Promotions Found: [count]

Performance Metrics:
  Total Time: [X.X]s
  Retrieval: [X.XXX]s
  Synthesis: [X.X]s
  Cache: [HIT/MISS]
  Top-3 Scores: ['0.XXX', '0.XXX', '0.XXX']
```

---

## Test Set Guide

### Running Tests
```bash
python3 cli.py --run-tests
```

### Test Coverage

The 10-question test set covers:
- ✅ All 7 retrieval strategies
- ✅ Thai and English queries
- ✅ Customer type filtering
- ✅ Product category search
- ✅ Discount range queries
- ✅ Active/expired status
- ✅ Multi-criteria queries
- ✅ Complex analytical queries

### Test Reports

**JSON Report:** `test_results/test_results_YYYYMMDD_HHMMSS.json`
- Complete test results
- Performance metrics
- Validation details

**Markdown Summary:** `test_results/test_summary_YYYYMMDD_HHMMSS.md`
- Human-readable summary
- Pass/fail status
- Recommendations

---

## Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution:**
```bash
# Add to .env file
echo "OPENAI_API_KEY=sk-your-key" >> .env
```

### Issue: "Embeddings not found"
**Solution:**
```bash
# Run Phase 2 first
cd ../docs_embedding
python3 batch_embedding.py
```

### Issue: "Thai text not displaying"
**Solution:**
```bash
# Ensure UTF-8 locale
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### Issue: "Slow response time"
**Phase 5 Improvements:**
- Response streaming enabled by default (perceived latency <1s)
- GPT-4o-mini model (10x faster than GPT-5)
- Query caching with warm-up

**For ultra-fast mode:**
```bash
# Skip GPT synthesis entirely (sub-second)
python3 cli.py -q "query" --no-synthesis
```

---

## Performance

**Phase 5 Performance (GPT-4o-mini + Streaming):**
- Pure retrieval: ~100-500ms
- With GPT-4o-mini synthesis: ~8-10s (actual)
- Perceived latency: <1s (streaming)
- Cached queries: ~50% faster
- Cost per query: $0.004 (73% reduction)

**Previous Performance (GPT-5):**
- With GPT-5 synthesis: ~25-28s
- Cost per query: $0.015

**Strategies by Speed:**
1. Metadata: ~30ms (fastest)
2. Vector: ~80ms
3. Hybrid: ~120ms
4. Summary: ~180ms
5. Recursive: ~250ms
6. Planner: ~400ms

---

## Advanced Usage

### Python API

```python
from retrieval.cli import SalesPromotionCLI

cli = SalesPromotionCLI(verbose=True)
cli.load_system()
cli.process_query("โปรโมชันเบียร์", top_k=10)
```

### Custom Configuration

```python
# Modify config before loading
from retrieval import config
config.CACHE_CONFIG["cache_max_size"] = 200

cli = SalesPromotionCLI()
cli.load_system()
```

---

**Document Version:** 2.0
**Last Updated:** 2025-10-26
**Status:** Production-Ready (Phase 5 Performance Optimized)

