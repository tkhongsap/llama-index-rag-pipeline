# iLand Retrieval Strategy Testing Framework

## Overview

This document describes the comprehensive testing framework for the 7 retrieval strategies in the iLand system. The framework tests how queries are routed to different strategies and validates the quality of results.

## 7 Retrieval Strategies Tested

1. **Vector**: Semantic similarity search using embeddings
2. **Hybrid**: Combines semantic search with Thai keyword matching  
3. **Metadata**: Filters based on Thai geographic and legal metadata
4. **Planner**: Multi-step query planning and execution
5. **Recursive**: Hierarchical retrieval from summaries to details
6. **Chunk Decoupling**: Separates chunk retrieval from context synthesis
7. **Summary**: Retrieves from document summaries first

## Test Framework Components

### 1. Test Cases (`tests/test_iland_retrieval_strategies.py`)

**10 carefully designed test queries** that cover all strategies:

| Query Type | Example | Expected Strategy | Purpose |
|------------|---------|------------------|---------|
| Simple Semantic | "โฉนดที่ดินในจังหวัดชัยนาท" | vector | Basic semantic search |
| Thai Keywords | "หาโฉนด นส.3 กรรมสิทธิ์บริษัท" | hybrid | Legal term matching |
| Geographic | "ที่ดินในจังหวัดอ่างทอง อำเภอเมืองอ่างทอง" | metadata | Location filtering |
| Multi-step | "วิเคราะห์ขั้นตอนการโอนกรรมสิทธิ์ที่ดิน จากนั้นแสดงเอกสาร..." | planner | Complex analysis |
| Hierarchical | "รายละเอียดโครงสร้างการถือครองที่ดิน..." | recursive | Multi-level exploration |
| Detailed Analysis | "รายละเอียดเฉพาะส่วนของการวัดพื้นที่..." | chunk_decoupling | Specific sections |
| Overview | "สรุปภาพรวมของข้อมูลโฉนดที่ดินในระบบ" | summary | High-level information |
| English Semantic | "find land deeds with company ownership" | vector | English queries |
| Mixed Language | "โฉนด land deed coordinates GPS location" | hybrid | Thai-English mix |
| Complex Metadata | "ที่ดินในภาคกลาง ประเภทกรรมสิทธิ์บริษัท ไม่ใช่คอนโด" | metadata | Multi-attribute filtering |

### 2. Mock Data

- **Realistic Thai land deed content** based on example documents
- **Province data**: Chai Nat, Ang Thong
- **Legal terms**: โฉนด, นส.3, กรรมสิทธิ์บริษัท
- **Geographic info**: จังหวัด, อำเภอ, ภาคกลาง
- **Area measurements**: ไร่, งาน, ตารางวา

### 3. Strategy Selectors Tested

1. **LLM Selector** - Uses GPT-4o-mini for intelligent routing
2. **Heuristic Selector** - Rule-based routing using keyword patterns
3. **Round-Robin Selector** - Cycles through strategies for testing

## Running Tests

### Via pytest
```bash
# Run all strategy tests
python -m pytest tests/test_iland_retrieval_strategies.py -v

# Run specific test
python -m pytest tests/test_iland_retrieval_strategies.py::test_all_retrieval_strategies_llm_selector -v
```

### Via direct execution
```bash
# Run comprehensive tests for all selectors
python tests/test_iland_retrieval_strategies.py
```

### Via CLI integration
```bash
# Run strategy tests via CLI
python src-iLand/retrieval/cli.py --test-retrieval-strategies --strategy-selector llm
```

### Quick demo
```bash
# Run quick demo with sample queries
python demo_strategy_test.py
```

## Test Results Analysis

### Success Metrics

- **Strategy Match**: Does the query route to the expected strategy?
- **Content Relevance**: Do results contain expected keywords?
- **Performance**: Latency and confidence measurements
- **Coverage**: Are all strategies being tested?

### Example Results

```
🧪 Running iLand Retrieval Strategy Tests
Strategy Selector: round_robin
================================================================================

✅ Routed to: vector (confidence: 0.50)
✅ Routed to: hybrid (confidence: 0.50)  
✅ Routed to: metadata (confidence: 0.50)
✅ Routed to: planner (confidence: 0.50)

📊 Test Summary
==================================================
Total Tests: 10
Successful Routes: 10 (100.0%)
Average Latency: 0.001s
Average Confidence: 0.50
Average Content Match: 0.83

📈 Strategy Usage:
vector: 2 uses, 100.0% accuracy
hybrid: 2 uses, 100.0% accuracy
metadata: 2 uses, 100.0% accuracy
planner: 1 uses, 100.0% accuracy
recursive: 1 uses, 100.0% accuracy
chunk_decoupling: 1 uses, 100.0% accuracy
summary: 1 uses, 100.0% accuracy
```

### Performance Comparison

| Selector | Success Rate | Avg Latency | Coverage |
|----------|-------------|-------------|----------|
| Round-Robin | 100% | 0.001s | 7/7 strategies |
| Heuristic | 40% | 0.001s | 3/7 strategies |
| LLM | 40% | 0.694s | 3/7 strategies |

## Key Features

### 1. Comprehensive Coverage
- Tests all 7 retrieval strategies
- Covers Thai and English queries
- Tests different query complexities

### 2. Realistic Test Data
- Based on actual Thai land deed documents
- Includes legal terminology and geographic data
- Proper Thai language content

### 3. Multiple Evaluation Metrics
- **Strategy routing accuracy**
- **Content relevance scoring**
- **Performance measurements**
- **Strategy distribution analysis**

### 4. Flexible Testing Framework
- Supports different strategy selectors
- Easy to add new test cases
- Detailed logging and analysis

## Integration with Existing Code

### 1. Router Integration
- Uses actual `iLandRouterRetriever` class
- Tests real routing logic and heuristics
- Validates strategy selection algorithms

### 2. CLI Integration
- Added `--test-retrieval-strategies` command
- Easy integration with existing workflows
- Supports all strategy selector options

### 3. Mock Adapters
- Realistic mock responses for each strategy
- Proper Thai content and metadata
- Maintains strategy-specific characteristics

## Usage Examples

### Testing Query Routing
```python
from test_iland_retrieval_strategies import iLandRetrievalStrategyTester

tester = iLandRetrievalStrategyTester()
router = tester.create_test_router("llm")

# Test specific query
query_bundle = QueryBundle(query_str="โฉนดที่ดินในจังหวัดชัยนาท")
nodes = router._retrieve(query_bundle)

# Check routing
metadata = nodes[0].node.metadata
strategy = metadata.get('selected_strategy')
confidence = metadata.get('strategy_confidence')
```

### Adding New Test Cases
```python
new_test = StrategyTestCase(
    query="Your test query here",
    expected_strategy="target_strategy",
    expected_results_contain=["keyword1", "keyword2"],
    description="Description of what this tests",
    confidence_threshold=0.5
)
```

## Benefits

1. **Validation**: Ensures routing logic works correctly
2. **Coverage**: Tests all retrieval strategies systematically  
3. **Performance**: Measures latency and accuracy
4. **Debugging**: Identifies routing issues and improvements
5. **Quality**: Validates content relevance of results

## Future Enhancements

1. **Real Data Testing**: Integration with actual iLand embeddings
2. **Performance Benchmarks**: Establish baseline metrics
3. **A/B Testing**: Compare different routing strategies
4. **Continuous Integration**: Automated testing in CI/CD
5. **Query Expansion**: Test more complex query patterns

---

This testing framework provides comprehensive validation of the iLand retrieval system's ability to route queries to appropriate strategies and return relevant results for Thai land deed data. 