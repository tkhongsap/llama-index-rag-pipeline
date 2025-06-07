# iLand Retrieval Package

This package implements an agentic retrieval workflow for Thai land deed data, mirroring the `src/agentic_retriever` system but adapted specifically for iLand data.

## Overview

The iLand retrieval system provides:

- **Index Classification**: Routes queries to appropriate iLand indices
- **Seven Retrieval Strategies**: Multiple approaches for different query types
- **Router Retriever**: Main orchestrator with LLM-based strategy selection
- **CLI Interface**: Command-line tools for testing and usage
- **Performance Logging**: Detailed statistics and performance tracking

## Architecture

```
Query → Index Classifier → Strategy Selector → Retriever Adapter → Results
```

### Two-Stage Routing

1. **Index Classification**: Determines which iLand index to search
2. **Strategy Selection**: Chooses the best retrieval strategy for the query

## Components

### Index Classifier (`index_classifier.py`)

Routes queries to appropriate iLand indices based on content analysis.

**Default Indices:**
- `iland_land_deeds`: Thai land deed documents (โฉนด, นส.3, นส.4, ส.ค.1)

**Classification Methods:**
- **LLM**: Uses GPT-4o-mini for intelligent routing
- **Embedding**: Cosine similarity with index descriptions

### Retrieval Strategies

#### 1. Vector Search (`retrievers/vector.py`)
- **Purpose**: Basic semantic similarity search
- **Best for**: General queries, fast results, conceptual similarity
- **Thai Support**: Works well with Thai embeddings

#### 2. Summary Retrieval (`retrievers/summary.py`)
- **Purpose**: Document summary-first retrieval
- **Best for**: Overview questions, high-level information
- **Thai Support**: Summarizes Thai land deed content

#### 3. Recursive Retrieval (`retrievers/recursive.py`)
- **Purpose**: Hierarchical retrieval with parent-child relationships
- **Best for**: Complex queries requiring multi-level information
- **Thai Support**: Navigates Thai document structure

#### 4. Metadata Filtering (`retrievers/metadata.py`)
- **Purpose**: Filters by Thai geographic and legal metadata
- **Best for**: Province/district queries, land deed type filtering
- **Thai Support**: 
  - Province names (จังหวัด): กรุงเทพมหานคร, สมุทรปราการ, etc.
  - Land deed types: โฉนด, นส.3, นส.4, ส.ค.1
  - Property types: ที่ดิน, บ้าน, อาคาร, คอนโด

#### 5. Chunk Decoupling (`retrievers/chunk_decoupling.py`)
- **Purpose**: Separates chunk retrieval from context synthesis
- **Best for**: Detailed analysis, precise chunk-level information
- **Thai Support**: Handles Thai text chunking and context

#### 6. Hybrid Search (`retrievers/hybrid.py`)
- **Purpose**: Combines semantic search with Thai keyword matching
- **Best for**: Queries needing both conceptual and exact term matches
- **Thai Support**: 
  - Thai keyword extraction (Unicode range \u0e00-\u0e7f)
  - Land deed terminology recognition
  - Weighted scoring (alpha parameter)

#### 7. Query Planning Agent (`retrievers/planner.py`)
- **Purpose**: Multi-step query planning and execution
- **Best for**: Complex multi-part questions, analytical tasks
- **Thai Support**: 
  - Decomposes Thai land deed queries
  - Handles location-based, legal, and procedural aspects
  - Fallback heuristics for Thai terms

### Router Retriever (`router.py`)

Main orchestrator that implements the two-stage routing process.

**Features:**
- LLM-based strategy selection with Thai context
- Heuristic fallbacks for reliability
- Round-robin testing mode
- Performance logging and metadata enrichment
- Debug logging support

**Strategy Selection Logic:**
- Analyzes query complexity and Thai content
- Considers land deed terminology and geographic references
- Applies reliability rankings based on performance
- Provides detailed reasoning for selections

## Usage

### Command Line Interface

```bash
# Load embeddings and start interactive mode
python -m src-iLand.retrieval.cli --load-embeddings latest --interactive

# Execute a single query
python -m src-iLand.retrieval.cli --load-embeddings latest --query "โฉนดที่ดินในกรุงเทพ"

# Test multiple strategies
python -m src-iLand.retrieval.cli --load-embeddings latest --test-queries "ที่ดินในสมุทรปราการ" "นส.3 คืออะไร"

# Show batch summary
python -m src-iLand.retrieval.cli --batch-summary
```

### Programmatic Usage

```python
from src_iland.retrieval import (
    iLandRouterRetriever,
    create_default_iland_classifier,
    VectorRetrieverAdapter
)
from src_iland.load_embedding import load_latest_iland_embeddings

# Load embeddings
embeddings = load_latest_iland_embeddings()

# Create adapters
adapters = {
    "iland_land_deeds": {
        "vector": VectorRetrieverAdapter.from_iland_embeddings(embeddings),
        # ... other strategies
    }
}

# Create router
router = iLandRouterRetriever.from_adapters(adapters)

# Execute query
from llama_index.core.schema import QueryBundle
results = router._retrieve(QueryBundle("โฉนดที่ดินในกรุงเทพ"))
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
CLASSIFIER_MODE=llm  # or "embedding"
```

### Strategy Selection Modes

- **llm**: Uses GPT-4o-mini for intelligent strategy selection
- **heuristic**: Rule-based selection using Thai content analysis
- **round_robin**: Cycles through strategies for testing

## Thai Language Support

### Text Processing
- Unicode range detection (\u0e00-\u0e7f)
- Thai keyword extraction and matching
- Province and district name recognition
- Land deed terminology handling

### Supported Thai Terms

**Geographic:**
- จังหวัด (Province)
- อำเภอ (District) 
- ตำบล (Subdistrict)
- กรุงเทพมหานคร, สมุทรปราการ, นนทบุรี, etc.

**Land Deed Types:**
- โฉนด (Title Deed)
- นส.3 (Nor Sor 3)
- นส.4 (Nor Sor 4) 
- ส.ค.1 (Sor Kor 1)

**Property Types:**
- ที่ดิน (Land)
- บ้าน (House)
- อาคาร (Building)
- คอนโด (Condominium)

## Performance Optimization

### Strategy Reliability Ranking
1. **Vector**: Most reliable, fast, always works
2. **Hybrid**: Good results, handles Thai keywords
3. **Recursive**: Good for complex hierarchical queries
4. **Chunk Decoupling**: Good for detailed analysis
5. **Planner**: Good for multi-step queries
6. **Metadata**: Good for geographic filtering
7. **Summary**: Good for overview queries

### Heuristic Rules
- Geographic queries → Metadata strategy
- Land deed type queries → Hybrid strategy
- Simple semantic queries → Vector strategy
- Complex multi-step queries → Planner strategy

## Example Queries

### Thai Queries
```
โฉนดที่ดินในกรุงเทพมหานคร
นส.3 คืออะไร
ที่ดินในอำเภอบางพลี จังหวัดสมุทรปราการ
ขั้นตอนการโอนโฉนดที่ดิน
ประเภทของเอกสารสิทธิ์ที่ดิน
```

### English Queries
```
Land deeds in Bangkok
What is Nor Sor 3?
Property ownership documents
Land transfer procedures
Types of land title documents
```

## Logging and Statistics

The system logs all retrieval calls with:
- Query text and routing decisions
- Index and strategy selection
- Confidence scores and reasoning
- Performance metrics (latency, result count)
- Error handling and fallback usage

## Testing

### Unit Tests
```bash
# Run retrieval tests
python -m pytest tests/test_iland_retrieval.py

# Test specific strategy
python -m pytest tests/test_iland_retrieval.py::test_vector_adapter
```

### Integration Tests
```bash
# Test full pipeline
python -m src-iLand.retrieval.cli --load-embeddings latest --test-queries "โฉนดที่ดิน" "นส.3"
```

## Troubleshooting

### Common Issues

1. **No embeddings loaded**
   - Solution: Run `--load-embeddings latest` first

2. **API key not found**
   - Solution: Set `OPENAI_API_KEY` environment variable

3. **Empty results**
   - Check embedding data quality
   - Try different strategies
   - Verify Thai text encoding

4. **Slow performance**
   - Use vector strategy for simple queries
   - Reduce top_k parameter
   - Check network connectivity for LLM calls

### Debug Mode
```python
router.enable_debug_logging(True)
```

## Future Enhancements

- Additional iLand indices (transactions, legal documents, geographic data)
- Advanced Thai NLP integration (pythainlp)
- Custom embedding models for Thai content
- Real-time performance monitoring
- Caching layer for frequent queries

## Related Documentation

- [iLand Load Embedding Package](../load_embedding/README.md)
- [Main Agentic Retriever](../../src/agentic_retriever/README.md)
- [Project Architecture](../../../diagram.md) 