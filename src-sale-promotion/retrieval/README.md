# Sales Promotion Retrieval Module

Agentic retrieval system with 7 strategies optimized for sales and promotional documents.

## Architecture

```
Query → [Index Classification] → [Strategy Selection] → [Retriever] → Results
```

### Two-Stage Routing

**Stage 1: Index Classification**
- Determines which sales/promotion index to search
- Options: `campaign_data`, `product_catalog`, `performance_metrics`, `customer_segments`

**Stage 2: Strategy Selection**
- LLM analyzes query to choose best retrieval strategy
- Considers query complexity, data type, performance needs

## Seven Retrieval Strategies

### 1. Vector Retriever (`retrievers/vector.py`)
**Best for**: Simple semantic similarity queries

```python
# Example queries:
"Find campaigns similar to summer sale"
"Show products like wireless headphones"
"What promotions target millennials?"
```

**Characteristics:**
- Fastest strategy (~100ms)
- Good for general semantic search
- Reliable baseline performance

### 2. Hybrid Retriever (`retrievers/hybrid.py`)
**Best for**: Queries needing both semantic and keyword matching

```python
# Example queries:
"Email campaigns with keyword 'BOGO'"
"Product category: electronics AND discount > 20%"
"Social media posts mentioning brand name"
```

**Characteristics:**
- Combines vector + BM25 keyword search
- Best overall performance for mixed queries
- Handles product names, campaign IDs well

### 3. Metadata Retriever (`retrievers/metadata.py`)
**Best for**: Structured attribute-based queries

```python
# Example queries:
"Campaigns with ROI > 150%"
"Products in electronics category priced $50-100"
"Email campaigns launched in Q4 2024"
"Show promotions in California with high conversion"
```

**Characteristics:**
- **Fast**: Sub-50ms with fast metadata indexing
- Structured filtering before vector search
- Perfect for analytical queries

**Indexed Fields:**
- Categorical: `campaign_type`, `product_category`, `region`, `channel`
- Numeric: `price`, `discount_percentage`, `roi`, `revenue`, `conversion_rate`
- Dates: `start_date`, `end_date`

### 4. Recursive Retriever (`retrievers/recursive.py`)
**Best for**: Hierarchical campaign structures

```python
# Example queries:
"Show Q4 holiday campaign hierarchy"
"Break down multi-channel campaign components"
"What products are in the summer sale bundle?"
```

**Characteristics:**
- Multi-level retrieval (campaign → sub-campaigns → products)
- Good for complex nested structures
- Preserves parent-child relationships

### 5. Summary Retriever (`retrievers/summary.py`)
**Best for**: Overview questions about campaigns

```python
# Example queries:
"Summarize 2024 email marketing performance"
"Overview of best-performing product categories"
"What were the main campaigns last quarter?"
```

**Characteristics:**
- Retrieves from document summaries first
- Fast for high-level questions
- Good for exploration

### 6. Chunk Decoupling Retriever (`retrievers/chunk_decoupling.py`)
**Best for**: Detailed analysis requiring precise context

```python
# Example queries:
"Detailed breakdown of campaign budget allocation"
"Explain the pricing strategy for product X"
"What were the exact terms of the promotion?"
```

**Characteristics:**
- Separates retrieval chunk from context
- More accurate for detailed questions
- Better context preservation

### 7. Planner Retriever (`retrievers/planner.py`)
**Best for**: Complex multi-step analytical questions

```python
# Example queries:
"Compare Q3 vs Q4 campaigns, analyze ROI trends, recommend best channel"
"Which products had highest discount but lowest revenue? Why?"
"Break down customer journey from impression to purchase for top campaign"
```

**Characteristics:**
- LLM-driven query decomposition
- Multi-step reasoning
- Best for complex analytics

## Components

### `router.py`
Main router that orchestrates the retrieval process:
```python
from src_sale_promotion.retrieval import SalesPromotionRouterRetriever

router = SalesPromotionRouterRetriever.from_embeddings(
    embeddings_path="data/embedding/sale-promotion/latest",
    strategy_selector="llm"
)

results = router.retrieve("Show me high ROI email campaigns")
```

### `index_classifier.py`
Classifies queries to appropriate indices:
```python
classifier = SalesPromotionIndexClassifier(
    available_indices={
        "campaign_data": "Campaign info, dates, budgets, channels",
        "performance_metrics": "ROI, revenue, conversions, analytics",
        "product_catalog": "Products, prices, categories, SKUs",
        "customer_segments": "Demographics, segments, targeting"
    }
)

result = classifier.classify_query("Show ROI for summer campaigns")
# Returns: {"selected_index": "performance_metrics", "confidence": 0.9}
```

### `fast_metadata_index.py`
High-performance metadata filtering:
```python
from src_sale_promotion.retrieval.fast_metadata_index import FastMetadataIndexManager

index_manager = FastMetadataIndexManager()
index_manager.build_from_nodes(nodes)

# Query with filters
filtered_docs = index_manager.query({
    "campaign_type": "email",
    "roi": {"min": 100, "max": 200},
    "region": "West Coast"
})
# Returns in <50ms for 50k+ documents
```

**Performance:**
- Sub-50ms response time
- 45-90% document reduction before vector search
- Inverted indices for categorical fields
- B-tree indices for numeric fields

### `cli.py`
Command-line interface:
```bash
# Interactive mode
python -m src_sale_promotion.retrieval.cli --interactive

# Single query
python -m src_sale_promotion.retrieval.cli --query "Show summer campaigns"

# Test multiple queries
python -m src_sale_promotion.retrieval.cli --test-queries "Query 1" "Query 2"

# Batch processing
python -m src_sale_promotion.retrieval.cli --batch-file queries.txt
```

### `streamlit_cli.py`
Interactive web interface:
```bash
streamlit run src-sale-promotion/retrieval/streamlit_cli.py
```

Features:
- WhatsApp-style chat interface
- Strategy selection visualization
- Confidence scores
- Metadata display
- Export results

## Usage Examples

### Basic Retrieval
```python
from src_sale_promotion.retrieval.cli import setup_retrieval_system

# Setup
router, index = setup_retrieval_system(
    embeddings_path="latest",
    strategy_selector="llm"
)

# Query
results = router.retrieve("What were our best email campaigns?")

for node in results:
    print(f"Campaign: {node.metadata['campaign_name']}")
    print(f"ROI: {node.metadata['roi']}%")
    print(f"Score: {node.score:.3f}\n")
```

### Strategy Comparison
```python
from src_sale_promotion.retrieval.retrievers import (
    VectorRetrieverAdapter,
    HybridRetrieverAdapter,
    MetadataRetrieverAdapter
)

query = "Email campaigns with ROI > 150%"

# Try different strategies
strategies = {
    "vector": VectorRetrieverAdapter(index),
    "hybrid": HybridRetrieverAdapter(index),
    "metadata": MetadataRetrieverAdapter(index, enable_fast_filtering=True)
}

for name, retriever in strategies.items():
    results = retriever.retrieve(query)
    print(f"{name}: {len(results)} results")
```

### Custom Routing Logic
```python
router = SalesPromotionRouterRetriever(
    retrievers={
        "campaign_data": {
            "vector": VectorRetrieverAdapter(...),
            "hybrid": HybridRetrieverAdapter(...),
            "metadata": MetadataRetrieverAdapter(...)
        }
    },
    strategy_selector="llm",  # or "round_robin", "heuristic"
    llm_strategy_mode="enhanced"
)

router.enable_debug_logging(True)
results = router.retrieve("Your query here")
```

## Performance Optimization

### For Speed
1. Use **metadata retriever** with fast indexing for filtered queries
2. Use **vector retriever** for simple semantic queries
3. Enable caching for repeated queries

### For Accuracy
1. Use **hybrid retriever** for product/campaign name queries
2. Use **planner retriever** for complex analytical questions
3. Use **recursive retriever** for hierarchical structures

### For Cost
1. Prefer lightweight strategies (vector, metadata) over planner
2. Cache frequently asked queries
3. Use smaller top_k values

## Configuration

Environment variables:
```bash
# Retrieval settings
TOP_K=5                      # Number of results to retrieve
SIMILARITY_THRESHOLD=0.7     # Minimum similarity score
ENABLE_FAST_FILTERING=true   # Enable fast metadata indexing

# Strategy selection
CLASSIFIER_MODE=llm          # or "embedding"
STRATEGY_SELECTOR=llm        # or "round_robin", "default"
LLM_STRATEGY_MODE=enhanced   # or "simple"

# Models
LLM_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

## Testing

```bash
# Unit tests
python -m pytest src-sale-promotion/test/test_retrieval.py -v

# Integration tests
python -m pytest src-sale-promotion/test/test_end_to_end.py -v

# Strategy performance test
python -m src_sale_promotion.retrieval.cli --benchmark
```

## Troubleshooting

**Empty Results**
- Check if embeddings are loaded correctly
- Verify metadata fields in documents
- Try different retrieval strategy

**Slow Performance**
- Enable fast metadata indexing
- Reduce top_k value
- Use simpler strategies (vector, metadata)

**Wrong Strategy Selected**
- Switch to manual strategy selection
- Adjust classifier_mode to "embedding"
- Enable debug logging to see routing decisions

## Reference

Based on implementations from:
- `src/agentic_retriever/` - Main agentic retrieval system
- `src-imap/retrieval/` - Modular retrieval architecture
