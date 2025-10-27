# Sales Promotion RAG Pipeline

A specialized RAG (Retrieval-Augmented Generation) pipeline for sales and promotion document processing. This pipeline follows the modular architecture established in the main project with focus on marketing materials, promotional campaigns, and sales documentation.

## Overview

This pipeline provides intelligent retrieval and analysis for sales/promotion documents including:
- **Marketing Campaigns**: Product launches, seasonal promotions, discount campaigns
- **Sales Materials**: Brochures, catalogs, product specifications, pricing sheets
- **Promotion Analytics**: Campaign performance, ROI metrics, conversion data
- **Customer Segments**: Target audience profiles, demographic data, buying patterns

## Architecture

```
src-sale-promotion/
├── common/                      # Shared utilities and constants
├── data_processing/             # CSV/Document → Markdown conversion
├── docs_embedding/              # Embedding generation and batch processing
├── load_embedding/              # Embedding loading and index reconstruction
├── retrieval/                   # Agentic retrieval with 7 strategies
│   ├── retrievers/              # Individual retrieval strategy implementations
│   ├── router.py                # Two-stage routing (index + strategy)
│   ├── index_classifier.py      # Routes queries to appropriate indices
│   ├── fast_metadata_index.py   # Sub-50ms metadata filtering
│   ├── cli.py                   # Command-line interface
│   └── streamlit_cli.py         # Interactive chat UI
└── test/                        # Unit and integration tests
```

## Key Features

### Data Processing
- Automatic CSV structure detection for sales/promotion data
- Metadata extraction for:
  - Campaign details (name, type, duration, budget)
  - Product information (category, SKU, price, discount)
  - Target demographics (age group, income level, region)
  - Performance metrics (views, clicks, conversions, revenue)

### Fast Metadata Indexing
- **Sub-50ms** metadata filtering on large promotional datasets
- Inverted indices for categorical fields (product_category, campaign_type, region)
- B-tree indices for numeric fields (price, discount_percentage, revenue, roi)
- Optimized for promotional calendar and seasonal analysis

### Seven Retrieval Strategies
1. **Vector**: Semantic similarity search for general queries
2. **Hybrid**: Combined semantic + keyword search for product/campaign names
3. **Metadata**: Fast filtering by campaign attributes, dates, regions
4. **Recursive**: Hierarchical retrieval for complex campaign structures
5. **Summary**: Document summary-first retrieval for overview questions
6. **Chunk Decoupling**: Separates embedding from content for detailed analysis
7. **Planner**: Multi-step query planning for analytical questions

## Quick Start

### 1. Data Processing
```bash
# Convert CSVs under data/input_docs/sale-promotion/ into JSONL + markdown
python -m src_sale_promotion.data_processing.run_data_processing_standalone

# Use the packaged mock dataset instead of local files
python -m src_sale_promotion.data_processing.run_data_processing_standalone --sample
```

### 2. Generate Embeddings
```bash
# Create throttled batch embeddings using the latest processed JSONL
python -m src_sale_promotion.docs_embedding.batch_embedding
```

### 3. Interactive Retrieval
```bash
# List available embedding drops (newest first)
python -m src_sale_promotion.retrieval.cli --list-embeddings

# CLI - Interactive mode (auto-loads the newest embeddings)
python -m src_sale_promotion.retrieval.cli --interactive

# CLI - Single query with explicit embedding directory override
python -m src_sale_promotion.retrieval.cli \
  --embedding-path data/embedding/sale-promotion/embeddings_sales_YYYYMMDD_HHMMSS \
  --query "What were our best performing summer campaigns?"

# Streamlit UI
streamlit run src-sale-promotion/retrieval/streamlit_cli.py
```

## Sample Queries

The pipeline is optimized for queries like:

**Campaign Analysis:**
- "Show me all summer campaigns with ROI > 150%"
- "Which products had the highest conversion rates in Q4?"
- "Compare email vs social media campaign performance"

**Product Intelligence:**
- "Find products with discounts above 30% that increased sales"
- "What are the top-selling product categories by region?"
- "List all electronic products launched in the last quarter"

**Audience Insights:**
- "Which demographics responded best to our premium line?"
- "Show campaigns targeting millennials with high engagement"
- "Analyze customer segments for our loyalty program"

**Performance Metrics:**
- "Calculate average ROI across all digital campaigns"
- "What was our revenue from promotional campaigns last year?"
- "Show conversion funnel breakdown for Black Friday sales"

## Metadata Schema

### Campaign Metadata
- `campaign_id`, `campaign_name`, `campaign_type`
- `start_date`, `end_date`, `status`
- `budget`, `spend`, `roi`
- `target_audience`, `channels`

### Product Metadata
- `product_id`, `product_name`, `category`, `subcategory`
- `price`, `discount_percentage`, `final_price`
- `sku`, `brand`, `inventory_status`

### Performance Metadata
- `impressions`, `clicks`, `ctr`
- `conversions`, `conversion_rate`
- `revenue`, `cost_per_acquisition`
- `engagement_rate`, `bounce_rate`

### Geographic Metadata
- `region`, `country`, `city`
- `store_location`, `distribution_channel`

## Environment Setup

Required environment variables in `.env`:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
CHUNK_SIZE=1024
CHUNK_OVERLAP=50
EMBED_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
BATCH_SIZE=10
MAX_WORKERS=4
CLASSIFIER_MODE=llm  # or "embedding"
```

## Development Guidelines

### Adding Custom Metadata Fields
Edit `data_processing/metadata_extractor.py` to add sales-specific fields:
```python
SALES_PROMOTION_FIELDS = {
    'campaign_type': ['email', 'social', 'display', 'search'],
    'product_category': ['electronics', 'fashion', 'home', 'beauty'],
    'season': ['spring', 'summer', 'fall', 'winter', 'holiday'],
    'target_age': ['18-24', '25-34', '35-44', '45-54', '55+']
}
```

### Performance Optimization
- Use **metadata retriever** for structured queries with filters
- Use **hybrid retriever** for product/campaign name searches
- Use **vector retriever** for semantic similarity questions
- Enable fast metadata indexing for large datasets (>10k documents)

## Testing

```bash
# Run all tests
python -m pytest src-sale-promotion/test/ -v

# Test specific module
python -m pytest src-sale-promotion/test/test_data_processing.py -v

# End-to-end test
python src-sale-promotion/test/end_to_end_test.py
```

## Integration with Main Pipeline

This pipeline follows the same patterns as:
- `src/` - Main general-purpose RAG pipeline
- `src-iLand/` - Thai land deed document processing
- `src-imap/` - iMap document processing
- `src-orion/` - Financial data processing

All pipelines share:
- Common retrieval strategies (7 strategies)
- Two-stage routing architecture
- Fast metadata indexing
- Modular component design

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Directory structure and module setup
- [x] Data processing pipeline
- [x] Embedding generation
- [x] Basic retrieval strategies

### Phase 2: Advanced Features (Planned)
- [ ] Seasonal campaign analysis
- [ ] A/B testing comparison tools
- [ ] Customer lifetime value calculations
- [ ] Predictive campaign performance

### Phase 3: Integrations (Future)
- [ ] CRM system integration
- [ ] Marketing automation platforms
- [ ] Analytics dashboard
- [ ] Real-time campaign monitoring

## Support

For questions and issues:
1. Check the main project documentation in `/README.md` and `/CLAUDE.md`
2. Review similar implementations in `src-imap/` and `src-iLand/`
3. Refer to retrieval strategy docs in `retrieval/README.md`

---

Built with ❤️ following the LlamaIndex RAG Pipeline architecture
