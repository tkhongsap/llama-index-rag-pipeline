# Sales Promotion Pipeline Tests

Test suite for the sales promotion RAG pipeline.

## Test Structure

```
test/
├── test_data_processing.py      # Data processing tests
├── test_embedding.py             # Embedding generation tests
├── test_load_embedding.py        # Embedding loading tests
├── test_retrieval.py             # Retrieval strategy tests
├── test_router.py                # Router and routing tests
└── test_end_to_end.py            # Full pipeline integration test
```

## Running Tests

### All Tests
```bash
python -m pytest src-sale-promotion/test/ -v
```

### Specific Module
```bash
python -m pytest src-sale-promotion/test/test_retrieval.py -v
```

### With Coverage
```bash
python -m pytest src-sale-promotion/test/ --cov=src_sale_promotion --cov-report=html
```

## Test Data

Sample test data located in:
```
data/input_docs/sale-promotion/test/
├── sample_campaigns.csv
├── sample_products.csv
└── sample_performance.csv
```

## Writing Tests

### Example Test
```python
import pytest
from src_sale_promotion.data_processing import SalesPromotionProcessor

def test_campaign_processing():
    processor = SalesPromotionProcessor(
        input_path="test_data/sample_campaigns.csv"
    )
    documents = processor.process()

    assert len(documents) > 0
    assert documents[0].campaign.campaign_name
    assert documents[0].campaign.campaign_type in CAMPAIGN_TYPES
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Fast execution
- No external dependencies

### Integration Tests
- Test component interactions
- May require embeddings
- Longer execution time

### End-to-End Tests
- Test complete pipeline
- Requires all components
- Full data flow validation
