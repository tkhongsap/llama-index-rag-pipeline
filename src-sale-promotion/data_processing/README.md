# Data Processing Module

Converts CSV files and raw documents into structured markdown format optimized for embedding and retrieval.

## Purpose

This module handles the first stage of the pipeline:
- **Input**: CSV files with sales/promotion data, marketing documents, campaign reports
- **Output**: Structured markdown documents with extracted metadata
- **Process**: CSV analysis → metadata extraction → document generation

## Components

### `models.py`
Data models for sales promotion documents:
```python
@dataclass
class SalesPromotionDocument:
    campaign_id: str
    campaign_name: str
    campaign_type: str
    product_info: Dict
    performance_metrics: Dict
    metadata: Dict
```

### `document_processor.py`
Main processor that:
- Analyzes CSV structure
- Extracts campaign and product metadata
- Generates structured markdown
- Handles sales-specific field mapping

### `config_manager.py`
Configuration management for:
- Field mapping rules
- Metadata extraction patterns
- CSV parsing options
- Output formatting

### `file_output.py`
Handles file writing:
- Markdown generation
- JSON metadata export
- Statistics tracking

## Usage

### Basic Usage
```bash
python -m src_sale_promotion.data_processing.run_data_processing_standalone
```

### Programmatic Usage
```python
from src_sale_promotion.data_processing import SalesPromotionProcessor

processor = SalesPromotionProcessor(
    input_path="data/input_docs/sale-promotion/campaigns.csv",
    output_dir="data/processed/sale-promotion"
)

documents = processor.process()
print(f"Processed {len(documents)} documents")
```

## Input Data Format

### Expected CSV Columns

**Campaign Fields:**
- `campaign_id`, `campaign_name`, `campaign_type`
- `start_date`, `end_date`, `status`
- `budget`, `channels`, `target_audience`

**Product Fields:**
- `product_id`, `product_name`, `category`
- `price`, `discount_percentage`, `sku`

**Performance Fields:**
- `impressions`, `clicks`, `conversions`
- `revenue`, `roi`, `ctr`

### Example CSV
```csv
campaign_id,campaign_name,campaign_type,product_name,category,price,discount_percentage,clicks,conversions,revenue
CAMP_001,Summer Sale 2024,seasonal_sale,Wireless Headphones,electronics,89.99,25,15420,892,67345
CAMP_002,Email Blast - New Arrivals,email,Yoga Mat,sports,34.99,15,8234,456,14256
```

## Output Format

### Markdown Document
```markdown
# Campaign: Summer Sale 2024

**Campaign ID**: CAMP_001
**Type**: Seasonal Sale
**Duration**: June 1 - August 31, 2024
**Status**: Active

## Products
- **Wireless Headphones** (Electronics)
  - Original Price: $89.99
  - Discount: 25%
  - Final Price: $67.49

## Performance
- Impressions: 250,000
- Clicks: 15,420 (CTR: 6.2%)
- Conversions: 892 (CVR: 5.8%)
- Revenue: $67,345
- ROI: 187%

## Target Audience
- Age: 25-44
- Interests: Technology, Music
- Channel: Email + Social Media
```

### Metadata JSON
```json
{
  "campaign_id": "CAMP_001",
  "campaign_name": "Summer Sale 2024",
  "campaign_type": "seasonal_sale",
  "products": [...],
  "performance": {...},
  "dates": {...}
}
```

## Configuration

### Custom Field Mapping
Edit `config.yaml` to map your CSV columns:
```yaml
field_mapping:
  campaign:
    id_field: "campaign_id"
    name_field: "campaign_name"
    type_field: "campaign_type"
  product:
    name_field: "product_name"
    category_field: "category"
    price_field: "price"
```

## Features

- ✅ Automatic CSV structure detection
- ✅ Sales-specific metadata extraction
- ✅ Campaign timeline parsing
- ✅ Product attribute categorization
- ✅ Performance metrics calculation
- ✅ Multi-product campaign support
- ✅ Geographic data normalization
- ✅ Batch processing for large datasets

## Testing

```bash
# Run tests
python -m pytest src-sale-promotion/test/test_data_processing.py -v

# Test with sample data
python -m src_sale_promotion.data_processing.run_data_processing_standalone --sample
```

## Performance

- **Processing Speed**: ~100-200 documents/second
- **Memory Usage**: ~50MB per 1000 documents
- **CSV Size Limit**: Up to 1M rows (tested)

## Troubleshooting

**Issue**: Missing required fields
- **Solution**: Check `config.yaml` field mapping matches your CSV

**Issue**: Date parsing errors
- **Solution**: Ensure dates in ISO format (YYYY-MM-DD) or configure format in settings

**Issue**: Large CSV memory issues
- **Solution**: Enable chunked processing in config: `chunk_size: 10000`
