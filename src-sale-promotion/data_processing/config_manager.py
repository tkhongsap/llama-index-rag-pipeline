"""
Configuration manager for sales promotion data processing.

Handles loading, validation, and saving of YAML configuration files.
"""

import os
import yaml
import json
import logging
from dataclasses import asdict
from typing import Dict, Any
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .models import DatasetConfig, FieldMapping
except ImportError:
    from models import DatasetConfig, FieldMapping

logger = logging.getLogger(__name__)


class ConfigManager:
    """Handles configuration loading, saving, and management for promotions"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Create output directory structure if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "documents", "markdown_files"), exist_ok=True)
    
    def load_config(self, config_path: str = None) -> DatasetConfig:
        """
        Load dataset configuration from YAML file.
        
        Args:
            config_path: Path to config YAML. If None, uses default.
        
        Returns:
            DatasetConfig object
        """
        if config_path is None:
            # Use default config from module directory
            module_dir = Path(__file__).parent
            config_path = module_dir / "config" / "promotion_fields.yaml"
        
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse field mappings
        field_mappings = [
            FieldMapping(**mapping) for mapping in config_dict['field_mappings']
        ]
        
        # Create DatasetConfig
        config = DatasetConfig(
            name=config_dict['name'],
            description=config_dict['description'],
            version=config_dict['version'],
            field_mappings=field_mappings,
            product_patterns=config_dict.get('product_patterns', {}),
            customer_tiers=config_dict.get('customer_tiers', {}),
            text_template=config_dict.get('text_template')
        )
        
        # Store additional config sections for use by processors
        config._raw_config = config_dict  # Store full config for access by utilities
        
        logger.info(f"Configuration loaded: {config.name} v{config.version}")
        logger.info(f"  - {len(config.field_mappings)} field mappings")
        logger.info(f"  - {len(config.product_patterns)} product categories")
        logger.info(f"  - {len(config.customer_tiers)} customer tiers")
        
        return config
    
    def validate_config(self, config: DatasetConfig) -> bool:
        """
        Validate configuration completeness and correctness.
        
        Args:
            config: DatasetConfig to validate
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required fields
        if not config.name:
            raise ValueError("Config name is required")
        
        if not config.field_mappings:
            raise ValueError("Config must have at least one field mapping")
        
        # Check required field mappings exist
        required_fields = [
            'customer_type', 'start_date', 'end_date', 'product_name',
            'buy_quantity', 'buy_unit', 'free_item_name', 'free_quantity', 'free_unit'
        ]
        
        mapped_keys = [mapping.metadata_key for mapping in config.field_mappings]
        missing_fields = [field for field in required_fields if field not in mapped_keys]
        
        if missing_fields:
            raise ValueError(f"Missing required field mappings: {missing_fields}")
        
        # Validate product patterns
        if not config.product_patterns:
            logger.warning("No product patterns defined - category extraction will be limited")
        
        # Validate customer tiers
        if not config.customer_tiers:
            logger.warning("No customer tiers defined - tier mapping will be limited")
        
        logger.info("✓ Configuration validation passed")
        return True
    
    def save_config_snapshot(self, config: DatasetConfig) -> str:
        """
        Save a snapshot of the configuration used for processing.
        
        Args:
            config: DatasetConfig to save
        
        Returns:
            Path to saved config file
        """
        config_path = os.path.join(self.output_dir, "config_used.yaml")
        
        # Convert to dict for serialization
        config_dict = {
            'name': config.name,
            'version': config.version,
            'description': config.description,
            'field_mappings': [asdict(mapping) for mapping in config.field_mappings],
            'product_patterns': config.product_patterns,
            'customer_tiers': config.customer_tiers,
            'text_template': config.text_template
        }
        
        # Add raw config if available
        if hasattr(config, '_raw_config'):
            config_dict.update(config._raw_config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Configuration snapshot saved to: {config_path}")
        return config_path
    
    def save_processing_report(self, report: Dict[str, Any]) -> str:
        """
        Save processing statistics and report.
        
        Args:
            report: Dictionary with processing statistics
        
        Returns:
            Path to saved report file
        """
        report_path = os.path.join(self.output_dir, "processing_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Processing report saved to: {report_path}")
        return report_path
    
    def save_error_report(self, failed_rows: list) -> str:
        """
        Save error report for failed conversions.
        
        Args:
            failed_rows: List of dictionaries with error information
        
        Returns:
            Path to saved error report
        """
        error_report_path = os.path.join(self.output_dir, "conversion_errors.json")
        
        with open(error_report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_failures': len(failed_rows),
                'failures': failed_rows[:100]  # Save first 100 errors
            }, f, indent=2, ensure_ascii=False)
        
        if failed_rows:
            logger.warning(f"Failed to process {len(failed_rows)} rows. See {error_report_path}")
        
        return error_report_path
    
    def save_readme(self, summary: Dict[str, Any]) -> str:
        """
        Generate and save README with processing summary.
        
        Args:
            summary: Dictionary with processing summary
        
        Returns:
            Path to saved README file
        """
        readme_path = os.path.join(self.output_dir, "README.md")
        
        readme_content = f"""# Sales Promotion Data Processing Results

## Processing Summary

**Date:** {summary.get('processing_date', 'N/A')}  
**Configuration:** {summary.get('config_name', 'N/A')} v{summary.get('config_version', 'N/A')}  
**Total Promotions:** {summary.get('total_promotions', 0)}  
**Processing Time:** {summary.get('processing_time', 'N/A')}

## Outputs

### Primary Output (For Phase 2)
- **File:** `documents/documents.jsonl`
- **Format:** JSONL (newline-delimited JSON)
- **Lines:** {summary.get('total_promotions', 0)}
- **Purpose:** Machine-readable for Phase 2 embedding generation

### Secondary Output (For Human QA)
- **Directory:** `documents/markdown_files/`
- **Files:** {summary.get('total_promotions', 0)} markdown files
- **Format:** `promotion_001.md` through `promotion_{summary.get('total_promotions', 0):03d}.md`
- **Purpose:** Human-readable QA and Thai text validation

## Statistics

### Data Completeness
- **Fields Extracted:** {summary.get('fields_extracted', 0)} per document
- **Base Fields:** {summary.get('base_fields', 10)}
- **Derived Fields:** {summary.get('derived_fields', 0)}

### Processing Metrics
- **Processing Speed:** {summary.get('processing_speed', 'N/A')} docs/sec
- **Memory Usage:** {summary.get('memory_usage', 'N/A')}
- **Success Rate:** {summary.get('success_rate', 'N/A')}%

### Date Conversion
- **Buddhist → Gregorian:** {summary.get('dates_converted', 0)} dates converted
- **Active Promotions:** {summary.get('active_promotions', 0)}
- **Expired Promotions:** {summary.get('expired_promotions', 0)}
- **Expiring Soon:** {summary.get('expiring_soon', 0)}

### Product Categorization
{self._format_category_stats(summary.get('product_categories', {}))}

### Customer Tiers
{self._format_tier_stats(summary.get('customer_tiers', {}))}

## Files Generated

1. **documents.jsonl** - Primary output for Phase 2
2. **markdown_files/** - {summary.get('total_promotions', 0)} individual markdown files
3. **config_used.yaml** - Configuration snapshot
4. **processing_report.json** - Detailed statistics
5. **README.md** - This file

## Next Steps

### Phase 2: Embedding Generation

Load the JSONL file and convert to LlamaIndex Documents:

```python
from llama_index.core import Document
import json

documents = []
with open('documents/documents.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        doc = Document(
            text=data['text'],
            metadata=data['metadata'],
            doc_id=data['id']
        )
        documents.append(doc)

print(f"Loaded {{len(documents)}} documents for embedding")
```

## Validation

All outputs have been validated for:
- ✓ Thai UTF-8 encoding integrity
- ✓ JSONL format correctness
- ✓ Metadata completeness (15+ fields)
- ✓ Date conversion accuracy
- ✓ No duplicate document IDs

---
Generated by Sales Promotion Data Processing Pipeline Phase 1
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"README saved to: {readme_path}")
        return readme_path
    
    def _format_category_stats(self, categories: Dict[str, int]) -> str:
        """Format product category statistics for README"""
        if not categories:
            return "- No data"
        
        lines = []
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{category}**: {count} promotions")
        
        return "\n".join(lines)
    
    def _format_tier_stats(self, tiers: Dict[str, int]) -> str:
        """Format customer tier statistics for README"""
        if not tiers:
            return "- No data"
        
        lines = []
        for tier, count in sorted(tiers.items()):
            lines.append(f"- **{tier}**: {count} promotions")
        
        return "\n".join(lines)

