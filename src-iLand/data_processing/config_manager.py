import os
import yaml
import json
import logging
from dataclasses import asdict
from typing import Dict, Any
from .models import DatasetConfig, FieldMapping

logger = logging.getLogger(__name__)


class ConfigManager:
    """Handles configuration loading, saving, and management"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "iland_markdown_files"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
    
    def save_config(self, config: DatasetConfig, config_path: str = None) -> str:
        """Save dataset configuration to YAML file"""
        if config_path is None:
            config_path = os.path.join(self.output_dir, f"{config.name}_config.yaml")
        
        # Convert to dict for serialization
        config_dict = {
            'name': config.name,
            'description': config.description,
            'field_mappings': [asdict(mapping) for mapping in config.field_mappings],
            'text_template': config.text_template,
            'embedding_fields': config.embedding_fields
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration saved to: {config_path}")
        return config_path
    
    def load_config(self, config_path: str) -> DatasetConfig:
        """Load dataset configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        field_mappings = [
            FieldMapping(**mapping) for mapping in config_dict['field_mappings']
        ]
        
        config = DatasetConfig(
            name=config_dict['name'],
            description=config_dict['description'],
            field_mappings=field_mappings,
            text_template=config_dict.get('text_template'),
            embedding_fields=config_dict.get('embedding_fields', [])
        )
        
        return config
    
    def save_analysis_report(self, analysis: Dict[str, Any]):
        """Save CSV analysis report"""
        report_path = os.path.join(self.output_dir, "reports", "iland_csv_analysis_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis report saved to: {report_path}")
        return report_path
    
    def save_conversion_summary(self, summary: Dict[str, Any]):
        """Save conversion summary statistics"""
        summary_path = os.path.join(self.output_dir, "reports", "conversion_summary.json")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion summary saved to: {summary_path}")
        return summary_path
    
    def save_error_report(self, failed_rows: list):
        """Save error report for failed conversions"""
        error_report_path = os.path.join(self.output_dir, "reports", "conversion_errors.json")
        
        with open(error_report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_failures': len(failed_rows),
                'failures': failed_rows[:100]  # Save first 100 errors
            }, f, indent=2, ensure_ascii=False)
        
        logger.warning(f"Failed to process {len(failed_rows)} rows. See {error_report_path}")
        return error_report_path 