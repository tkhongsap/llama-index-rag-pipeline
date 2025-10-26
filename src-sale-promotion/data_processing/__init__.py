"""
Sales Promotion Data Processing Module - Phase 1

This module transforms raw Thai promotional CSV data into SimpleDocument objects
with natural language text and rich metadata.

Main Components:
- models: Data structures (SimpleDocument, PromotionDocument, DatasetConfig)
- config_manager: Configuration loading and validation
- processors: CSV processing, metadata extraction, Thai normalization
- utils: Date conversion, product categorization, customer tier mapping
- file_output: JSONL and markdown file generation
- statistics_generator: Processing metrics and reports
- main: Pipeline orchestrator

Usage:
    python -m src_sale_promotion.data_processing.main

Output:
    - documents.jsonl (primary, for Phase 2)
    - markdown_files/ (secondary, for human QA)
    - processing_report.json
    - config_used.yaml
    - README.md
"""

__version__ = "2.0"
__author__ = "Data Engineering Team"

from .models import SimpleDocument, PromotionDocument, DatasetConfig, FieldMapping
from .config_manager import ConfigManager
from .file_output import FileOutputManager
from .statistics_generator import StatisticsGenerator
from .main import PromotionDataProcessor

__all__ = [
    'SimpleDocument',
    'PromotionDocument',
    'DatasetConfig',
    'FieldMapping',
    'ConfigManager',
    'FileOutputManager',
    'StatisticsGenerator',
    'PromotionDataProcessor',
]
