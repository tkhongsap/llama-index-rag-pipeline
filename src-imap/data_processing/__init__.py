"""
iLand Data Processing Package

This package provides tools for processing iLand dataset CSV files into documents
suitable for embedding and retrieval systems.

Main Components:
- iLandCSVConverter: Main converter class
- DatasetConfig: Configuration for field mappings
- SimpleDocument: Document representation
"""

from .iland_converter import iLandCSVConverter
from .models import DatasetConfig, FieldMapping, SimpleDocument
from .csv_analyzer import CSVAnalyzer
from .config_manager import ConfigManager
from .document_processor import DocumentProcessor
from .file_output import FileOutputManager
from .statistics_generator import StatisticsGenerator

__all__ = [
    'iLandCSVConverter',
    'DatasetConfig',
    'FieldMapping', 
    'SimpleDocument',
    'CSVAnalyzer',
    'ConfigManager',
    'DocumentProcessor',
    'FileOutputManager',
    'StatisticsGenerator'
]

__version__ = "1.0.0" 