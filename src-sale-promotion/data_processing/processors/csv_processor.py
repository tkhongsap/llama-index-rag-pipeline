"""
CSV processor for loading and parsing promotion data.

Handles UTF-8 Thai text, field mapping, and base metadata extraction.
"""

import pandas as pd
import logging
from typing import Dict, Any, List
from pathlib import Path

# Handle both relative and absolute imports
try:
    from ..models import DatasetConfig
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from models import DatasetConfig

logger = logging.getLogger(__name__)


class CSVProcessor:
    """
    Processes CSV files and extracts base fields according to configuration.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize CSV processor with configuration.
        
        Args:
            config: DatasetConfig with field mappings
        """
        self.config = config
        self.encoding = 'utf-8'
    
    def load_csv(self, csv_path: str, batch_size: int = 1000) -> pd.DataFrame:
        """
        Load CSV file with UTF-8 encoding.
        
        Args:
            csv_path: Path to CSV file
            batch_size: Batch size for chunk processing (not used for small files)
        
        Returns:
            DataFrame with CSV data
        """
        logger.info(f"Loading CSV from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, encoding=self.encoding)
            logger.info(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
            logger.info(f"  Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """
        Validate that CSV has all required columns.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If validation fails
        """
        csv_columns = set(df.columns)
        
        # Get required columns from config
        required_columns = [
            mapping.csv_column for mapping in self.config.field_mappings 
            if mapping.required
        ]
        
        missing_columns = []
        for col in required_columns:
            # Check if column or any of its aliases exist
            mapping = next((m for m in self.config.field_mappings if m.csv_column == col), None)
            if mapping:
                found = col in csv_columns or any(alias in csv_columns for alias in mapping.aliases)
                if not found:
                    missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info("✓ CSV structure validation passed")
        return True
    
    def extract_fields(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract base fields from CSV row according to field mappings.
        
        Args:
            row: DataFrame row (Series)
        
        Returns:
            Dictionary with mapped fields
        """
        extracted = {}
        
        for mapping in self.config.field_mappings:
            # Try to find the column (check main name and aliases)
            column_value = None
            
            if mapping.csv_column in row.index:
                column_value = row[mapping.csv_column]
            else:
                # Try aliases
                for alias in mapping.aliases:
                    if alias in row.index:
                        column_value = row[alias]
                        break
            
            if column_value is not None:
                # Clean and convert the value
                cleaned_value = self._clean_value(column_value, mapping.data_type)
                extracted[mapping.metadata_key] = cleaned_value
        
        return extracted
    
    def _clean_value(self, value: Any, data_type: str) -> Any:
        """
        Clean and convert value based on data type.
        
        Args:
            value: Raw value from CSV
            data_type: Target data type
        
        Returns:
            Cleaned value
        """
        # Handle NaN/None
        if pd.isna(value):
            if data_type == 'numeric':
                return 0
            elif data_type == 'boolean':
                return False
            else:
                return None
        
        # Clean based on data type
        if data_type == 'string' or data_type == 'text':
            value_str = str(value).strip()
            
            # Check for empty or placeholder values
            if not value_str or value_str.lower() in ['nan', 'null', 'none', '-', 'n/a', 'na']:
                return None
            
            return value_str
        
        elif data_type == 'numeric':
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert '{value}' to numeric, using 0")
                    return 0
        
        elif data_type == 'boolean':
            if isinstance(value, bool):
                return value
            value_str = str(value).lower()
            return value_str in ['true', '1', 'yes', 'y', 'ใช่']
        
        elif data_type == 'date':
            # Return as string, will be converted later by date_utils
            return str(value).strip()
        
        else:
            # Default: return as-is
            return value
    
    def get_csv_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for CSV.
        
        Args:
            df: DataFrame to summarize
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'null_counts': df.isnull().sum().to_dict(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return summary

