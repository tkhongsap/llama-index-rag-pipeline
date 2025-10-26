"""
Thai text normalizer for handling Thai characters and text formatting.

Handles whitespace normalization, list extraction, and UTF-8 preservation.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ThaiNormalizer:
    """
    Normalizes Thai text for consistent processing.
    """
    
    def __init__(self):
        """Initialize Thai normalizer"""
        pass
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Thai text by cleaning whitespace and formatting.
        
        Args:
            text: Raw Thai text
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Handle list format: "[item1, item2]"
        if text.startswith('[') and text.endswith(']'):
            text = self._extract_from_list(text)
        
        return text
    
    def _extract_from_list(self, list_text: str) -> str:
        """
        Extract first item from list format.
        
        Args:
            list_text: Text in format "[item1, item2]"
        
        Returns:
            First item
        """
        # Remove brackets
        list_text = list_text[1:-1]
        
        # Split by comma and take first item
        items = list_text.split(',')
        if items:
            return items[0].strip()
        
        return ""
    
    def normalize_unit(self, unit: str) -> str:
        """
        Normalize Thai unit names.
        
        Args:
            unit: Thai unit (แพ็ค, ลัง, etc.)
        
        Returns:
            Normalized unit
        """
        if not unit:
            return ""
        
        unit = unit.strip()
        
        # Standardize common variants
        unit_mappings = {
            'แพค': 'แพ็ค',
            'ลุง': 'ลัง',
            'ขวด': 'ขวด',
            'ถาด': 'ถาด',
            'กระป๋อง': 'กระป๋อง'
        }
        
        return unit_mappings.get(unit, unit)
    
    def extract_sku_info(self, text: str) -> Optional[str]:
        """
        Extract SKU or product code information.
        
        Args:
            text: Product text
        
        Returns:
            SKU if found, None otherwise
        """
        if not text:
            return None
        
        # Pattern for SKU: typically alphanumeric codes
        sku_pattern = r'\b([A-Z]{2,}\d{3,})\b'
        match = re.search(sku_pattern, text)
        
        if match:
            return match.group(1)
        
        return None
    
    def clean_punctuation(self, text: str) -> str:
        """
        Clean and normalize Thai punctuation to ASCII equivalents.
        
        Args:
            text: Text with mixed punctuation
        
        Returns:
            Text with normalized punctuation
        """
        if not text:
            return ""
        
        # Thai punctuation to ASCII mapping
        punct_map = {
            "\u2018": "'",  # '
            "\u2019": "'",  # '
            "\u201c": '"',  # "
            "\u201d": '"',  # "
            "\u2013": '-',  # –
            "\u2014": '-',  # —
            "\uff0c": ',',  # ，
            "\uff01": '!',  # ！
            "\uff1f": '?',  # ？
            "\uff1a": ':',  # ：
            "\uff1b": ';',  # ；
            "\uff0e": '.',  # ．
        }
        
        text = text.translate(str.maketrans(punct_map))
        return text

