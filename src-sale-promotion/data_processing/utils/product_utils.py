"""
Product utilities for category extraction, brand detection, and size parsing.

Handles Thai product names and extracts structured information.
"""

import re
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


def extract_category(
    product_name: str,
    product_patterns: Dict[str, List[str]]
) -> str:
    """
    Extract product category from product name using pattern matching.
    
    Args:
        product_name: Thai product name
        product_patterns: Dictionary mapping categories to pattern lists
    
    Returns:
        Category name (water/beer/spirits/soda) or "other"
    
    Examples:
        >>> patterns = {"water": ["น้ำดื่ม", "PET"]}
        >>> extract_category("น้ำดื่มช้าง PET 600 CC", patterns)
        'water'
    """
    if not product_name:
        return "other"
    
    product_name_lower = product_name.lower()
    
    # Check each category's patterns
    for category, patterns in product_patterns.items():
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in product_name_lower:
                return category
    
    return "other"


def extract_brand(product_name: str, brand_patterns: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Extract brand name from product name.
    
    Args:
        product_name: Thai product name
        brand_patterns: Optional dictionary mapping brands to pattern lists
    
    Returns:
        Brand name or empty string if not found
    
    Examples:
        >>> extract_brand("น้ำดื่มช้าง PET 600 CC")
        'ช้าง'
    """
    if not product_name:
        return ""
    
    # Default brand patterns (can be overridden by config)
    if brand_patterns is None:
        brand_patterns = {
            "ช้าง": ["ช้าง", "Chang", "chang"],
            "นิยมไทย": ["นิยมไทย", "Niyom Thai"],
            "ซาโต้": ["ซาโต้", "SATO", "Sato", "sato"],
            "เลโอ": ["เลโอ", "LEO", "Leo"],
        }
    
    product_name_lower = product_name.lower()
    
    # Check each brand's patterns
    for brand, patterns in brand_patterns.items():
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in product_name_lower:
                return brand
    
    return ""


def extract_size(product_name: str) -> Optional[str]:
    """
    Extract size/volume information from product name.
    
    Args:
        product_name: Thai product name
    
    Returns:
        Size string (e.g., "600 CC", "1500 ml") or None
    
    Examples:
        >>> extract_size("น้ำดื่มช้าง PET 600 CC")
        '600 CC'
        
        >>> extract_size("เบียร์ช้าง 330 มล.")
        '330 มล.'
    """
    if not product_name:
        return None
    
    # Pattern for size extraction
    size_patterns = [
        r'(\d+)\s*(CC|cc)',  # 600 CC
        r'(\d+)\s*(มล\.?|ml|ML)',  # 330 มล.
        r'(\d+)\s*(L|l|ลิตร)',  # 1.5 L
        r'(\d+)\s*(gram|g|กรัม)',  # 250 g
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, product_name)
        if match:
            # Return full matched string
            return match.group(0)
    
    return None


def normalize_product_name(product_name: str) -> str:
    """
    Normalize product name by removing extra whitespace and formatting.
    
    Args:
        product_name: Raw product name
    
    Returns:
        Normalized product name
    
    Examples:
        >>> normalize_product_name("  น้ำดื่มช้าง   PET 600 CC  ")
        'น้ำดื่มช้าง PET 600 CC'
    """
    if not product_name:
        return ""
    
    # Remove extra whitespace
    normalized = ' '.join(product_name.split())
    
    # Handle list format if present: "[item1, item2]"
    if normalized.startswith('[') and normalized.endswith(']'):
        # Extract first item from list
        normalized = normalized[1:-1].split(',')[0].strip()
    
    return normalized


def generate_product_keywords(
    product_name: str,
    category: str,
    brand: str,
    size: Optional[str]
) -> List[str]:
    """
    Generate searchable keywords from product information.
    
    Args:
        product_name: Product name
        category: Product category
        brand: Product brand
        size: Product size
    
    Returns:
        List of keywords for search
    
    Examples:
        >>> generate_product_keywords("น้ำดื่มช้าง PET 600", "water", "ช้าง", "600 CC")
        ['น้ำดื่มช้าง', 'PET', '600', 'water', 'ช้าง']
    """
    keywords = []
    
    # Add major words from product name (Thai)
    if product_name:
        # Split by whitespace and common separators
        words = re.split(r'[\s\-\(\)]+', product_name)
        keywords.extend([w for w in words if w and len(w) > 1])
    
    # Add category
    if category and category != "other":
        keywords.append(category)
    
    # Add brand
    if brand:
        keywords.append(brand)
    
    # Add size numbers
    if size:
        # Extract numbers from size
        numbers = re.findall(r'\d+', size)
        keywords.extend(numbers)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower not in seen:
            seen.add(keyword_lower)
            unique_keywords.append(keyword)
    
    return unique_keywords


def extract_pack_info(product_name: str) -> Optional[Dict[str, str]]:
    """
    Extract pack/package information from product name.
    
    Args:
        product_name: Product name
    
    Returns:
        Dictionary with pack_size and pack_unit, or None
    
    Examples:
        >>> extract_pack_info("น้ำดื่มช้าง (1 แพ็ค เท่ากับ 12 ขวด)")
        {'pack_size': '12', 'pack_unit': 'ขวด'}
    """
    if not product_name:
        return None
    
    # Pattern: (1 แพ็ค เท่ากับ 12 ขวด)
    pattern = r'\(.*?(\d+)\s*(ขวด|แพ็ค|ลัง|ถาด|กระป๋อง).*?\)'
    match = re.search(pattern, product_name)
    
    if match:
        return {
            'pack_size': match.group(1),
            'pack_unit': match.group(2)
        }
    
    return None

