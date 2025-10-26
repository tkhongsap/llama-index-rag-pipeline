"""
Utility modules for sales promotion data processing.

This package contains utility functions for:
- Date conversion (Buddhist → Gregorian)
- Product categorization and extraction
- Customer tier mapping
"""

from .date_utils import (
    convert_buddhist_to_gregorian,
    is_promotion_active,
    calculate_duration_days,
    check_expires_soon
)

from .product_utils import (
    extract_category,
    extract_brand,
    extract_size,
    normalize_product_name
)

from .customer_utils import (
    map_customer_tier,
    is_wholesale,
    is_retail
)

__all__ = [
    # Date utilities
    'convert_buddhist_to_gregorian',
    'is_promotion_active',
    'calculate_duration_days',
    'check_expires_soon',
    
    # Product utilities
    'extract_category',
    'extract_brand',
    'extract_size',
    'normalize_product_name',
    
    # Customer utilities
    'map_customer_tier',
    'is_wholesale',
    'is_retail',
]

