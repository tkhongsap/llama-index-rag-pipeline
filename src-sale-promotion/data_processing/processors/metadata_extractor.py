"""
Metadata extractor for deriving rich metadata from base fields.

Derives 15+ metadata fields including dates, categories, tiers, and calculations.
"""

from datetime import datetime
from typing import Dict, Any
import logging
from pathlib import Path

# Handle both relative and absolute imports
try:
    from ..models import DatasetConfig
    from ..utils import date_utils, product_utils, customer_utils
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from models import DatasetConfig
    from utils import date_utils, product_utils, customer_utils

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts and derives comprehensive metadata from base promotion fields.
    
    Derives:
    - Customer metadata (3 fields): tier, is_wholesale, is_retail
    - Date metadata (5 fields): gregorian dates, is_active, expires_soon, duration
    - Product metadata (3 fields): category, brand, size
    - Promotion metadata (2 fields): discount_percentage, type
    - Searchability (1 field): keywords
    
    Total: 15+ derived fields
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize metadata extractor with configuration.
        
        Args:
            config: DatasetConfig with product patterns and customer tiers
        """
        self.config = config
        self.date_threshold_days = 7  # For expires_soon flag
        
        # Load config sections
        raw_config = getattr(config, '_raw_config', {})
        self.brand_patterns = raw_config.get('brand_patterns', {})
        self.wholesale_types = raw_config.get('wholesale_types', [])
        self.retail_types = raw_config.get('retail_types', [])
    
    def derive_metadata(self, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive all metadata fields from base fields.
        
        Args:
            base_metadata: Dictionary with base fields from CSV
        
        Returns:
            Dictionary with derived metadata fields
        """
        derived = {}
        
        # Customer metadata
        derived.update(self._derive_customer_metadata(base_metadata))
        
        # Date metadata
        derived.update(self._derive_date_metadata(base_metadata))
        
        # Product metadata
        derived.update(self._derive_product_metadata(base_metadata))
        
        # Promotion metadata
        derived.update(self._derive_promotion_metadata(base_metadata))
        
        # Searchability metadata
        derived.update(self._derive_searchability_metadata(base_metadata, derived))
        
        return derived
    
    def _derive_customer_metadata(self, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Derive customer tier and type flags (3 fields)"""
        customer_type = base_metadata.get('customer_type', '')
        
        # Map to tier
        customer_tier = customer_utils.map_customer_tier(
            customer_type,
            self.config.customer_tiers
        )
        
        # Determine wholesale/retail flags
        is_wholesale_val = customer_utils.is_wholesale(customer_tier)
        is_retail_val = customer_utils.is_retail(customer_tier)
        
        return {
            'customer_tier': customer_tier,
            'is_wholesale': is_wholesale_val,
            'is_retail': is_retail_val
        }
    
    def _derive_date_metadata(self, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Derive date-related metadata (5 fields)"""
        start_date_str = base_metadata.get('start_date', '')
        end_date_str = base_metadata.get('end_date', '')
        
        # Convert Buddhist dates to Gregorian
        start_date_gregorian = date_utils.convert_buddhist_to_gregorian(start_date_str)
        end_date_gregorian = date_utils.convert_buddhist_to_gregorian(end_date_str)
        
        # Calculate derived fields
        is_active = False
        expires_soon = False
        duration_days = 0
        
        if start_date_gregorian and end_date_gregorian:
            is_active = date_utils.is_promotion_active(start_date_gregorian, end_date_gregorian)
            expires_soon = date_utils.check_expires_soon(
                end_date_gregorian,
                threshold_days=self.date_threshold_days
            )
            duration_days = date_utils.calculate_duration_days(start_date_gregorian, end_date_gregorian)
        
        return {
            'start_date_gregorian': start_date_gregorian,
            'end_date_gregorian': end_date_gregorian,
            'is_active': is_active,
            'expires_soon': expires_soon,
            'duration_days': duration_days
        }
    
    def _derive_product_metadata(self, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Derive product category, brand, and size (3 fields)"""
        product_name = base_metadata.get('product_name', '')
        
        # Normalize product name
        product_name_normalized = product_utils.normalize_product_name(product_name)
        
        # Extract category
        category = product_utils.extract_category(
            product_name_normalized,
            self.config.product_patterns
        )
        
        # Extract brand
        brand = product_utils.extract_brand(product_name_normalized, self.brand_patterns)
        
        # Extract size
        size = product_utils.extract_size(product_name_normalized)
        
        return {
            'product_category': category,
            'product_brand': brand,
            'product_size': size
        }
    
    def _derive_promotion_metadata(self, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Derive promotion type and discount percentage (2 fields)"""
        buy_quantity = base_metadata.get('buy_quantity', 0)
        free_quantity = base_metadata.get('free_quantity', 0)
        
        # Calculate discount percentage
        discount_percentage = self._calculate_discount_percentage(buy_quantity, free_quantity)
        
        # Set promotion type (all are buy_x_get_y for this dataset)
        promotion_type = "buy_x_get_y"
        
        return {
            'discount_percentage': discount_percentage,
            'promotion_type': promotion_type
        }
    
    def _derive_searchability_metadata(
        self,
        base_metadata: Dict[str, Any],
        derived_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Derive searchability keywords (1 field)"""
        product_name = base_metadata.get('product_name', '')
        category = derived_metadata.get('product_category', '')
        brand = derived_metadata.get('product_brand', '')
        size = derived_metadata.get('product_size')
        customer_type = base_metadata.get('customer_type', '')
        
        # Generate keywords
        keywords = product_utils.generate_product_keywords(
            product_name,
            category,
            brand,
            size
        )
        
        # Add customer type to keywords
        if customer_type:
            keywords.append(customer_type)
        
        return {
            'keywords': keywords
        }
    
    def _calculate_discount_percentage(self, buy_quantity: int, free_quantity: int) -> float:
        """
        Calculate effective discount percentage.
        
        Formula: (free_quantity / (buy_quantity + free_quantity)) * 100
        
        Args:
            buy_quantity: Minimum purchase quantity
            free_quantity: Free item quantity
        
        Returns:
            Discount percentage (0-100)
        
        Examples:
            >>> _calculate_discount_percentage(108, 42)
            28.0
        """
        if buy_quantity == 0 and free_quantity == 0:
            return 0.0
        
        total_quantity = buy_quantity + free_quantity
        if total_quantity == 0:
            return 0.0
        
        discount = (free_quantity / total_quantity) * 100
        return round(discount, 2)

