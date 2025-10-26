"""
Customer utilities for tier mapping and customer type classification.

Handles Thai customer types and maps them to standardized tiers.
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def map_customer_tier(
    customer_type: str,
    customer_tiers: Dict[str, List[str]]
) -> str:
    """
    Map customer type to standardized tier (tier_1, tier_2, tier_3).
    
    Args:
        customer_type: Thai customer type name
        customer_tiers: Dictionary mapping tier names to customer type lists
    
    Returns:
        Tier name (tier_1/tier_2/tier_3) or "unknown"
    
    Examples:
        >>> tiers = {"tier_1": ["เอเย่นต์", "Super Sub"]}
        >>> map_customer_tier("เอเย่นต์", tiers)
        'tier_1'
    """
    if not customer_type:
        return "unknown"
    
    customer_type_stripped = customer_type.strip()
    
    # Check each tier's customer types
    for tier, customer_types in customer_tiers.items():
        if customer_type_stripped in customer_types:
            return tier
    
    logger.warning(f"Unknown customer type: '{customer_type}' - defaulting to 'unknown'")
    return "unknown"


def is_wholesale(customer_tier: str) -> bool:
    """
    Check if customer tier is wholesale.
    
    Args:
        customer_tier: Customer tier (tier_1/tier_2/tier_3)
    
    Returns:
        True if wholesale tier (tier_1 or tier_2)
    
    Examples:
        >>> is_wholesale("tier_1")
        True
        
        >>> is_wholesale("tier_3")
        False
    """
    return customer_tier in ["tier_1", "tier_2"]


def is_retail(customer_tier: str) -> bool:
    """
    Check if customer tier is retail.
    
    Args:
        customer_tier: Customer tier (tier_1/tier_2/tier_3)
    
    Returns:
        True if retail tier (tier_3)
    
    Examples:
        >>> is_retail("tier_3")
        True
        
        >>> is_retail("tier_1")
        False
    """
    return customer_tier == "tier_3"


def normalize_customer_type(customer_type: str) -> str:
    """
    Normalize customer type name (trim, standardize).
    
    Args:
        customer_type: Raw customer type
    
    Returns:
        Normalized customer type
    
    Examples:
        >>> normalize_customer_type("  เอเย่นต์  ")
        'เอเย่นต์'
    """
    if not customer_type:
        return ""
    
    # Remove extra whitespace
    normalized = ' '.join(customer_type.split())
    
    return normalized


def get_customer_type_description(customer_tier: str) -> str:
    """
    Get human-readable description of customer tier.
    
    Args:
        customer_tier: Customer tier
    
    Returns:
        Description string
    
    Examples:
        >>> get_customer_type_description("tier_1")
        'Tier 1: Premium Wholesale'
    """
    descriptions = {
        "tier_1": "Tier 1: Premium Wholesale",
        "tier_2": "Tier 2: Standard Wholesale",
        "tier_3": "Tier 3: Retail",
        "unknown": "Unknown Tier"
    }
    
    return descriptions.get(customer_tier, "Unknown Tier")

