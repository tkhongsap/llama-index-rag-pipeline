"""
Data models for sales promotion data processing.

This module defines the core data structures used in Phase 1 data preparation:
- SimpleDocument: Lightweight document class (no LlamaIndex dependency)
- PromotionDocument: Dataclass for promotion records with derived metadata
- FieldMapping: Configuration for CSV field mappings
- DatasetConfig: Configuration for the dataset schema
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class FieldMapping:
    """Configuration for mapping CSV columns to metadata fields"""
    csv_column: str
    metadata_key: str
    field_type: str  # 'categorical', 'temporal', 'numeric', 'text'
    data_type: str = 'string'  # 'string', 'numeric', 'date', 'boolean'
    required: bool = False
    transform: Optional[str] = None  # e.g., 'buddhist_to_gregorian'
    aliases: List[str] = None  # Alternative column names
    description: str = ""  # Field description for better understanding
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset schema"""
    name: str
    description: str
    version: str
    field_mappings: List[FieldMapping]
    product_patterns: Dict[str, List[str]] = None
    customer_tiers: Dict[str, List[str]] = None
    text_template: str = None  # Custom text generation template
    
    def __post_init__(self):
        if self.product_patterns is None:
            self.product_patterns = {}
        if self.customer_tiers is None:
            self.customer_tiers = {}


class SimpleDocument:
    """
    Lightweight document class for Phase 1 data preparation.
    
    NO LlamaIndex dependency - Phase 2 will convert to LlamaIndex Documents.
    This class represents a processed promotion document with text and metadata.
    """
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        """
        Initialize SimpleDocument.
        
        Args:
            text: Natural language description (50-200 tokens)
            metadata: Dictionary with 15+ fields
        """
        self.text = text
        self.metadata = metadata
        self.id = metadata.get('document_id', self._generate_id())
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content"""
        content = f"{self.text}{str(self.metadata)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """
        Serialize to dict for JSONL output.
        
        Returns:
            Dict with id, text, metadata fields for Phase 2 consumption
        """
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata
        }
    
    def to_markdown(self) -> str:
        """
        Generate markdown representation for human review.
        
        Returns:
            Markdown formatted string
        """
        lines = [
            f"# Promotion: {self.id}",
            "",
            "## Description",
            "",
            self.text,
            "",
            "## Metadata",
            ""
        ]
        
        # Group metadata by categories for better readability
        categorized = self._categorize_metadata()
        
        for category, fields in categorized.items():
            if fields:
                lines.append(f"### {category}")
                lines.append("")
                for key, value in fields.items():
                    lines.append(f"- **{key}**: {value}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _categorize_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Categorize metadata for better markdown organization"""
        categories = {
            'Basic Information': {},
            'Customer Information': {},
            'Product Information': {},
            'Promotion Terms': {},
            'Dates & Status': {},
            'Derived Metadata': {}
        }
        
        basic_keys = ['document_id']
        customer_keys = ['customer_type', 'customer_tier', 'is_wholesale', 'is_retail']
        product_keys = ['product_name', 'product_category', 'product_brand', 'product_size']
        terms_keys = ['buy_quantity', 'buy_unit', 'free_item_name', 'free_quantity', 
                      'free_unit', 'discount_percentage', 'promotion_type']
        date_keys = ['start_date', 'end_date', 'start_date_gregorian', 'end_date_gregorian',
                     'is_active', 'expires_soon', 'duration_days']
        
        for key, value in self.metadata.items():
            if key in basic_keys:
                categories['Basic Information'][key] = value
            elif key in customer_keys:
                categories['Customer Information'][key] = value
            elif key in product_keys:
                categories['Product Information'][key] = value
            elif key in terms_keys:
                categories['Promotion Terms'][key] = value
            elif key in date_keys:
                categories['Dates & Status'][key] = value
            else:
                categories['Derived Metadata'][key] = value
        
        return categories


@dataclass
class PromotionDocument:
    """
    Dataclass representing a sales promotion with base and derived metadata.
    
    This is an intermediate representation that gets converted to SimpleDocument.
    Includes 10 base fields from CSV + 15+ derived metadata fields.
    """
    
    # Base fields from CSV (10 fields)
    document_id: str
    customer_type: str
    start_date: str  # Buddhist calendar (original)
    end_date: str    # Buddhist calendar (original)
    product_name: str
    buy_quantity: int
    buy_unit: str
    free_item_name: str
    free_quantity: int
    free_unit: str
    additional_conditions: Optional[str] = None
    
    # Derived metadata - Customer (3 fields)
    customer_tier: str = ""
    is_wholesale: bool = False
    is_retail: bool = False
    
    # Derived metadata - Dates (5 fields)
    start_date_gregorian: Optional[datetime] = None
    end_date_gregorian: Optional[datetime] = None
    is_active: bool = False
    expires_soon: bool = False
    duration_days: int = 0
    
    # Derived metadata - Product (3 fields)
    product_category: str = "other"
    product_brand: str = ""
    product_size: Optional[str] = None
    
    # Derived metadata - Promotion (2 fields)
    discount_percentage: float = 0.0
    promotion_type: str = "buy_x_get_y"
    
    # Derived metadata - Searchability (1 field)
    keywords: List[str] = field(default_factory=list)
    
    def to_simple_document(self) -> SimpleDocument:
        """
        Convert to SimpleDocument for Phase 1 output.
        
        Phase 2 will convert SimpleDocument → LlamaIndex Document.
        
        Returns:
            SimpleDocument object with text + metadata
        """
        text = self._generate_text()
        metadata = self.to_metadata_dict()
        
        return SimpleDocument(text=text, metadata=metadata)
    
    def _generate_text(self) -> str:
        """
        Generate natural language text description.
        
        Returns:
            Human-readable promotion description (50-200 tokens)
        """
        # Format dates for display
        start_str = self.start_date_gregorian.strftime('%Y-%m-%d') if self.start_date_gregorian else 'N/A'
        end_str = self.end_date_gregorian.strftime('%Y-%m-%d') if self.end_date_gregorian else 'N/A'
        
        # Build the main promotion text
        text_parts = []
        
        # Main promotion offer
        text_parts.append(
            f"Promotion for {self.customer_type} customers: "
            f"Buy {self.buy_quantity} {self.buy_unit} of {self.product_name} "
            f"and get {self.free_quantity} {self.free_unit} of {self.free_item_name} FREE."
        )
        
        # Promotion period and status
        text_parts.append("")
        text_parts.append(
            f"Promotion Period: Valid from {start_str} to {end_str} ({self.duration_days} days)."
        )
        status_text = 'Active ✓' if self.is_active else 'Expired ✗'
        text_parts.append(f"Status: {status_text}")
        if self.expires_soon:
            text_parts.append("⚠️ Expires soon (within 7 days)")
        
        # Promotion terms
        text_parts.append("")
        text_parts.append("Promotion Terms:")
        text_parts.append(f"- Purchase minimum: {self.buy_quantity} {self.buy_unit}")
        text_parts.append(f"- Free items: {self.free_quantity} {self.free_unit}")
        text_parts.append(f"- Effective discount: {self.discount_percentage:.1f}% off total quantity")
        
        customer_type_label = 'Wholesale' if self.is_wholesale else 'Retail'
        text_parts.append(f"- Customer tier: {self.customer_tier} ({customer_type_label})")
        
        # Product information
        text_parts.append("")
        text_parts.append("Product Information:")
        text_parts.append(f"- Product: {self.product_name}")
        text_parts.append(f"- Category: {self.product_category}")
        if self.product_brand:
            text_parts.append(f"- Brand: {self.product_brand}")
        if self.product_size:
            text_parts.append(f"- Size: {self.product_size}")
        
        # Additional conditions
        text_parts.append("")
        if self.additional_conditions:
            text_parts.append(f"Additional Conditions: {self.additional_conditions}")
        else:
            text_parts.append("No additional conditions.")
        
        return "\n".join(text_parts)
    
    def to_metadata_dict(self) -> dict:
        """
        Generate complete metadata dictionary.
        
        Returns:
            All fields as dict for SimpleDocument metadata
        """
        return {
            # Base fields
            "document_id": self.document_id,
            "customer_type": self.customer_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "product_name": self.product_name,
            "buy_quantity": self.buy_quantity,
            "buy_unit": self.buy_unit,
            "free_item_name": self.free_item_name,
            "free_quantity": self.free_quantity,
            "free_unit": self.free_unit,
            "additional_conditions": self.additional_conditions,
            
            # Derived metadata - Customer
            "customer_tier": self.customer_tier,
            "is_wholesale": self.is_wholesale,
            "is_retail": self.is_retail,
            
            # Derived metadata - Dates
            "start_date_gregorian": self.start_date_gregorian.strftime('%Y-%m-%d') if self.start_date_gregorian else None,
            "end_date_gregorian": self.end_date_gregorian.strftime('%Y-%m-%d') if self.end_date_gregorian else None,
            "is_active": self.is_active,
            "expires_soon": self.expires_soon,
            "duration_days": self.duration_days,
            
            # Derived metadata - Product
            "product_category": self.product_category,
            "product_brand": self.product_brand,
            "product_size": self.product_size,
            
            # Derived metadata - Promotion
            "discount_percentage": self.discount_percentage,
            "promotion_type": self.promotion_type,
            
            # Derived metadata - Searchability
            "keywords": self.keywords
        }

