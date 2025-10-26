"""
Data models for sales promotion documents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class CampaignInfo:
    """Campaign information."""
    campaign_id: str
    campaign_name: str
    campaign_type: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: str = "active"
    budget: Optional[float] = None
    channels: List[str] = field(default_factory=list)
    target_audience: Optional[str] = None


@dataclass
class ProductInfo:
    """Product information."""
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    sku: Optional[str] = None
    price: Optional[float] = None
    discount_percentage: Optional[float] = None
    final_price: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for campaigns."""
    impressions: Optional[int] = None
    clicks: Optional[int] = None
    ctr: Optional[float] = None
    conversions: Optional[int] = None
    conversion_rate: Optional[float] = None
    revenue: Optional[float] = None
    roi: Optional[float] = None
    cost_per_click: Optional[float] = None
    cost_per_acquisition: Optional[float] = None
    engagement_rate: Optional[float] = None
    bounce_rate: Optional[float] = None
    average_order_value: Optional[float] = None


@dataclass
class GeographicInfo:
    """Geographic information."""
    region: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    store_location: Optional[str] = None
    distribution_channel: Optional[str] = None


@dataclass
class SalesPromotionDocument:
    """
    Complete sales promotion document with all metadata.
    """
    # Core identifiers
    doc_id: str

    # Campaign information
    campaign: CampaignInfo

    # Product information
    products: List[ProductInfo] = field(default_factory=list)

    # Performance metrics
    performance: Optional[PerformanceMetrics] = None

    # Geographic data
    geographic: Optional[GeographicInfo] = None

    # Raw content
    content: str = ""

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "campaign": {
                "campaign_id": self.campaign.campaign_id,
                "campaign_name": self.campaign.campaign_name,
                "campaign_type": self.campaign.campaign_type,
                "start_date": self.campaign.start_date,
                "end_date": self.campaign.end_date,
                "status": self.campaign.status,
                "budget": self.campaign.budget,
                "channels": self.campaign.channels,
                "target_audience": self.campaign.target_audience,
            },
            "products": [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "subcategory": p.subcategory,
                    "brand": p.brand,
                    "sku": p.sku,
                    "price": p.price,
                    "discount_percentage": p.discount_percentage,
                    "final_price": p.final_price,
                }
                for p in self.products
            ],
            "performance": {
                "impressions": self.performance.impressions if self.performance else None,
                "clicks": self.performance.clicks if self.performance else None,
                "ctr": self.performance.ctr if self.performance else None,
                "conversions": self.performance.conversions if self.performance else None,
                "conversion_rate": self.performance.conversion_rate if self.performance else None,
                "revenue": self.performance.revenue if self.performance else None,
                "roi": self.performance.roi if self.performance else None,
            } if self.performance else {},
            "geographic": {
                "region": self.geographic.region if self.geographic else None,
                "country": self.geographic.country if self.geographic else None,
                "city": self.geographic.city if self.geographic else None,
            } if self.geographic else {},
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "source_file": self.source_file,
        }
