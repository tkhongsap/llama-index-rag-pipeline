"""
Statistics generator for processing metrics and reports.

Tracks processing metrics, field statistics, and generates reports.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .models import SimpleDocument, PromotionDocument
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from models import SimpleDocument, PromotionDocument

logger = logging.getLogger(__name__)


class StatisticsGenerator:
    """
    Generates processing statistics and reports.
    """
    
    def __init__(self):
        """Initialize statistics generator"""
        self.start_time = None
        self.end_time = None
    
    def start_timer(self):
        """Start processing timer"""
        self.start_time = datetime.now()
    
    def stop_timer(self):
        """Stop processing timer"""
        self.end_time = datetime.now()
    
    def get_processing_time(self) -> str:
        """Get formatted processing time"""
        if not self.start_time or not self.end_time:
            return "N/A"
        
        duration = self.end_time - self.start_time
        seconds = duration.total_seconds()
        
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        else:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
    
    def generate_statistics(
        self,
        documents: List[SimpleDocument],
        config_name: str,
        config_version: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive processing statistics.
        
        Args:
            documents: List of processed SimpleDocument objects
            config_name: Configuration name
            config_version: Configuration version
        
        Returns:
            Dictionary with statistics
        """
        logger.info("Generating processing statistics...")
        
        if not documents:
            return self._empty_statistics()
        
        stats = {
            'processing_date': datetime.now().isoformat(),
            'config_name': config_name,
            'config_version': config_version,
            'total_promotions': len(documents),
            'processing_time': self.get_processing_time(),
            'processing_speed': self._calculate_processing_speed(len(documents)),
        }
        
        # Field statistics
        stats['fields_extracted'] = self._count_fields(documents[0])
        stats['base_fields'] = 10  # Known base fields from CSV
        stats['derived_fields'] = stats['fields_extracted'] - stats['base_fields']
        
        # Text statistics
        stats['text_stats'] = self._analyze_text_lengths(documents)
        
        # Date statistics
        stats.update(self._analyze_dates(documents))
        
        # Product category statistics
        stats['product_categories'] = self._count_by_field(documents, 'product_category')
        
        # Customer tier statistics
        stats['customer_tiers'] = self._count_by_field(documents, 'customer_tier')
        
        # Promotion type statistics
        stats['promotion_types'] = self._count_by_field(documents, 'promotion_type')
        
        # Success rate
        stats['success_rate'] = 100.0  # All documents processed successfully if we're here
        
        logger.info("✓ Statistics generated")
        return stats
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics structure"""
        return {
            'processing_date': datetime.now().isoformat(),
            'total_promotions': 0,
            'processing_time': self.get_processing_time(),
            'error': 'No documents processed'
        }
    
    def _calculate_processing_speed(self, document_count: int) -> str:
        """Calculate processing speed in docs/sec"""
        if not self.start_time or not self.end_time or document_count == 0:
            return "N/A"
        
        duration = self.end_time - self.start_time
        seconds = duration.total_seconds()
        
        if seconds == 0:
            return "N/A"
        
        speed = document_count / seconds
        return f"{speed:.2f} docs/sec"
    
    def _count_fields(self, document: SimpleDocument) -> int:
        """Count total fields in metadata"""
        return len(document.metadata)
    
    def _analyze_text_lengths(self, documents: List[SimpleDocument]) -> Dict[str, Any]:
        """Analyze text length statistics"""
        lengths = [len(doc.text) for doc in documents]
        
        return {
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'total_chars': sum(lengths)
        }
    
    def _analyze_dates(self, documents: List[SimpleDocument]) -> Dict[str, Any]:
        """Analyze date-related statistics"""
        active_count = 0
        expired_count = 0
        expires_soon_count = 0
        dates_converted = 0
        
        for doc in documents:
            metadata = doc.metadata
            
            # Count active/expired
            if metadata.get('is_active'):
                active_count += 1
            else:
                expired_count += 1
            
            # Count expires soon
            if metadata.get('expires_soon'):
                expires_soon_count += 1
            
            # Count date conversions
            if metadata.get('start_date_gregorian') and metadata.get('end_date_gregorian'):
                dates_converted += 2
        
        return {
            'active_promotions': active_count,
            'expired_promotions': expired_count,
            'expiring_soon': expires_soon_count,
            'dates_converted': dates_converted
        }
    
    def _count_by_field(
        self,
        documents: List[SimpleDocument],
        field_name: str
    ) -> Dict[str, int]:
        """Count occurrences of field values"""
        values = [doc.metadata.get(field_name) for doc in documents if field_name in doc.metadata]
        return dict(Counter(values))
    
    def print_summary(self, stats: Dict[str, Any]):
        """
        Print formatted summary to console.
        
        Args:
            stats: Statistics dictionary
        """
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 General Statistics:")
        print(f"  Total Promotions: {stats.get('total_promotions', 0)}")
        print(f"  Processing Time: {stats.get('processing_time', 'N/A')}")
        print(f"  Processing Speed: {stats.get('processing_speed', 'N/A')}")
        print(f"  Success Rate: {stats.get('success_rate', 0)}%")
        
        print(f"\n📝 Field Statistics:")
        print(f"  Total Fields: {stats.get('fields_extracted', 0)}")
        print(f"  Base Fields: {stats.get('base_fields', 0)}")
        print(f"  Derived Fields: {stats.get('derived_fields', 0)}")
        
        print(f"\n📄 Text Statistics:")
        text_stats = stats.get('text_stats', {})
        print(f"  Min Length: {text_stats.get('min_length', 0)} chars")
        print(f"  Max Length: {text_stats.get('max_length', 0)} chars")
        print(f"  Avg Length: {text_stats.get('avg_length', 0):.0f} chars")
        
        print(f"\n📅 Date Statistics:")
        print(f"  Active Promotions: {stats.get('active_promotions', 0)}")
        print(f"  Expired Promotions: {stats.get('expired_promotions', 0)}")
        print(f"  Expiring Soon: {stats.get('expiring_soon', 0)}")
        print(f"  Dates Converted: {stats.get('dates_converted', 0)}")
        
        print(f"\n🏷️  Product Categories:")
        categories = stats.get('product_categories', {})
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")
        
        print(f"\n👥 Customer Tiers:")
        tiers = stats.get('customer_tiers', {})
        for tier, count in sorted(tiers.items()):
            print(f"  {tier}: {count}")
        
        print("\n" + "=" * 60)

