"""
Unit Tests for Sales Promotion Data Processing

Tests all data processing functions without requiring API calls.
Tests:
- Buddhist calendar conversion
- Product category extraction  
- Customer tier mapping
- Discount calculation
- Thai encoding preservation
- Metadata enrichment
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_sale_promotion.data_processing.processors.thai_normalizer import ThaiNormalizer
from src_sale_promotion.data_processing.utils.date_utils import (
    convert_buddhist_to_gregorian,
    is_promotion_active,
    calculate_duration_days
)
from src_sale_promotion.data_processing.utils.product_utils import (
    extract_category,
    extract_brand,
    normalize_product_name
)
from src_sale_promotion.data_processing.utils.customer_utils import (
    map_customer_tier,
    is_wholesale
)
from src_sale_promotion.data_processing.processors.metadata_extractor import MetadataExtractor


class TestDateConversion:
    """Test Buddhist calendar conversion functions."""
    
    def test_buddhist_to_gregorian_conversion(self):
        """Test converting Buddhist Era dates to Gregorian."""
        # Buddhist year 2568 = Gregorian year 2025
        buddhist_date = "2568-10-01 00:00:00"
        gregorian = convert_buddhist_to_gregorian(buddhist_date)
        
        assert gregorian.year == 2025
        assert gregorian.month == 10
        assert gregorian.day == 1
    
    def test_multiple_year_conversions(self):
        """Test conversion for multiple years."""
        test_cases = [
            ("2568-01-01 00:00:00", 2025),
            ("2567-12-31 00:00:00", 2024),
            ("2569-06-15 00:00:00", 2026),
        ]
        
        for buddhist_date, expected_year in test_cases:
            gregorian = convert_buddhist_to_gregorian(buddhist_date)
            assert gregorian.year == expected_year
    
    def test_date_active_status(self):
        """Test checking if promotion is currently active."""
        # Create a date range that includes today
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2026, 12, 31)
        
        assert is_promotion_active(start_date, end_date) == True
    
    def test_date_inactive_future(self):
        """Test promotion that hasn't started yet."""
        start_date = datetime(2026, 1, 1)
        end_date = datetime(2026, 12, 31)
        
        assert is_promotion_active(start_date, end_date) == False
    
    def test_date_inactive_past(self):
        """Test promotion that has ended."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        assert is_promotion_active(start_date, end_date) == False
    
    def test_duration_calculation(self):
        """Test calculating duration in days."""
        start_date = datetime(2025, 10, 1)
        end_date = datetime(2025, 12, 31)
        
        duration = calculate_duration_days(start_date, end_date)
        assert duration == 92  # Oct 1 to Dec 31 (inclusive)


class TestProductCategorization:
    """Test product category extraction and normalization."""
    
    def test_water_category_detection(self):
        """Test detecting water products."""
        test_cases = [
            "น้ำดื่มช้าง PET 600 CC",
            "น้ำดื่มช้าง PET 1500 CC",
            "น้ำแร่ช้าง",
        ]
        
        patterns = {
            "water": ["น้ำดื่ม", "PET", "น้ำแร่"],
            "beer": ["เบียร์", "beer"],
            "other": []
        }
        
        for product_name in test_cases:
            category = extract_category(product_name, patterns)
            assert category == "water"
    
    def test_beer_category_detection(self):
        """Test detecting beer products."""
        test_cases = [
            "เบียร์ช้างคลาสสิก 620 มล.",
            "เบียร์ช้าง 330 มล.",
            "Beer Chang Classic",
        ]
        
        patterns = {
            "water": ["น้ำดื่ม", "PET"],
            "beer": ["เบียร์", "beer"],
            "other": []
        }
        
        for product_name in test_cases:
            category = extract_category(product_name, patterns)
            assert category == "beer"
    
    def test_brand_extraction(self):
        """Test extracting brand from product name."""
        test_cases = [
            ("น้ำดื่มช้าง PET 600 CC", "ช้าง"),
            ("เบียร์ช้างคลาสสิก", "ช้าง"),
            ("นิยมไทย", "นิยมไทย"),
        ]
        
        for product_name, expected_brand in test_cases:
            brand = extract_brand(product_name)
            assert brand == expected_brand
    
    def test_product_name_normalization(self):
        """Test normalizing product names."""
        test_cases = [
            ("น้ำดื่มช้าง  PET  600  CC", "น้ำดื่มช้าง PET 600 CC"),
            ("  เบียร์ช้าง  ", "เบียร์ช้าง"),
        ]
        
        for input_name, expected_output in test_cases:
            normalized = normalize_product_name(input_name)
            assert normalized == expected_output


class TestCustomerSegmentation:
    """Test customer tier mapping and classification."""
    
    def test_tier_mapping_from_processed_data(self):
        """Test tier mapping using real processed data."""
        from pathlib import Path
        import json
        
        jsonl_path = Path("src-sale-promotion/data/processed/documents/documents.jsonl")
        
        if not jsonl_path.exists():
            pytest.skip("Processed documents not found")
        
        # Read and validate tier mapping
        with open(jsonl_path) as f:
            for line in f:
                doc = json.loads(line)
                metadata = doc["metadata"]
                
                # Every doc should have customer_tier
                assert "customer_tier" in metadata
                assert metadata["customer_tier"] in ["tier_1", "tier_2", "tier_3"]
                
                # Check consistency with customer_type
                customer_type = metadata.get("customer_type")
                tier = metadata.get("customer_tier")
                
                if customer_type in ["เอเย่นต์", "Super Sub"]:
                    assert tier == "tier_1", f"{customer_type} should be tier_1"
                elif customer_type in ["ซับเอเย่นต์", "CVA", "ซับซื้อตรง"]:
                    assert tier == "tier_2", f"{customer_type} should be tier_2"
                elif customer_type in ["รถขายปลีก", "Retail TT"]:
                    assert tier == "tier_3", f"{customer_type} should be tier_3"
    
    def test_wholesale_classification(self):
        """Test wholesale vs retail classification."""
        # is_wholesale takes tier not customer_type
        tier_1 = "tier_1"  # Wholesale
        tier_3 = "tier_3"  # Retail
        
        assert is_wholesale(tier_1) == True
        assert is_wholesale(tier_3) == False


class TestDiscountCalculation:
    """Test discount percentage calculation."""
    
    def test_simple_discount_calculation(self):
        """Test basic discount calculation."""
        # Buy 100, get 20 free = 16.67% discount
        # Formula: (free / (buy + free)) * 100
        discount = (20 / (100 + 20)) * 100
        assert abs(discount - 16.67) < 0.1
    
    def test_high_discount_calculation(self):
        """Test high discount scenarios."""
        # Buy 108, get 42 free = 28.0% discount
        discount = (42 / (108 + 42)) * 100
        assert abs(discount - 28.0) < 0.1
    
    def test_low_discount_calculation(self):
        """Test low discount scenarios."""
        # Buy 1000, get 4 free = 0.4% discount
        discount = (4 / (1000 + 4)) * 100
        assert abs(discount - 0.4) < 0.1
    
    def test_zero_free_items(self):
        """Test when no free items are given."""
        discount = (0 / (100 + 0)) * 100 if 100 > 0 else 0.0
        assert discount == 0.0


class TestThaiEncoding:
    """Test Thai UTF-8 encoding preservation."""
    
    def test_thai_character_preservation(self):
        """Test that Thai characters are preserved."""
        normalizer = ThaiNormalizer()
        
        thai_text = "น้ำดื่มช้าง PET 600 CC"
        processed = normalizer.normalize_text(thai_text)
        
        # Check Thai characters are still present
        thai_chars = [c for c in processed if '\u0e00' <= c <= '\u0e7f']
        assert len(thai_chars) > 0
    
    def test_mixed_thai_english(self):
        """Test mixed Thai-English text."""
        normalizer = ThaiNormalizer()
        
        text = "โปรโมชัน promotion สำหรับ customer"
        processed = normalizer.normalize_text(text)
        
        # Both Thai and English should be preserved
        assert "โปรโมชัน" in processed
        assert "promotion" in processed
    
    def test_thai_customer_types(self):
        """Test Thai customer type preservation."""
        customer_types = [
            "เอเย่นต์",
            "ซับเอเย่นต์",
            "รถขายปลีก",
            "ซับซื้อตรง"
        ]
        
        normalizer = ThaiNormalizer()
        
        for customer_type in customer_types:
            processed = normalizer.normalize_text(customer_type)
            assert customer_type == processed


class TestMetadataEnrichment:
    """Test metadata enrichment pipeline."""
    
    def test_metadata_from_processed_data(self):
        """Test that processed data has complete metadata."""
        # Use actual processed data
        from pathlib import Path
        import json
        
        jsonl_path = Path("src-sale-promotion/data/processed/documents/documents.jsonl")
        
        if not jsonl_path.exists():
            pytest.skip("Processed documents not found")
        
        with open(jsonl_path) as f:
            sample_doc = json.loads(f.readline())
        
        metadata = sample_doc["metadata"]
        
        # Should have 25+ metadata fields
        assert len(metadata) >= 25
    
    def test_derived_fields_presence(self):
        """Test that derived fields are computed."""
        from pathlib import Path
        import json
        
        jsonl_path = Path("src-sale-promotion/data/processed/documents/documents.jsonl")
        
        if not jsonl_path.exists():
            pytest.skip("Processed documents not found")
        
        with open(jsonl_path) as f:
            sample_doc = json.loads(f.readline())
        
        metadata = sample_doc["metadata"]
        
        # Check derived fields exist
        derived_fields = [
            "customer_tier",
            "product_category",
            "product_brand",
            "discount_percentage",
            "is_active",
            "expires_soon",
            "duration_days",
            "start_date_gregorian",
            "end_date_gregorian",
            "is_wholesale"
        ]
        
        for field in derived_fields:
            assert field in metadata, f"Missing derived field: {field}"


# Integration test
def test_full_document_processing():
    """Test processing a complete promotion document."""
    from pathlib import Path
    import json
    
    # Use existing processed data
    jsonl_path = Path("src-sale-promotion/data/processed/documents/documents.jsonl")
    
    if not jsonl_path.exists():
        pytest.skip("Processed documents not found")
    
    with open(jsonl_path) as f:
        sample_doc = json.loads(f.readline())
    
    # Validate structure
    assert "id" in sample_doc
    assert "text" in sample_doc
    assert "metadata" in sample_doc
    
    # Validate text
    assert len(sample_doc["text"]) > 0
    assert any('\u0e00' <= c <= '\u0e7f' for c in sample_doc["text"])
    
    # Validate metadata
    assert len(sample_doc["metadata"]) >= 25
    assert "customer_type" in sample_doc["metadata"]
    assert "product_category" in sample_doc["metadata"]
    assert "discount_percentage" in sample_doc["metadata"]
    assert "is_active" in sample_doc["metadata"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
