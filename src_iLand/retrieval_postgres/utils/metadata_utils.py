"""
Metadata utilities for PostgreSQL retrieval

Handles JSONB metadata operations, filtering, and Thai-specific metadata processing.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class MetadataUtils:
    """Utilities for handling metadata operations in PostgreSQL."""
    
    # Thai provinces for geographic filtering
    THAI_PROVINCES = [
        "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
        "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ",
        "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก",
        "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์",
        "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี",
        "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
        "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
        "แพร่", "พะเยา", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
        "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี",
        "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล",
        "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
        "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์",
        "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี",
        "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี"
    ]
    
    # Thai land deed types
    THAI_DEED_TYPES = [
        "โฉนด", "นส.3", "นส.3ก", "นส.3ข", "นส.4", "ส.ค.1", "ป.ป.36"
    ]
    
    @staticmethod
    def build_metadata_filter_sql(
        filters: Dict[str, Any],
        base_params: List[Any]
    ) -> tuple[str, List[Any]]:
        """
        Build SQL WHERE clause for metadata filtering.
        
        Args:
            filters: Dictionary of metadata filters
            base_params: Existing query parameters
            
        Returns:
            Tuple of (WHERE clause, updated parameters)
        """
        where_clauses = []
        params = base_params.copy()
        
        for key, value in filters.items():
            if isinstance(value, (list, tuple)):
                # Handle IN queries for multiple values
                placeholders = ", ".join(["%s"] * len(value))
                where_clauses.append(f"metadata_->>'{key}' IN ({placeholders})")
                params.extend([str(v) for v in value])
            elif isinstance(value, dict):
                # Handle nested JSON queries
                if "gte" in value:
                    where_clauses.append(f"(metadata_->>'{key}')::numeric >= %s")
                    params.append(value["gte"])
                if "lte" in value:
                    where_clauses.append(f"(metadata_->>'{key}')::numeric <= %s")
                    params.append(value["lte"])
                if "gt" in value:
                    where_clauses.append(f"(metadata_->>'{key}')::numeric > %s")
                    params.append(value["gt"])
                if "lt" in value:
                    where_clauses.append(f"(metadata_->>'{key}')::numeric < %s")
                    params.append(value["lt"])
                if "contains" in value:
                    where_clauses.append(f"metadata_->>'{key}' ILIKE %s")
                    params.append(f"%{value['contains']}%")
            elif value is None:
                # Handle NULL checks
                where_clauses.append(f"metadata_->>'{key}' IS NULL")
            else:
                # Handle exact match
                where_clauses.append(f"metadata_->>'{key}' = %s")
                params.append(str(value))
        
        where_sql = " AND ".join(where_clauses) if where_clauses else ""
        return where_sql, params
    
    @staticmethod
    def extract_geographic_metadata(text: str) -> Dict[str, Any]:
        """
        Extract geographic information from Thai text.
        
        Args:
            text: Thai text to analyze
            
        Returns:
            Dictionary with geographic metadata
        """
        geo_metadata = {}
        
        # Extract provinces
        found_provinces = []
        for province in MetadataUtils.THAI_PROVINCES:
            if province in text:
                found_provinces.append(province)
        
        if found_provinces:
            geo_metadata["provinces"] = found_provinces
            geo_metadata["primary_province"] = found_provinces[0]
        
        # Extract districts (อำเภอ)
        import re
        district_pattern = r"อำเภอ([ก-๙\s]+?)(?:\s|จังหวัด|$)"
        districts = re.findall(district_pattern, text)
        if districts:
            geo_metadata["districts"] = [d.strip() for d in districts]
        
        # Extract sub-districts (ตำบล)
        subdistrict_pattern = r"ตำบล([ก-๙\s]+?)(?:\s|อำเภอ|จังหวัด|$)"
        subdistricts = re.findall(subdistrict_pattern, text)
        if subdistricts:
            geo_metadata["subdistricts"] = [s.strip() for s in subdistricts]
        
        return geo_metadata
    
    @staticmethod
    def extract_deed_metadata(text: str) -> Dict[str, Any]:
        """
        Extract land deed information from Thai text.
        
        Args:
            text: Thai text to analyze
            
        Returns:
            Dictionary with deed metadata
        """
        deed_metadata = {}
        
        # Extract deed types
        found_deed_types = []
        for deed_type in MetadataUtils.THAI_DEED_TYPES:
            if deed_type in text:
                found_deed_types.append(deed_type)
        
        if found_deed_types:
            deed_metadata["deed_types"] = found_deed_types
            deed_metadata["primary_deed_type"] = found_deed_types[0]
        
        # Extract deed numbers (pattern matching)
        import re
        deed_number_patterns = [
            r"เลขที่\s*(\d+)",
            r"หมายเลข\s*(\d+)",
            r"เลขโฉนด\s*(\d+)"
        ]
        
        deed_numbers = []
        for pattern in deed_number_patterns:
            matches = re.findall(pattern, text)
            deed_numbers.extend(matches)
        
        if deed_numbers:
            deed_metadata["deed_numbers"] = deed_numbers
        
        # Extract land area information
        area_patterns = [
            r"(\d+)\s*ไร่",
            r"(\d+)\s*งาน",
            r"(\d+)\s*ตารางวา"
        ]
        
        areas = {}
        for i, unit in enumerate(["rai", "ngan", "wah"]):
            matches = re.findall(area_patterns[i], text)
            if matches:
                areas[unit] = [int(m) for m in matches]
        
        if areas:
            deed_metadata["land_areas"] = areas
        
        return deed_metadata
    
    @staticmethod
    def enrich_metadata(
        metadata: Dict[str, Any],
        text: str
    ) -> Dict[str, Any]:
        """
        Enrich existing metadata with extracted information.
        
        Args:
            metadata: Existing metadata dictionary
            text: Text to analyze for additional metadata
            
        Returns:
            Enriched metadata dictionary
        """
        enriched = metadata.copy() if metadata else {}
        
        # Extract and merge geographic metadata
        geo_metadata = MetadataUtils.extract_geographic_metadata(text)
        if geo_metadata:
            enriched.update(geo_metadata)
        
        # Extract and merge deed metadata
        deed_metadata = MetadataUtils.extract_deed_metadata(text)
        if deed_metadata:
            enriched.update(deed_metadata)
        
        # Add text statistics
        enriched["text_length"] = len(text)
        enriched["word_count"] = len(text.split())
        
        return enriched
    
    @staticmethod
    def filter_thai_provinces(province_query: str) -> List[str]:
        """
        Filter Thai provinces by partial match.
        
        Args:
            province_query: Partial province name
            
        Returns:
            List of matching provinces
        """
        query = province_query.strip()
        if not query:
            return MetadataUtils.THAI_PROVINCES
        
        matches = []
        for province in MetadataUtils.THAI_PROVINCES:
            if query in province or province.startswith(query):
                matches.append(province)
        
        return matches
    
    @staticmethod
    def validate_metadata_filters(filters: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate metadata filters and return validation errors.
        
        Args:
            filters: Metadata filters to validate
            
        Returns:
            Dictionary of validation errors by field
        """
        errors = {}
        
        for key, value in filters.items():
            field_errors = []
            
            # Validate province filters
            if key == "province" or key == "provinces":
                if isinstance(value, str):
                    if value not in MetadataUtils.THAI_PROVINCES:
                        field_errors.append(f"Invalid province: {value}")
                elif isinstance(value, list):
                    for province in value:
                        if province not in MetadataUtils.THAI_PROVINCES:
                            field_errors.append(f"Invalid province: {province}")
            
            # Validate deed type filters
            elif key == "deed_type" or key == "deed_types":
                if isinstance(value, str):
                    if value not in MetadataUtils.THAI_DEED_TYPES:
                        field_errors.append(f"Invalid deed type: {value}")
                elif isinstance(value, list):
                    for deed_type in value:
                        if deed_type not in MetadataUtils.THAI_DEED_TYPES:
                            field_errors.append(f"Invalid deed type: {deed_type}")
            
            # Validate numeric ranges
            elif isinstance(value, dict):
                for range_key in ["gte", "lte", "gt", "lt"]:
                    if range_key in value:
                        try:
                            float(value[range_key])
                        except (ValueError, TypeError):
                            field_errors.append(f"Invalid numeric value for {range_key}: {value[range_key]}")
            
            if field_errors:
                errors[key] = field_errors
        
        return errors 