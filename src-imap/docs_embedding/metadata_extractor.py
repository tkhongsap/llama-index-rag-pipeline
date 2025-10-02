"""
Metadata extraction module for iLand Thai land deed documents.
Handles structured metadata extraction from markdown files.
"""

import re
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime


class iLandMetadataExtractor:
    """Extracts structured metadata from iLand Thai land deed markdown files."""
    
    def __init__(self):
        # Metadata patterns for iLand Thai land deed records
        self.metadata_patterns = {
            # Deed Information
            'deed_serial_no': r'Deed Serial No[:\s]*([^\n]+)',
            'deed_type': r'Deed Type[:\s]*([^\n]+)',
            'deed_book_no': r'Deed Book No[:\s]*([^\n]+)',
            'deed_page_no': r'Deed Page No[:\s]*([^\n]+)',
            'deed_surface_no': r'Deed Surface No[:\s]*([^\n]+)',
            'deed_holding_type': r'Deed Holding Type[:\s]*([^\n]+)',
            'deed_group_type': r'Deed Group Type[:\s]*([^\n]+)',
            
            # Location Information
            'province': r'Province[:\s]*([^\n]+)',
            'district': r'District[:\s]*([^\n]+)',
            'country': r'Country[:\s]*([^\n]+)',
            'region': r'Region[:\s]*([^\n]+)',
            'location_hierarchy': r'Location Hierarchy[:\s]*([^\n]+)',
            
            # Land Details
            'land_name': r'Land Name[:\s]*([^\n]+)',
            'land_passport': r'Land Passport[:\s]*([^\n]+)',
            'land_main_category': r'Land Main Category[:\s]*([^\n]+)',
            'land_sub_category': r'Land Sub Category[:\s]*([^\n]+)',
            'is_condo': r'Is Condo[:\s]*([^\n]+)',
            
            # Area Measurements (Thai units)
            'deed_rai': r'Deed Rai[:\s]*([0-9.]+)',
            'deed_ngan': r'Deed Ngan[:\s]*([0-9.]+)',
            'deed_wa': r'Deed Wa[:\s]*([0-9.]+)',
            'deed_total_square_wa': r'Deed Total Square Wa[:\s]*([0-9.]+)',
            'deed_square_meter': r'Deed Square Meter[:\s]*([0-9.]+)',
            'land_rai': r'Land Rai[:\s]*([0-9.]+)',
            'land_ngan': r'Land Ngan[:\s]*([0-9.]+)',
            'land_wa': r'Land Wa[:\s]*([0-9.]+)',
            'land_total_square_wa': r'Land Total Square Wa[:\s]*([0-9.]+)',
            'land_sum_rai': r'Land Sum Rai[:\s]*([0-9.]+)',
            'land_sum_ngan': r'Land Sum Ngan[:\s]*([0-9.]+)',
            'land_sum_wa': r'Land Sum Wa[:\s]*([0-9.]+)',
            'land_sum_total_square_wa': r'Land Sum Total Square Wa[:\s]*([0-9.]+)',
            
            # Important Dates
            'owner_date': r'Owner Date[:\s]*([0-9-]+)',
            'transfer_date': r'Transfer Date[:\s]*([0-9-]+)',
            'receive_date': r'Receive Date[:\s]*([0-9-]+)',
            
            # System and Additional Information
            'doc_id': r'Doc Id[:\s]*([^\n]+)',
            'deed_id': r'Deed Id[:\s]*([^\n]+)',
            'land_id': r'Land Id[:\s]*([^\n]+)',
            'land_code': r'Land Code[:\s]*([^\n]+)',
            'deed_no': r'Deed No[:\s]*([^\n]+)',
            'deed_ravang': r'Deed Ravang[:\s]*([^\n]+)',
            'land_geom_point': r'Land Geom Point[:\s]*([^\n]+)',
            'row_index': r'Row Index[:\s]*([0-9]+)',
            'search_text': r'Search Text[:\s]*([^\n]+)',
            'source': r'Source[:\s]*([^\n]+)',
            'created_at': r'Created At[:\s]*([^\n]+)',
            'doc_type': r'Doc Type[:\s]*([^\n]+)',
            'config_name': r'Config Name[:\s]*([^\n]+)',
        }
    
    def extract_from_content(self, content: str) -> Dict[str, Any]:
        """Extract structured metadata using regex patterns."""
        metadata = {}
        
        for key, pattern in self.metadata_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                
                # Clean and convert values
                if key in ['deed_rai', 'deed_ngan', 'deed_wa', 'deed_total_square_wa', 'deed_square_meter',
                          'land_rai', 'land_ngan', 'land_wa', 'land_total_square_wa', 
                          'land_sum_rai', 'land_sum_ngan', 'land_sum_wa', 'land_sum_total_square_wa',
                          'row_index'] and value:
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = value
                elif key in ['is_condo'] and value:
                    metadata[key] = value.lower() == 'true'
                else:
                    metadata[key] = value
        
        return metadata
    
    def classify_content_types(self, content: str) -> List[str]:
        """Classify what types of content are present in the document."""
        content_types = []
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['โฉนด', 'deed', 'land deed']):
            content_types.append('land_deed')
        
        if any(term in content_lower for term in ['ที่ดิน', 'land', 'property']):
            content_types.append('land_property')
        
        if any(term in content_lower for term in ['กรรมสิทธิ์', 'ownership']):
            content_types.append('ownership')
        
        if any(term in content_lower for term in ['จังหวัด', 'province', 'อำเภอ', 'district']):
            content_types.append('location')
        
        if any(term in content_lower for term in ['ไร่', 'งาน', 'ตารางวา', 'rai', 'ngan', 'wa']):
            content_types.append('area_measurement')
        
        if any(term in content_lower for term in ['agricultural', 'ทำนา', 'ทำสวน', 'ทำไร่']):
            content_types.append('agricultural')
        
        return content_types
    
    def derive_categories(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Derive additional metadata categories for better filtering."""
        derived = {}
        
        # Area categories based on total area
        total_area = metadata.get('deed_total_square_wa', 0) or metadata.get('land_sum_total_square_wa', 0)
        if isinstance(total_area, (int, float)) and total_area > 0:
            if total_area < 400:  # Less than 1 rai
                derived['area_category'] = 'small'
            elif total_area < 1600:  # 1-4 rai
                derived['area_category'] = 'medium'
            elif total_area < 4000:  # 4-10 rai
                derived['area_category'] = 'large'
            else:
                derived['area_category'] = 'very_large'
        
        # Deed type category
        deed_type = metadata.get('deed_type', '').lower()
        if 'โฉนด' in deed_type:
            derived['deed_type_category'] = 'chanote'
        elif 'น.ส.3' in deed_type:
            derived['deed_type_category'] = 'nor_sor_3'
        elif 'น.ส.4' in deed_type:
            derived['deed_type_category'] = 'nor_sor_4'
        elif 'ส.ค.1' in deed_type:
            derived['deed_type_category'] = 'sor_kor_1'
        else:
            derived['deed_type_category'] = 'other'
        
        # Region category
        region = metadata.get('region', '').lower()
        if 'กลาง' in region:
            derived['region_category'] = 'central'
        elif 'เหนือ' in region:
            derived['region_category'] = 'north'
        elif 'ตะวันออก' in region:
            derived['region_category'] = 'east'
        elif 'ใต้' in region:
            derived['region_category'] = 'south'
        else:
            derived['region_category'] = 'other'
        
        # Land use category
        land_category = metadata.get('land_main_category', '').lower()
        if 'agricultural' in land_category:
            derived['land_use_category'] = 'agricultural'
        elif 'residential' in land_category:
            derived['land_use_category'] = 'residential'
        elif 'commercial' in land_category:
            derived['land_use_category'] = 'commercial'
        elif 'industrial' in land_category:
            derived['land_use_category'] = 'industrial'
        else:
            derived['land_use_category'] = 'other'
        
        # Ownership type
        holding_type = metadata.get('deed_holding_type', '').lower()
        if 'บริษัท' in holding_type or 'company' in holding_type:
            derived['ownership_category'] = 'corporate'
        elif 'บุคคล' in holding_type or 'individual' in holding_type:
            derived['ownership_category'] = 'individual'
        else:
            derived['ownership_category'] = 'other'
        
        return derived
    
    def extract_document_title(self, metadata: Dict[str, Any], doc_number: int = 0) -> str:
        """Extract a meaningful document title from metadata."""
        deed_type = metadata.get("deed_type", "")
        province = metadata.get("province", "")
        district = metadata.get("district", "")
        deed_no = metadata.get("deed_serial_no", "")
        
        if deed_type and province:
            title = f"{deed_type} - {province}"
            if district:
                title += f" > {district}"
            if deed_no:
                title += f" (#{deed_no})"
            return title
        
        return (
            metadata.get("file_name") or 
            metadata.get("filename") or
            metadata.get("file_path", "").split("/")[-1] or
            f"Land Deed {doc_number}"
        )
