import pandas as pd
import re
import hashlib
import logging
from typing import Dict, Any, List
from datetime import datetime
# Handle both relative and absolute imports
try:
    from .models import SimpleDocument, DatasetConfig
except ImportError:
    from models import SimpleDocument, DatasetConfig

# Import section parser for chunking
try:
    from ..docs_embedding_postgres.standalone_section_parser import StandaloneLandDeedSectionParser
except ImportError:
    logger.warning("Could not import StandaloneLandDeedSectionParser, section chunking will be disabled")
    StandaloneLandDeedSectionParser = None

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing, text generation, and metadata extraction"""
    
    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        
        # Initialize section parser if available
        self.section_parser = None
        if StandaloneLandDeedSectionParser:
            self.section_parser = StandaloneLandDeedSectionParser(
                chunk_size=512,
                chunk_overlap=50,
                min_section_size=50
            )
            logger.info("Section-based chunking enabled with BGE-M3 optimized settings")
    
    def clean_value(self, value: Any) -> Any:
        """Clean and normalize values for metadata"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            # Clean whitespace
            value = value.strip()
            # Normalize Thai punctuation to ASCII equivalents
            punct_map = {
                "\u2018": "'",  # ‘
                "\u2019": "'",  # ’
                "\u201c": '"',  # “
                "\u201d": '"',  # ”
                "\u2013": '-',  # –
                "\u2014": '-',  # —
                "\uff0c": ',',  # ，
                "\uff01": '!',  # ！
                "\uff1f": '?',  # ？
                "\uff1a": ':',  # ：
                "\uff1b": ';',  # ；
                "\uff0e": '.',  # ．
            }
            value = value.translate(str.maketrans(punct_map))
            # Remove multiple spaces
            value = re.sub(r'\s+', ' ', value)
            # Normalize common placeholders
            placeholders = ['nan', 'null', 'none', '-', 'n/a', 'na', 'ไม่ระบุ']
            if not value or value.lower() in placeholders or value in placeholders:
                return None
        return value
    
    def format_area_for_display(self, rai: Any, ngan: Any, wa: Any) -> str:
        """Format Thai area measurements for display"""
        parts = []
        if rai and self.clean_value(rai) is not None:
            parts.append(f"{rai} ไร่")
        if ngan and self.clean_value(ngan) is not None:
            parts.append(f"{ngan} งาน")
        if wa and self.clean_value(wa) is not None:
            parts.append(f"{wa} ตร.ว.")
        
        return " ".join(parts) if parts else "ไม่ระบุ"
    
    def _validate_coordinates(self, longitude: float, latitude: float) -> bool:
        """Return True if coordinates are within expected bounds."""
        return 0 <= longitude <= 180 and 0 <= latitude <= 90

    def _format_coordinates(self, longitude: float, latitude: float) -> Dict[str, Any]:
        return {
            'longitude': longitude,
            'latitude': latitude,
            'coordinates_formatted': f"{latitude:.6f}, {longitude:.6f}",
            'google_maps_url': f"https://www.google.com/maps?q={latitude},{longitude}"
        }

    def parse_geom_point(self, geom_point_str: str) -> Dict[str, Any]:
        """Parse coordinates from various POINT formats."""
        if not geom_point_str or not isinstance(geom_point_str, str):
            return {}

        cleaned = geom_point_str.strip()

        # Match patterns like "POINT(100.4514 14.5486)" or "POINT(100.4514,14.5486)"
        point_pattern = r'POINT\s*\(\s*([-+]?\d*\.?\d+)\s*(?:,|\s)\s*([-+]?\d*\.?\d+)\s*\)'
        match = re.search(point_pattern, cleaned, re.IGNORECASE)
        if match:
            try:
                lon = float(match.group(1))
                lat = float(match.group(2))
                if self._validate_coordinates(lon, lat):
                    return self._format_coordinates(lon, lat)
                logger.warning(f"Coordinates out of range in '{geom_point_str}'")
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse coordinates from '{geom_point_str}': {e}")
            return {}

        # Fallback for formats like "14.5486 100.4514" or "14.5486,100.4514"
        pair_pattern = r'^\s*([-+]?\d*\.?\d+)\s*(?:,|\s)\s*([-+]?\d*\.?\d+)\s*$'
        match = re.search(pair_pattern, cleaned)
        if match:
            try:
                first = float(match.group(1))
                second = float(match.group(2))
                # Assume lat lon first
                lat, lon = first, second
                if self._validate_coordinates(lon, lat):
                    return self._format_coordinates(lon, lat)
                # Try lon lat order if above failed
                lon, lat = first, second
                if self._validate_coordinates(lon, lat):
                    return self._format_coordinates(lon, lat)
                logger.warning(f"Coordinates out of range in '{geom_point_str}'")
            except ValueError as e:
                logger.warning(f"Failed to parse coordinates from '{geom_point_str}': {e}")
        return {}
    
    def extract_metadata_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata from a CSV row using the configured field mappings"""
        metadata = {}
        
        for mapping in self.dataset_config.field_mappings:
            # Try to find the column (support for aliases)
            column_value = None
            
            if mapping.csv_column in row.index:
                column_value = row[mapping.csv_column]
            else:
                # Try aliases
                for alias in mapping.aliases:
                    if alias in row.index:
                        column_value = row[alias]
                        break
            
            if column_value is not None:
                cleaned_value = self.clean_value(column_value)
                if cleaned_value is not None:
                    # Special handling for certain data types
                    if mapping.data_type == 'date' and isinstance(cleaned_value, str):
                        # Try to parse date
                        try:
                            cleaned_value = pd.to_datetime(cleaned_value).strftime('%Y-%m-%d')
                        except:
                            pass
                    elif mapping.data_type == 'boolean':
                        cleaned_value = str(cleaned_value).lower() in ['true', '1', 'yes', 'y']
                    
                    metadata[mapping.metadata_key] = cleaned_value
        
        # Add computed metadata
        self._add_computed_metadata(metadata, row)
        
        return metadata
    
    def _add_computed_metadata(self, metadata: Dict[str, Any], row: pd.Series):
        """Add computed/derived metadata fields"""
        
        # Create a searchable text field combining key attributes
        search_text_parts = []
        
        # Add location hierarchy
        location_parts = []
        for loc_field in ['province', 'district', 'subdistrict']:
            if loc_field in metadata and metadata[loc_field]:
                location_parts.append(metadata[loc_field])
        
        if location_parts:
            metadata['location_hierarchy'] = " > ".join(location_parts)
            search_text_parts.append(metadata['location_hierarchy'])
        
        # Parse geolocation from land_geom_point if available
        if 'land_geom_point' in metadata and metadata['land_geom_point']:
            geom_data = self.parse_geom_point(metadata['land_geom_point'])
            if geom_data:
                metadata.update(geom_data)
                search_text_parts.append(f"พิกัด: {geom_data.get('coordinates_formatted', '')}")
        
        # Add formatted area
        area_rai = metadata.get('area_rai')
        area_ngan = metadata.get('area_ngan')
        area_wa = metadata.get('area_wa')
        
        if any([area_rai, area_ngan, area_wa]):
            metadata['area_formatted'] = self.format_area_for_display(area_rai, area_ngan, area_wa)
            search_text_parts.append(metadata['area_formatted'])
        
        # Calculate total area in square meters if possible
        if area_rai or area_ngan or area_wa:
            try:
                total_sqm = 0
                if area_rai:
                    total_sqm += float(area_rai) * 1600  # 1 rai = 1600 sqm
                if area_ngan:
                    total_sqm += float(area_ngan) * 400   # 1 ngan = 400 sqm
                if area_wa:
                    total_sqm += float(area_wa) * 4       # 1 sq wa = 4 sqm
                metadata['area_total_sqm'] = total_sqm
            except:
                pass
        
        # Add deed type to search text
        if 'deed_type' in metadata and metadata['deed_type']:
            search_text_parts.append(f"โฉนด{metadata['deed_type']}")
        
        # Add land use to search text
        if 'land_use_type' in metadata and metadata['land_use_type']:
            search_text_parts.append(metadata['land_use_type'])
        
        # Create combined search text
        if search_text_parts:
            metadata['search_text'] = " | ".join(search_text_parts)
    
    def generate_document_text(self, row: pd.Series, metadata: Dict[str, Any]) -> str:
        """Generate document text content for land deed records"""
        
        # Group metadata by field types
        field_groups = {
            'identifier': [],
            'deed_info': [],
            'location': [],
            'geolocation': [],
            'land_details': [],
            'area_measurements': [],
            'dates': [],
            'classification': [],
            'financial': [],
            'other': []
        }
        
        # Map metadata back to field types with Thai labels
        field_labels = {
            'deed_id': 'รหัสโฉนด',
            'land_id': 'รหัสที่ดิน',
            'deed_type': 'ประเภทโฉนด',
            'deed_holding_type': 'ประเภทการถือครอง',
            'deed_serial_no': 'เลขที่โฉนด',
            'deed_book_no': 'เล่มที่',
            'deed_page_no': 'หน้าที่',
            'province': 'จังหวัด',
            'district': 'อำเภอ',
            'subdistrict': 'ตำบล',
            'land_geom_point': 'พิกัดภูมิศาสตร์',
            'longitude': 'ลองจิจูด',
            'latitude': 'ละติจูด',
            'coordinates_formatted': 'พิกัด',
            'land_use_type': 'ประเภทการใช้ที่ดิน',
            'area_formatted': 'เนื้อที่',
            'area_total_sqm': 'พื้นที่รวม (ตร.ม.)',
            'owner_date': 'วันที่ได้มา',
            'land_main_category': 'หมวดหมู่หลัก',
            'land_sub_category': 'หมวดหมู่ย่อย'
        }
        
        # Process each field mapping
        for mapping in self.dataset_config.field_mappings:
            if mapping.metadata_key in metadata:
                value = metadata[mapping.metadata_key]
                # Use Thai label if available, otherwise use formatted English
                label = field_labels.get(mapping.metadata_key, 
                                       mapping.metadata_key.replace('_', ' ').title())
                formatted_field = f"{label}: {value}"
                field_groups[mapping.field_type].append(formatted_field)
        
        # Add computed fields
        if 'location_hierarchy' in metadata:
            field_groups['location'].insert(0, f"ที่ตั้ง: {metadata['location_hierarchy']}")
        
        if 'area_formatted' in metadata:
            field_groups['area_measurements'].insert(0, f"เนื้อที่: {metadata['area_formatted']}")
        
        # Build text sections
        sections = {}
        
        # Process each section with proper formatting
        section_builders = {
            'deed_info': self._build_deed_section,
            'location': self._build_location_section,
            'geolocation': self._build_geolocation_section,
            'land_details': self._build_land_details_section,
            'area_measurements': self._build_area_section,
            'dates': self._build_dates_section,
            'classification': self._build_classification_section,
            'financial': self._build_financial_section,
            'other': self._build_other_section
        }
        
        for section_key, builder_func in section_builders.items():
            if field_groups[section_key]:
                sections[f"{section_key}_section"] = builder_func(field_groups[section_key], metadata)
        
        # Generate final text using template or default format
        if self.dataset_config.text_template:
            try:
                # Fill in template sections
                template_data = {key: value for key, value in sections.items()}
                # Add empty strings for missing sections
                for section in ['deed_info_section', 'location_section', 'geolocation_section', 'land_details_section',
                              'area_section', 'classification_section', 'dates_section',
                              'additional_section']:
                    if section not in template_data:
                        template_data[section] = "ไม่มีข้อมูล"
                
                return self.dataset_config.text_template.format(**template_data)
            except Exception as e:
                logger.warning(f"Failed to use template: {e}")
        
        # Fallback to structured text
        return self._generate_structured_text(sections, metadata)
    
    def _build_deed_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build deed information section"""
        return "\n".join(f"- {field}" for field in fields)
    
    def _build_location_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build location section with hierarchy and coordinates"""
        location_fields = fields.copy()
        
        # Add geolocation information if available
        if 'longitude' in metadata and 'latitude' in metadata:
            location_fields.append(f"พิกัด: {metadata['coordinates_formatted']}")
            location_fields.append(f"ลองจิจูด: {metadata['longitude']}")
            location_fields.append(f"ละติจูด: {metadata['latitude']}")
        
        return "\n".join(f"- {field}" for field in location_fields)
    
    def _build_geolocation_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build geolocation section with coordinates and map links"""
        geolocation_fields = fields.copy()
        
        # Add formatted coordinate information if available
        if 'longitude' in metadata and 'latitude' in metadata:
            if not any('พิกัด:' in field for field in geolocation_fields):
                geolocation_fields.append(f"พิกัด: {metadata['coordinates_formatted']}")
            if not any('ลองจิจูด:' in field for field in geolocation_fields):
                geolocation_fields.append(f"ลองจิจูด: {metadata['longitude']}")
            if not any('ละติจูด:' in field for field in geolocation_fields):
                geolocation_fields.append(f"ละติจูด: {metadata['latitude']}")
            if 'google_maps_url' in metadata:
                geolocation_fields.append(f"ลิงก์แผนที่: {metadata['google_maps_url']}")
        
        return "\n".join(f"- {field}" for field in geolocation_fields) if geolocation_fields else "ไม่มีข้อมูล"
    
    def _build_land_details_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build land details section"""
        return "\n".join(f"- {field}" for field in fields)
    
    def _build_area_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build area measurements section"""
        return "\n".join(f"- {field}" for field in fields)
    
    def _build_dates_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build dates section"""
        return "\n".join(f"- {field}" for field in fields)
    
    def _build_classification_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build classification section"""
        return "\n".join(f"- {field}" for field in fields)
    
    def _build_financial_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build financial section"""
        return "\n".join(f"- {field}" for field in fields)
    
    def _build_other_section(self, fields: List[str], metadata: Dict[str, Any]) -> str:
        """Build other information section"""
        return "\n".join(f"- {field}" for field in fields)
    
    def _generate_structured_text(self, sections: Dict[str, str], metadata: Dict[str, Any]) -> str:
        """Generate structured text when template fails"""
        text_parts = []
        text_parts.append("# บันทึกข้อมูลโฉนดที่ดิน (Land Deed Record)")
        text_parts.append("")
        
        # Add search summary if available
        if 'search_text' in metadata:
            text_parts.append(f"**สรุป:** {metadata['search_text']}")
            text_parts.append("")
        
        section_titles = {
            'deed_info_section': 'ข้อมูลโฉนด (Deed Information)',
            'location_section': 'ที่ตั้ง (Location)',
            'geolocation_section': 'พิกัดภูมิศาสตร์ (Geolocation)',
            'land_details_section': 'รายละเอียดที่ดิน (Land Details)',
            'area_section': 'ขนาดพื้นที่ (Area Measurements)',
            'classification_section': 'การจำแนกประเภท (Classification)',
            'dates_section': 'วันที่สำคัญ (Important Dates)',
            'financial_section': 'ข้อมูลการเงิน (Financial Information)',
            'other_section': 'ข้อมูลเพิ่มเติม (Additional Information)'
        }
        
        for section_key, section_content in sections.items():
            if section_content:
                title = section_titles.get(section_key, section_key.replace('_', ' ').title())
                text_parts.append(f"## {title}")
                text_parts.append(section_content)
                text_parts.append("")
        
        return "\n".join(text_parts)
    
    def convert_row_to_document(self, row: pd.Series, row_index: int = 0) -> SimpleDocument:
        """Convert a single CSV row to a Document"""
        
        # Extract metadata using configuration
        metadata = self.extract_metadata_from_row(row)
        
        # Generate document text
        text_content = self.generate_document_text(row, metadata)
        
        # Add processing metadata
        metadata['source'] = 'iland_csv_import'
        metadata['created_at'] = datetime.now().isoformat()
        metadata['doc_type'] = 'land_deed_record'
        metadata['config_name'] = self.dataset_config.name
        metadata['row_index'] = row_index
        
        # Create unique document ID
        doc_id = self._create_document_id(metadata)
        metadata['doc_id'] = doc_id
        
        return SimpleDocument(text=text_content, metadata=metadata)
    
    def _create_document_id(self, metadata: Dict[str, Any]) -> str:
        """Create a unique document ID based on key fields"""
        # Try to use existing IDs first
        if 'deed_id' in metadata and metadata['deed_id']:
            return f"deed_{metadata['deed_id']}"
        elif 'land_id' in metadata and metadata['land_id']:
            return f"land_{metadata['land_id']}"
        else:
            # Create ID from location and other attributes
            id_parts = []
            for field in ['province', 'district', 'deed_type', 'deed_serial_no']:
                if field in metadata and metadata[field]:
                    id_parts.append(str(metadata[field])[:10])
            
            if id_parts:
                id_string = "_".join(id_parts)
                return f"iland_{hashlib.md5(id_string.encode()).hexdigest()[:12]}"
            else:
                # Fallback to timestamp-based ID
                return f"iland_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:16]}"
    
    def generate_section_chunks(self, document: SimpleDocument) -> List[Dict[str, Any]]:
        """
        Generate section-based chunks from a document for BGE-M3 embedding.
        
        Returns list of chunks with metadata ready for embedding.
        """
        if not self.section_parser:
            logger.warning("Section parser not available, returning single document")
            return [{
                'text': document.text,
                'metadata': document.metadata,
                'section_type': 'full_document',
                'section_name': 'Complete Document',
                'chunk_type': 'fallback',
                'is_primary': True,
                'chunk_index': 0,
                'total_chunks': 1
            }]
        
        # Parse document into sections
        sections = self.section_parser.parse_document_to_sections(
            document_text=document.text,
            metadata=document.metadata
        )
        
        chunks = []
        for idx, section_node in enumerate(sections):
            # Extract section metadata
            section_metadata = section_node.metadata
            
            chunk_data = {
                'text': section_node.text,
                'deed_id': document.metadata.get('deed_id'),
                'section_type': section_metadata.get('section', 'unknown'),
                'section_name': section_metadata.get('original_header', section_metadata.get('section', 'unknown')),
                'chunk_type': section_metadata.get('chunk_type', 'section'),
                'is_primary': section_metadata.get('is_primary_chunk', False),
                'chunk_index': idx,
                'total_chunks': len(sections),
                'metadata': {
                    **document.metadata,  # Include all document metadata
                    **section_metadata    # Add section-specific metadata
                }
            }
            chunks.append(chunk_data)
        
        # Log chunking statistics
        stats = self.section_parser.get_chunking_statistics(sections)
        logger.info(f"Document {document.metadata.get('deed_id', 'unknown')} chunked into {len(chunks)} sections")
        logger.info(f"Chunk types: {stats['chunk_types']}")
        logger.info(f"Sections: {stats['sections']}")
        
        return chunks 