import pandas as pd
import json
import os
import yaml
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import warnings

# Suppress pandas date parsing warnings to reduce noise
warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed individually')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FieldMapping:
    """Configuration for mapping CSV columns to metadata fields"""
    csv_column: str
    metadata_key: str
    field_type: str  # 'identifier', 'deed_info', 'location', 'land_details', 'area_measurements', 'dates', 'classification'
    data_type: str = 'string'  # 'string', 'numeric', 'date', 'boolean'
    required: bool = False
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
    field_mappings: List[FieldMapping]
    text_template: str = None  # Custom text generation template
    embedding_fields: List[str] = None  # Fields to prioritize for embedding
    
    def __post_init__(self):
        if self.embedding_fields is None:
            # Default fields that are most important for embeddings
            self.embedding_fields = [
                'deed_type', 'province', 'district', 'land_use_type',
                'area_rai', 'land_main_category'
            ]
    
class SimpleDocument:
    """Simple document class that mimics LlamaIndex Document structure"""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.id = metadata.get('doc_id', self._generate_id())
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content"""
        content = f"{self.text}{str(self.metadata)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

class iLandCSVConverter:
    """Flexible CSV to Document converter specifically for iLand dataset"""
    
    def __init__(self, input_csv_path: str, output_dir: str, config_path: str = None):
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.config_path = config_path
        self.dataset_config = None
        self.csv_columns = None
        self.encoding = 'utf-8'  # Changed to UTF-8 for Thai text support
        self.thai_provinces = self._load_thai_provinces()
        self.ensure_output_dir()
    
    def _load_thai_provinces(self) -> List[str]:
        """Load list of Thai provinces for validation"""
        # Common Thai provinces (subset for validation)
        return [
            'กรุงเทพมหานคร', 'เชียงใหม่', 'เชียงราย', 'ภูเก็ต', 'สุราษฎร์ธานี',
            'นครราชสีมา', 'ขอนแก่น', 'อุดรธานี', 'นครศรีธรรมราช', 'สงขลา',
            'ระยอง', 'ชลบุรี', 'พัทยา', 'สมุทรปราการ', 'นนทบุรี'
        ]
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "iland_markdown_files"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
    
    def analyze_csv_structure(self) -> Dict[str, Any]:
        """Analyze CSV structure and suggest field mappings for iLand dataset"""
        logger.info(f"Analyzing iLand CSV structure: {self.input_csv_path}")
        
        # Try UTF-8 first, fallback to other encodings if needed
        encodings_to_try = ['utf-8', 'utf-8-sig', 'cp874', 'latin-1']
        sample_df = None
        used_encoding = None
        
        for encoding in encodings_to_try:
            try:
                sample_df = pd.read_csv(self.input_csv_path, nrows=200, encoding=encoding)
                used_encoding = encoding
                logger.info(f"Successfully read CSV with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with encoding: {encoding}")
                continue
            except Exception as e:
                logger.warning(f"Error reading with encoding {encoding}: {e}")
                continue
        
        if sample_df is None:
            raise ValueError(f"Could not read CSV file with any of the tried encodings: {encodings_to_try}")
        
        # Update the encoding based on what worked
        self.encoding = used_encoding
        self.csv_columns = list(sample_df.columns)
        
        # Calculate statistics
        stats = {
            'null_counts': sample_df.isnull().sum().to_dict(),
            'unique_counts': {col: sample_df[col].nunique() for col in sample_df.columns},
            'value_examples': {}
        }
        
        # Get example values for each column
        for col in sample_df.columns:
            non_null_values = sample_df[col].dropna().unique()[:5]
            stats['value_examples'][col] = list(non_null_values)
        
        analysis = {
            'total_columns': len(self.csv_columns),
            'columns': self.csv_columns,
            'suggested_mappings': self._suggest_field_mappings(sample_df),
            'data_types': dict(sample_df.dtypes.astype(str)),
            'sample_data': sample_df.head(3).to_dict('records'),
            'encoding_used': self.encoding,
            'statistics': stats
        }
        
        return analysis
    
    def _suggest_field_mappings(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Automatically suggest field mappings based on iLand dataset patterns"""
        
        # Enhanced pattern mappings with descriptions
        field_patterns = {
            'identifier': {
                'patterns': [
                    (r'deed_id$', 'Primary deed identifier'),
                    (r'land_id$', 'Land parcel identifier'),
                    (r'_no$', 'Reference number'),
                    (r'_code$', 'Reference code')
                ],
                'metadata_keys': ['deed_id', 'land_id', 'deed_no', 'land_code']
            },
            'deed_info': {
                'patterns': [
                    (r'^deed_type', 'Type of land deed document'),
                    (r'deed_holding', 'Deed holding classification'),
                    (r'deed_serial', 'Serial number of deed'),
                    (r'deed_book', 'Deed registry book number'),
                    (r'deed_page', 'Page number in deed book'),
                    (r'deed_surface', 'Surface number reference'),
                    (r'deed_note', 'Additional deed notes')
                ],
                'metadata_keys': ['deed_type', 'deed_holding_type', 'deed_serial_no', 
                                'deed_book_no', 'deed_page_no', 'deed_surface_no', 'deed_note']
            },
            'location': {
                'patterns': [
                    (r'province', 'Province location'),
                    (r'district', 'District/Amphoe'),
                    (r'subdistrict', 'Subdistrict/Tambon'),
                    (r'region', 'Regional classification'),
                    (r'country', 'Country')
                ],
                'metadata_keys': ['province', 'district', 'subdistrict', 'region', 'country']
            },
            'land_details': {
                'patterns': [
                    (r'land_name', 'Name or description of land'),
                    (r'land_use_type', 'Primary land use classification'),
                    (r'land_use_detail', 'Detailed land use description'),
                    (r'land_category', 'Land category classification'),
                    (r'land_passport', 'Land passport reference'),
                    (r'land_document', 'Related document references'),
                    (r'land_company', 'Associated company')
                ],
                'metadata_keys': ['land_name', 'land_use_type', 'land_use_detail', 
                                'land_category', 'land_passport', 'land_document_no', 'land_company_id']
            },
            'area_measurements': {
                'patterns': [
                    (r'rai', 'Area in rai (Thai unit)'),
                    (r'ngan', 'Area in ngan (Thai unit)'),
                    (r'wa', 'Area in square wa (Thai unit)'),
                    (r'square.*meter', 'Area in square meters')
                ],
                'metadata_keys': ['area_rai', 'area_ngan', 'area_wa', 'area_square_wa', 'area_square_meter']
            },
            'dates': {
                'patterns': [
                    (r'owner_date', 'Date of ownership'),
                    (r'transfer_date', 'Date of transfer'),
                    (r'receive_date', 'Date received'),
                    (r'contract.*date', 'Contract related dates')
                ],
                'metadata_keys': ['owner_date', 'transfer_date', 'receive_date', 
                                'contract_start_date', 'contract_end_date']
            },
            'classification': {
                'patterns': [
                    (r'group_type', 'Deed group classification'),
                    (r'main_category', 'Primary category'),
                    (r'sub_category', 'Secondary category'),
                    (r'is_condo', 'Condominium flag')
                ],
                'metadata_keys': ['deed_group_type', 'land_main_category', 
                                'land_sub_category', 'is_condo']
            },
            'financial': {
                'patterns': [
                    (r'revenue', 'Revenue information'),
                    (r'value', 'Valuation data'),
                    (r'price', 'Price information'),
                    (r'cost', 'Cost data')
                ],
                'metadata_keys': ['land_total_revenue', 'land_value', 'price', 'cost']
            }
        }
        
        suggestions = []
        
        for column in df.columns:
            column_lower = column.lower()
            best_match = None
            best_confidence = 0
            best_description = ""
            
            for field_type, config in field_patterns.items():
                for pattern_tuple in config['patterns']:
                    pattern, description = pattern_tuple if isinstance(pattern_tuple, tuple) else (pattern_tuple, "")
                    if re.search(pattern, column_lower):
                        # Calculate confidence based on pattern match quality
                        match = re.search(pattern, column_lower)
                        confidence = len(match.group()) / len(column_lower) if match else 0
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_description = description
                            best_match = {
                                'csv_column': column,
                                'field_type': field_type,
                                'suggested_metadata_key': self._suggest_metadata_key(column, config['metadata_keys']),
                                'confidence': confidence,
                                'data_type': self._infer_data_type(df[column]),
                                'description': description
                            }
            
            if best_match:
                suggestions.append(best_match)
            else:
                # For unmatched columns, suggest as generic metadata
                suggestions.append({
                    'csv_column': column,
                    'field_type': 'other',
                    'suggested_metadata_key': self._normalize_key(column),
                    'confidence': 0.1,
                    'data_type': self._infer_data_type(df[column]),
                    'description': 'Unclassified field'
                })
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def _suggest_metadata_key(self, column_name: str, possible_keys: List[str]) -> str:
        """Suggest the best metadata key based on column name"""
        column_lower = column_name.lower()
        
        # Try exact matches first
        for key in possible_keys:
            if key.lower() in column_lower:
                return key
        
        # Try partial matches
        for key in possible_keys:
            key_parts = key.lower().split('_')
            if all(part in column_lower for part in key_parts):
                return key
        
        # Return normalized column name as fallback
        return self._normalize_key(column_name)
    
    def _normalize_key(self, key: str) -> str:
        """Normalize a column name to a valid metadata key"""
        # Convert to lowercase, replace spaces and special chars with underscores
        normalized = re.sub(r'[^a-zA-Z0-9ก-๙]+', '_', key.lower())
        normalized = re.sub(r'_+', '_', normalized)  # Remove multiple underscores
        return normalized.strip('_')
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer the data type of a pandas Series"""
        # Check for dates first
        if series.dtype == 'object':
            # Try to parse as date
            try:
                pd.to_datetime(series.dropna().head(10))
                return 'date'
            except:
                pass
        
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32']:
                return 'integer'
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'date'
        elif series.dtype == 'bool':
            return 'boolean'
        else:
            return 'string'
    
    def create_config_from_analysis(self, analysis: Dict[str, Any], config_name: str) -> DatasetConfig:
        """Create a dataset configuration from CSV analysis"""
        
        field_mappings = []
        
        for suggestion in analysis['suggested_mappings']:
            mapping = FieldMapping(
                csv_column=suggestion['csv_column'],
                metadata_key=suggestion['suggested_metadata_key'],
                field_type=suggestion['field_type'],
                data_type=suggestion['data_type'],
                required=suggestion['confidence'] > 0.5,  # Medium confidence fields are required
                description=suggestion.get('description', '')
            )
            field_mappings.append(mapping)
        
        # Enhanced text template for land deed documents
        text_template = """# ข้อมูลโฉนดที่ดิน (Land Deed Record)

## ข้อมูลโฉนด (Deed Information)
{deed_info_section}

## ที่ตั้ง (Location)
{location_section}

## รายละเอียดที่ดิน (Land Details)
{land_details_section}

## ขนาดพื้นที่ (Area Measurements)
{area_section}

## การจำแนกประเภท (Classification)
{classification_section}

## วันที่สำคัญ (Important Dates)
{dates_section}

## ข้อมูลเพิ่มเติม (Additional Information)
{additional_section}"""
        
        # Define priority fields for embeddings
        embedding_fields = [
            'deed_type', 'province', 'district', 'subdistrict',
            'land_use_type', 'land_main_category', 'area_rai',
            'deed_group_type'
        ]
        
        config = DatasetConfig(
            name=config_name,
            description=f"Configuration for iLand dataset - Thai land deed records with {len(field_mappings)} fields",
            field_mappings=field_mappings,
            text_template=text_template,
            embedding_fields=embedding_fields
        )
        
        return config
    
    def save_config(self, config: DatasetConfig, config_path: str = None):
        """Save dataset configuration to YAML file"""
        if config_path is None:
            config_path = os.path.join(self.output_dir, f"{config.name}_config.yaml")
        
        # Convert to dict for serialization
        config_dict = {
            'name': config.name,
            'description': config.description,
            'field_mappings': [asdict(mapping) for mapping in config.field_mappings],
            'text_template': config.text_template,
            'embedding_fields': config.embedding_fields
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration saved to: {config_path}")
        return config_path
    
    def load_config(self, config_path: str) -> DatasetConfig:
        """Load dataset configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        field_mappings = [
            FieldMapping(**mapping) for mapping in config_dict['field_mappings']
        ]
        
        config = DatasetConfig(
            name=config_dict['name'],
            description=config_dict['description'],
            field_mappings=field_mappings,
            text_template=config_dict.get('text_template'),
            embedding_fields=config_dict.get('embedding_fields', [])
        )
        
        return config
    
    def setup_configuration(self, config_name: str = None, auto_generate: bool = True) -> DatasetConfig:
        """Setup configuration for the CSV conversion"""
        
        if self.config_path and os.path.exists(self.config_path):
            logger.info(f"Loading existing configuration: {self.config_path}")
            self.dataset_config = self.load_config(self.config_path)
        elif auto_generate:
            logger.info("Auto-generating configuration from iLand CSV analysis")
            analysis = self.analyze_csv_structure()
            
            if config_name is None:
                config_name = "iland_deed_records"
            
            self.dataset_config = self.create_config_from_analysis(analysis, config_name)
            
            # Save the generated config for future use
            config_path = self.save_config(self.dataset_config)
            logger.info(f"Generated configuration saved to: {config_path}")
            
            # Also save the analysis report
            self._save_analysis_report(analysis)
        else:
            raise ValueError("No configuration provided and auto_generate is False")
        
        return self.dataset_config
    
    def _save_analysis_report(self, analysis: Dict[str, Any]):
        """Save CSV analysis report"""
        report_path = os.path.join(self.output_dir, "reports", "iland_csv_analysis_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis report saved to: {report_path}")
    
    def clean_value(self, value: Any) -> Any:
        """Clean and normalize values for metadata"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            # Clean whitespace
            value = value.strip()
            # Remove multiple spaces
            value = re.sub(r'\s+', ' ', value)
            # Return None for empty strings
            if not value or value.lower() in ['nan', 'null', 'none', '-']:
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
                for section in ['deed_info_section', 'location_section', 'land_details_section',
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
        """Build location section with hierarchy"""
        return "\n".join(f"- {field}" for field in fields)
    
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
        metadata['encoding_used'] = self.encoding
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
    
    def process_csv_to_documents(self, batch_size: int = 1000) -> List[SimpleDocument]:
        """Process entire CSV file and convert to documents"""
        
        if self.dataset_config is None:
            raise ValueError("Configuration not set up. Call setup_configuration() first.")
        
        logger.info(f"Starting conversion using config: {self.dataset_config.name}")
        logger.info(f"Using encoding: {self.encoding}")
        logger.info(f"Priority fields for embedding: {self.dataset_config.embedding_fields}")
        
        documents = []
        chunk_num = 0
        total_rows = 0
        failed_rows = []
        start_time = datetime.now()
        
        for chunk in pd.read_csv(self.input_csv_path, chunksize=batch_size, encoding=self.encoding):
            chunk_num += 1
            chunk_start_time = datetime.now()
            
            logger.info(f"Processing chunk {chunk_num} ({len(chunk)} rows) - Total processed so far: {total_rows:,}")
            
            chunk_failed = 0
            for idx, row in chunk.iterrows():
                total_rows += 1
                try:
                    document = self.convert_row_to_document(row, row_index=idx)
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {e}")
                    failed_rows.append({'row': idx, 'error': str(e)})
                    chunk_failed += 1
            
            # Chunk completion info
            chunk_time = (datetime.now() - chunk_start_time).total_seconds()
            success_rate = ((len(chunk) - chunk_failed) / len(chunk)) * 100
            elapsed_total = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✓ Chunk {chunk_num} completed in {chunk_time:.1f}s - Success: {success_rate:.1f}% - Total time: {elapsed_total/60:.1f}min")
            
            # Progress summary every 10 chunks
            if chunk_num % 10 == 0:
                avg_time_per_chunk = elapsed_total / chunk_num
                logger.info(f"📊 Progress: {chunk_num} chunks, {len(documents):,} documents, avg {avg_time_per_chunk:.1f}s/chunk")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Processing completed in {total_time/60:.1f} minutes")
        
        # Save error report if there were failures
        if failed_rows:
            error_report_path = os.path.join(self.output_dir, "reports", "conversion_errors.json")
            with open(error_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_failures': len(failed_rows),
                    'failures': failed_rows[:100]  # Save first 100 errors
                }, f, indent=2, ensure_ascii=False)
            logger.warning(f"Failed to process {len(failed_rows)} rows. See {error_report_path}")
        
        logger.info(f"Successfully converted {len(documents)} out of {total_rows} rows to documents")
        
        # Generate summary statistics
        self._generate_conversion_summary(documents)
        
        return documents
    
    def _generate_conversion_summary(self, documents: List[SimpleDocument]):
        """Generate summary statistics for the conversion"""
        summary = {
            'total_documents': len(documents),
            'conversion_date': datetime.now().isoformat(),
            'configuration': self.dataset_config.name,
            'field_coverage': {},
            'location_distribution': {},
            'deed_type_distribution': {},
            'area_statistics': {
                'min_area_sqm': float('inf'),
                'max_area_sqm': 0,
                'avg_area_sqm': 0
            }
        }
        
        # Analyze field coverage
        field_counts = {}
        province_counts = {}
        deed_type_counts = {}
        area_values = []
        
        for doc in documents:
            # Count field coverage
            for field in self.dataset_config.embedding_fields:
                if field in doc.metadata and doc.metadata[field] is not None:
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            # Count provinces
            if 'province' in doc.metadata and doc.metadata['province']:
                province = doc.metadata['province']
                province_counts[province] = province_counts.get(province, 0) + 1
            
            # Count deed types
            if 'deed_type' in doc.metadata and doc.metadata['deed_type']:
                deed_type = doc.metadata['deed_type']
                deed_type_counts[deed_type] = deed_type_counts.get(deed_type, 0) + 1
            
            # Collect area statistics
            if 'area_total_sqm' in doc.metadata and doc.metadata['area_total_sqm']:
                area_values.append(doc.metadata['area_total_sqm'])
        
        # Calculate coverage percentages
        for field in self.dataset_config.embedding_fields:
            count = field_counts.get(field, 0)
            summary['field_coverage'][field] = {
                'count': count,
                'percentage': round(count / len(documents) * 100, 2)
            }
        
        # Top locations
        summary['location_distribution'] = dict(
            sorted(province_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Deed types
        summary['deed_type_distribution'] = deed_type_counts
        
        # Area statistics
        if area_values:
            summary['area_statistics'] = {
                'min_area_sqm': min(area_values),
                'max_area_sqm': max(area_values),
                'avg_area_sqm': sum(area_values) / len(area_values),
                'total_records_with_area': len(area_values)
            }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "reports", "conversion_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion summary saved to: {summary_path}")
    
    def save_documents_as_jsonl(self, documents: List[SimpleDocument], filename: str = None):
        """Save all documents as a single JSONL file"""
        
        if filename is None:
            config_name = self.dataset_config.name if self.dataset_config else 'iland_documents'
            filename = f'{config_name}_documents.jsonl'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                doc_data = {
                    'id': doc.id,
                    'text': doc.text,
                    'metadata': doc.metadata,
                    'embedding_priority_fields': {
                        field: doc.metadata.get(field) 
                        for field in self.dataset_config.embedding_fields 
                        if field in doc.metadata
                    }
                }
                f.write(json.dumps(doc_data, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(documents)} documents to: {filepath}")
        return filepath

    def save_documents_as_markdown_files(self, documents: List[SimpleDocument], prefix: str = "deed", batch_size: int = 1000):
        """Save each document as an individual markdown file with better organization"""
        
        if not documents:
            logger.warning("No documents to save as markdown files")
            return []
        
        # Create a subdirectory for markdown files
        markdown_dir = os.path.join(self.output_dir, "iland_markdown_files")
        os.makedirs(markdown_dir, exist_ok=True)
        
        # Create subdirectories by province for better organization
        province_dirs = {}
        
        saved_files = []
        
        # Determine padding based on total number of documents
        total_docs = len(documents)
        padding = len(str(total_docs))
        
        for idx, doc in enumerate(documents, 1):
            # Get province for organization
            province = doc.metadata.get('province', 'unknown')
            province_safe = re.sub(r'[^\w\s-]', '', province).strip().replace(' ', '_')
            
            if province_safe not in province_dirs:
                province_dir = os.path.join(markdown_dir, province_safe)
                os.makedirs(province_dir, exist_ok=True)
                province_dirs[province_safe] = province_dir
            
            # Create filename with document ID and index
            doc_id = doc.metadata.get('doc_id', f'doc_{idx}')
            filename = f"{prefix}_{str(idx).zfill(padding)}_{doc_id}.md"
            filepath = os.path.join(province_dirs[province_safe], filename)
            
            # Use the document text directly as it's already formatted as markdown
            markdown_content = doc.text
            
            # Add enhanced metadata section at the end
            markdown_content += "\n\n---\n## ข้อมูลเมตา (Document Metadata)\n\n"
            
            # Group metadata by type for better organization
            metadata_groups = {
                'identifiers': [],
                'deed_info': [],
                'location': [],
                'land_details': [],
                'area_measurements': [],
                'dates': [],
                'classification': [],
                'financial': [],
                'computed': [],
                'system': [],
                'other': []
            }
            
            # Categorize metadata based on field mappings
            field_type_map = {}
            if self.dataset_config:
                for mapping in self.dataset_config.field_mappings:
                    field_type_map[mapping.metadata_key] = mapping.field_type
            
            for key, value in doc.metadata.items():
                if value is None:
                    continue
                    
                # Determine category
                if key in ['doc_id', 'row_index', 'search_text']:
                    metadata_groups['computed'].append((key, value))
                elif key in ['source', 'created_at', 'doc_type', 'config_name', 'encoding_used']:
                    metadata_groups['system'].append((key, value))
                elif key in ['location_hierarchy', 'area_formatted', 'area_total_sqm']:
                    metadata_groups['computed'].append((key, value))
                else:
                    field_type = field_type_map.get(key, 'other')
                    if field_type in metadata_groups:
                        metadata_groups[field_type].append((key, value))
                    else:
                        metadata_groups['other'].append((key, value))
            
            # Add metadata sections with Thai/English headers
            section_titles = {
                'identifiers': 'รหัสอ้างอิง (Identifiers)',
                'deed_info': 'ข้อมูลโฉนด (Deed Information)',
                'location': 'ที่ตั้ง (Location)',
                'land_details': 'รายละเอียดที่ดิน (Land Details)',
                'area_measurements': 'ขนาดพื้นที่ (Area Measurements)',
                'dates': 'วันที่ (Dates)',
                'classification': 'การจำแนก (Classification)',
                'financial': 'การเงิน (Financial)',
                'computed': 'ข้อมูลประมวลผล (Computed Fields)',
                'other': 'ข้อมูลอื่นๆ (Additional Information)',
                'system': 'ข้อมูลระบบ (System Information)'
            }
            
            for group_key, items in metadata_groups.items():
                if items:
                    markdown_content += f"\n### {section_titles[group_key]}\n\n"
                    
                    for key, value in items:
                        formatted_key = key.replace('_', ' ').title()
                        # Special formatting for certain fields
                        if key == 'area_total_sqm' and isinstance(value, (int, float)):
                            formatted_value = f"{value:,.2f} ตร.ม."
                        elif key == 'created_at':
                            formatted_value = value
                        else:
                            formatted_value = value
                        
                        markdown_content += f"- **{formatted_key}**: {formatted_value}\n"
                    
                    markdown_content += "\n"
            
            # Add embedding priority fields section
            if self.dataset_config and self.dataset_config.embedding_fields:
                markdown_content += "\n### ฟิลด์สำคัญสำหรับการค้นหา (Key Search Fields)\n\n"
                for field in self.dataset_config.embedding_fields:
                    if field in doc.metadata and doc.metadata[field]:
                        formatted_key = field.replace('_', ' ').title()
                        markdown_content += f"- **{formatted_key}**: {doc.metadata[field]}\n"
                markdown_content += "\n"
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                saved_files.append(filepath)
                
                # Log progress for large datasets
                if idx % batch_size == 0:
                    logger.info(f"Saved {idx}/{total_docs} markdown files...")
                    
            except Exception as e:
                logger.error(f"Failed to save markdown file {filename}: {e}")
        
        # Create index file for easy navigation
        self._create_markdown_index(markdown_dir, saved_files, documents)
        
        logger.info(f"Saved {len(saved_files)} markdown files to: {markdown_dir}")
        logger.info(f"Files organized into {len(province_dirs)} province directories")
        return saved_files
    
    def _create_markdown_index(self, markdown_dir: str, saved_files: List[str], documents: List[SimpleDocument]):
        """Create an index markdown file for easy navigation"""
        index_path = os.path.join(markdown_dir, "README.md")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# iLand Dataset - Land Deed Records Index\n\n")
            f.write(f"Total Documents: {len(documents)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by province
            province_groups = {}
            for doc, filepath in zip(documents, saved_files):
                province = doc.metadata.get('province', 'Unknown')
                if province not in province_groups:
                    province_groups[province] = []
                province_groups[province].append({
                    'doc': doc,
                    'filepath': os.path.relpath(filepath, markdown_dir)
                })
            
            f.write("## Documents by Province\n\n")
            for province in sorted(province_groups.keys()):
                f.write(f"### {province} ({len(province_groups[province])} documents)\n\n")
                
                # Show first 10 documents as examples
                for item in province_groups[province][:10]:
                    doc = item['doc']
                    filepath = item['filepath']
                    
                    # Create summary
                    summary_parts = []
                    if 'deed_type' in doc.metadata:
                        summary_parts.append(f"Type: {doc.metadata['deed_type']}")
                    if 'district' in doc.metadata:
                        summary_parts.append(f"District: {doc.metadata['district']}")
                    if 'area_formatted' in doc.metadata:
                        summary_parts.append(f"Area: {doc.metadata['area_formatted']}")
                    
                    summary = " | ".join(summary_parts) if summary_parts else "No summary available"
                    
                    f.write(f"- [{os.path.basename(filepath)}]({filepath}) - {summary}\n")
                
                if len(province_groups[province]) > 10:
                    f.write(f"- ... and {len(province_groups[province]) - 10} more documents\n")
                f.write("\n")
        
        logger.info(f"Created index file: {index_path}")

def main():
    """Main execution function for iLand dataset processing"""
    
    # Auto-detect correct paths based on script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up two levels from src-iLand/data_processing/
    input_dir = project_root / "data" / "input_docs"
    
    # Look for the iLand CSV file
    input_csv = input_dir / "input_dataset_iLand.csv"
    
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find iLand CSV file: {input_csv}\n"
            "Please ensure input_dataset_iLand.csv exists in data/input_docs/"
        )
    
    # Set output directory
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using iLand CSV file: {input_csv}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create iLand converter
    converter = iLandCSVConverter(str(input_csv), output_dir)
    
    # Setup configuration (auto-generate from CSV analysis)
    config = converter.setup_configuration(config_name="iland_deed_records", auto_generate=True)
    
    # Process documents in smaller batches due to large dataset
    logger.info("Processing large dataset in batches...")
    documents = converter.process_csv_to_documents(batch_size=500)
    
    # Save documents as JSONL
    jsonl_path = converter.save_documents_as_jsonl(documents)
    
    # Save documents as individual markdown files
    markdown_files = converter.save_documents_as_markdown_files(documents, batch_size=1000)
    
    logger.info("iLand dataset conversion completed successfully!")
    logger.info(f"Total documents created: {len(documents)}")
    logger.info(f"Configuration: {config.name}")
    logger.info(f"JSONL output: {jsonl_path}")
    logger.info(f"Markdown files: {len(markdown_files)} files in {output_dir}/iland_markdown_files/")
    
    # Print summary statistics
    logger.info("\n=== Conversion Summary ===")
    logger.info(f"Documents with complete location data: {sum(1 for d in documents if all(k in d.metadata for k in ['province', 'district', 'subdistrict']))}")
    logger.info(f"Documents with area measurements: {sum(1 for d in documents if 'area_formatted' in d.metadata)}")
    logger.info(f"Documents with deed type: {sum(1 for d in documents if 'deed_type' in d.metadata and d.metadata['deed_type'])}")

if __name__ == "__main__":
    main()