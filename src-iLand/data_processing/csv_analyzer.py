import pandas as pd
import re
import logging
from typing import List, Dict, Any

# Handle both relative and absolute imports
try:
    from ..common.thai_provinces import THAI_PROVINCES
except ImportError:  # When package context is missing
    from common.thai_provinces import THAI_PROVINCES

try:
    from .models import FieldMapping, DatasetConfig
except ImportError:
    from models import FieldMapping, DatasetConfig

logger = logging.getLogger(__name__)


class CSVAnalyzer:
    """Handles CSV structure analysis and field mapping suggestions"""
    
    def __init__(self):
        self.thai_provinces = self._load_thai_provinces()
    
    def _load_thai_provinces(self) -> List[str]:
        """Load list of Thai provinces for validation"""
        return THAI_PROVINCES
    
    def analyze_csv_structure(self, csv_path: str) -> Dict[str, Any]:
        """Analyze CSV structure and suggest field mappings for iLand dataset"""
        logger.info(f"Analyzing iLand CSV structure: {csv_path}")
        
        # Smart encoding detection for Thai CSV files
        logger.info("Detecting optimal encoding for CSV file...")
        # Try Thai encoding first since this is for iLand (Thai) dataset  
        encodings_to_try = ['cp874', 'utf-8', 'utf-8-sig', 'latin-1']
        sample_df = None
        used_encoding = None
        
        for i, encoding in enumerate(encodings_to_try):
            try:
                sample_df = pd.read_csv(csv_path, nrows=200, encoding=encoding)
                used_encoding = encoding
                logger.info(f"✓ Successfully read CSV with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                # Only show debug info, not warnings, since this is expected behavior
                logger.debug(f"Encoding {encoding} not compatible, trying next...")
                continue
            except Exception as e:
                logger.debug(f"Error with encoding {encoding}: {str(e)[:50]}")
                continue
        
        if sample_df is None:
            raise ValueError(f"Could not read CSV file with any of the tried encodings: {encodings_to_try}")
        
        csv_columns = list(sample_df.columns)
        
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
            'total_columns': len(csv_columns),
            'columns': csv_columns,
            'suggested_mappings': self._suggest_field_mappings(sample_df),
            'data_types': dict(sample_df.dtypes.astype(str)),
            'sample_data': sample_df.head(3).to_dict('records'),
            'encoding_used': used_encoding,
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
            'geolocation': {
                'patterns': [
                    (r'geom.*point', 'Geographic coordinates point'),
                    (r'land.*geom', 'Land geometry data'),
                    (r'coordinates', 'Coordinate information'),
                    (r'lat.*lon', 'Latitude longitude'),
                    (r'point.*geo', 'Geographic point data')
                ],
                'metadata_keys': ['land_geom_point', 'coordinates', 'latitude', 'longitude']
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
            
            for field_type, config in field_patterns.items():
                for pattern_tuple in config['patterns']:
                    pattern, description = pattern_tuple if isinstance(pattern_tuple, tuple) else (pattern_tuple, "")
                    if re.search(pattern, column_lower):
                        # Calculate confidence based on pattern match quality
                        match = re.search(pattern, column_lower)
                        confidence = len(match.group()) / len(column_lower) if match else 0
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
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
            # Try to parse as date with common Thai/international formats
            sample_values = series.dropna().astype(str).head(10)
            if len(sample_values) > 0:
                try:
                    # Additional handling for Thai Buddhist Era dates
                    be_match = 0
                    be_patterns = [
                        re.compile(r'^(\d{1,2})/(\d{1,2})/(\d{4})$'),  # 25/09/2567
                        re.compile(r'^(\d{4})-(\d{1,2})-(\d{1,2})$'),   # 2567-09-25
                    ]

                    for val in sample_values:
                        for pat in be_patterns:
                            m = pat.match(val.strip())
                            if m:
                                if pat.pattern.startswith('^(\\d{1,2})/'):
                                    d, mth, y = map(int, m.groups())
                                else:
                                    y, mth, d = map(int, m.groups())
                                if y > 2500:
                                    y -= 543  # convert BE to CE
                                try:
                                    pd.Timestamp(year=y, month=mth, day=d)
                                    be_match += 1
                                    break
                                except Exception:
                                    pass

                    if be_match >= len(sample_values) * 0.7:
                        return 'date'

                    # Try common date formats first to avoid warnings
                    common_formats = [
                        '%Y-%m-%d',      # 2023-12-31
                        '%d/%m/%Y',      # 31/12/2023
                        '%d-%m-%Y',      # 31-12-2023
                        '%Y/%m/%d',      # 2023/12/31
                        '%d/%m/%y',      # 31/12/23
                        '%Y%m%d',        # 20231231
                    ]
                    
                    # Test if any common format works for most values
                    for fmt in common_formats:
                        try:
                            parsed_count = 0
                            for val in sample_values:
                                try:
                                    pd.to_datetime(str(val), format=fmt)
                                    parsed_count += 1
                                except:
                                    continue
                            
                            # If most values parse successfully with this format
                            if parsed_count >= len(sample_values) * 0.7:
                                return 'date'
                        except:
                            continue
                    
                    # Fallback: try pandas auto-detection but suppress warnings
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(sample_values, errors='coerce')
                        # If we get here without exception, it might be dates
                        non_null_parsed = pd.to_datetime(sample_values, errors='coerce').notna().sum()
                        if non_null_parsed >= len(sample_values) * 0.5:
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

## พิกัดภูมิศาสตร์ (Geolocation)
{geolocation_section}

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
            'deed_group_type', 'coordinates_formatted', 'longitude', 'latitude'
        ]
        
        config = DatasetConfig(
            name=config_name,
            description=f"Configuration for iLand dataset - Thai land deed records with {len(field_mappings)} fields",
            field_mappings=field_mappings,
            text_template=text_template,
            embedding_fields=embedding_fields
        )
        
        return config 