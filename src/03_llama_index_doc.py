import pandas as pd
import json
import os
import yaml
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

# LlamaIndex imports
try:
    from llama_index.core import Document
except ImportError:
    try:
        from llama_index import Document
    except ImportError:
        raise ImportError("LlamaIndex not found. Please install with: pip install llama-index")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FieldMapping:
    """Configuration for mapping CSV columns to metadata fields"""
    csv_column: str
    metadata_key: str
    field_type: str  # 'identifier', 'demographic', 'education', 'career', 'compensation', 'assessment'
    data_type: str = 'string'  # 'string', 'numeric', 'date', 'boolean'
    required: bool = False
    aliases: List[str] = None  # Alternative column names
    
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

class LlamaIndexCSVConverter:
    """CSV to LlamaIndex Document converter with markdown output"""
    
    def __init__(self, input_csv_path: str, output_dir: str, config_path: str = None):
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.config_path = config_path
        self.dataset_config = None
        self.csv_columns = None
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_csv_structure(self) -> Dict[str, Any]:
        """Analyze CSV structure and suggest field mappings"""
        logger.info(f"Analyzing CSV structure: {self.input_csv_path}")
        
        # Read a sample of the CSV
        sample_df = pd.read_csv(self.input_csv_path, nrows=100)
        self.csv_columns = list(sample_df.columns)
        
        analysis = {
            'total_columns': len(self.csv_columns),
            'columns': self.csv_columns,
            'suggested_mappings': self._suggest_field_mappings(sample_df),
            'data_types': dict(sample_df.dtypes.astype(str)),
            'sample_data': sample_df.head(3).to_dict('records')
        }
        
        return analysis
    
    def _suggest_field_mappings(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Automatically suggest field mappings based on column names and content"""
        
        # Define pattern mappings for common field types
        field_patterns = {
            'identifier': {
                'patterns': [r'id', r'no', r'number', r'code'],
                'metadata_keys': ['id', 'number', 'code']
            },
            'demographic': {
                'patterns': [r'age', r'province', r'state', r'region', r'location', r'country'],
                'metadata_keys': ['age', 'province', 'state', 'region', 'location']
            },
            'education': {
                'patterns': [r'education', r'degree', r'major', r'institute', r'university', r'school'],
                'metadata_keys': ['education_level', 'degree', 'major', 'institute']
            },
            'career': {
                'patterns': [r'job', r'position', r'title', r'company', r'industry', r'experience', r'work'],
                'metadata_keys': ['position', 'company', 'industry', 'experience_level', 'job_family']
            },
            'compensation': {
                'patterns': [r'salary', r'compensation', r'pay', r'wage', r'bonus', r'currency'],
                'metadata_keys': ['salary', 'bonus', 'currency']
            },
            'assessment': {
                'patterns': [r'test', r'score', r'assessment', r'skill', r'evaluation'],
                'metadata_keys': ['test_type', 'test_score', 'skill_level']
            }
        }
        
        suggestions = []
        
        for column in df.columns:
            column_lower = column.lower()
            best_match = None
            best_confidence = 0
            
            for field_type, config in field_patterns.items():
                for pattern in config['patterns']:
                    if re.search(pattern, column_lower):
                        confidence = len(pattern) / len(column_lower)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_match = {
                                'csv_column': column,
                                'field_type': field_type,
                                'suggested_metadata_key': self._suggest_metadata_key(column, config['metadata_keys']),
                                'confidence': confidence,
                                'data_type': self._infer_data_type(df[column])
                            }
            
            if best_match:
                suggestions.append(best_match)
            else:
                # For unmatched columns, suggest as generic metadata
                suggestions.append({
                    'csv_column': column,
                    'field_type': 'generic',
                    'suggested_metadata_key': self._normalize_key(column),
                    'confidence': 0.1,
                    'data_type': self._infer_data_type(df[column])
                })
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def _suggest_metadata_key(self, column_name: str, possible_keys: List[str]) -> str:
        """Suggest the best metadata key based on column name"""
        column_lower = column_name.lower()
        
        # Try exact matches first
        for key in possible_keys:
            if key in column_lower:
                return key
        
        # Return normalized column name as fallback
        return self._normalize_key(column_name)
    
    def _normalize_key(self, key: str) -> str:
        """Normalize a column name to a valid metadata key"""
        # Convert to lowercase, replace spaces and special chars with underscores
        normalized = re.sub(r'[^a-zA-Z0-9]+', '_', key.lower())
        normalized = re.sub(r'_+', '_', normalized)  # Remove multiple underscores
        return normalized.strip('_')
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer the data type of a pandas Series"""
        if pd.api.types.is_numeric_dtype(series):
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
                required=suggestion['confidence'] > 0.7  # High confidence fields are required
            )
            field_mappings.append(mapping)
        
        config = DatasetConfig(
            name=config_name,
            description=f"Auto-generated configuration for {config_name}",
            field_mappings=field_mappings
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
            'text_template': config.text_template
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
            text_template=config_dict.get('text_template')
        )
        
        return config
    
    def setup_configuration(self, config_name: str = None, auto_generate: bool = True) -> DatasetConfig:
        """Setup configuration for the CSV conversion"""
        
        if self.config_path and os.path.exists(self.config_path):
            logger.info(f"Loading existing configuration: {self.config_path}")
            self.dataset_config = self.load_config(self.config_path)
        elif auto_generate:
            logger.info("Auto-generating configuration from CSV analysis")
            analysis = self.analyze_csv_structure()
            
            if config_name is None:
                config_name = Path(self.input_csv_path).stem
            
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
        report_path = os.path.join(self.output_dir, "csv_analysis_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis report saved to: {report_path}")
    
    def clean_value(self, value: Any) -> Any:
        """Clean and normalize values for metadata"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip()
        return value
    
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
                    # Convert numeric values to appropriate types for LlamaIndex
                    if mapping.data_type == 'numeric' and isinstance(cleaned_value, (int, float)):
                        metadata[mapping.metadata_key] = float(cleaned_value)
                    elif mapping.data_type == 'boolean':
                        metadata[mapping.metadata_key] = bool(cleaned_value)
                    else:
                        metadata[mapping.metadata_key] = str(cleaned_value)
        
        return metadata
    
    def generate_document_text(self, row: pd.Series, metadata: Dict[str, Any]) -> str:
        """Generate document text content"""
        
        if self.dataset_config.text_template:
            # Use custom template if provided
            return self.dataset_config.text_template.format(**metadata)
        
        # Use default template based on field types
        return self._generate_default_text(metadata)
    
    def _generate_default_text(self, metadata: Dict[str, Any]) -> str:
        """Generate default text content from metadata"""
        
        # Group metadata by field types
        field_groups = {
            'identifier': [],
            'demographic': [],
            'education': [],
            'career': [],
            'compensation': [],
            'assessment': [],
            'generic': []
        }
        
        # Map metadata back to field types
        for mapping in self.dataset_config.field_mappings:
            if mapping.metadata_key in metadata:
                field_groups[mapping.field_type].append(
                    f"{mapping.metadata_key.replace('_', ' ').title()}: {metadata[mapping.metadata_key]}"
                )
        
        # Build text sections
        text_parts = []
        
        if field_groups['identifier']:
            text_parts.append(f"Profile: {', '.join(field_groups['identifier'])}")
        
        if field_groups['demographic']:
            text_parts.append(f"Demographics: {', '.join(field_groups['demographic'])}")
        
        if field_groups['education']:
            text_parts.append(f"Education: {', '.join(field_groups['education'])}")
        
        if field_groups['career']:
            text_parts.append(f"Career: {', '.join(field_groups['career'])}")
        
        if field_groups['compensation']:
            text_parts.append(f"Compensation: {', '.join(field_groups['compensation'])}")
        
        if field_groups['assessment']:
            text_parts.append(f"Assessments: {', '.join(field_groups['assessment'])}")
        
        if field_groups['generic']:
            text_parts.append(f"Additional Information: {', '.join(field_groups['generic'])}")
        
        return "\n\n".join(text_parts)
    
    def convert_row_to_document(self, row: pd.Series, row_index: int) -> Document:
        """Convert a single CSV row to a LlamaIndex Document"""
        
        # Extract metadata using configuration
        metadata = self.extract_metadata_from_row(row)
        
        # Generate document text
        text_content = self.generate_document_text(row, metadata)
        
        # Get document ID from metadata or use row index
        doc_id = None
        if 'id' in metadata:
            doc_id = str(metadata['id'])
        elif 'number' in metadata:
            doc_id = str(metadata['number'])
        else:
            doc_id = f"row_{row_index}"
        
        # Create LlamaIndex Document
        doc = Document(text=text_content, doc_id=doc_id)
        
        # Add processing metadata
        enhanced_metadata = metadata.copy()
        enhanced_metadata.update({
            'source': 'csv_import',
            'created_at': datetime.now().isoformat(),
            'doc_type': 'csv_record',
            'config_name': self.dataset_config.name,
            'row_index': row_index
        })
        
        # Set structured metadata for filtering (following LlamaIndex pattern)
        doc.metadata = enhanced_metadata
        
        return doc
    
    def process_csv_to_documents(self, batch_size: int = 1000) -> List[Document]:
        """Process entire CSV file and convert to LlamaIndex Documents"""
        
        if self.dataset_config is None:
            raise ValueError("Configuration not set up. Call setup_configuration() first.")
        
        logger.info(f"Starting conversion using config: {self.dataset_config.name}")
        
        documents = []
        chunk_num = 0
        global_row_index = 0
        
        for chunk in pd.read_csv(self.input_csv_path, chunksize=batch_size):
            chunk_num += 1
            logger.info(f"Processing chunk {chunk_num} ({len(chunk)} rows)")
            
            for idx, row in chunk.iterrows():
                try:
                    global_row_index += 1
                    document = self.convert_row_to_document(row, global_row_index)
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {e}")
        
        logger.info(f"Converted {len(documents)} rows to LlamaIndex Documents")
        return documents
    
    def save_documents_as_jsonl(self, documents: List[Document], filename: str = None):
        """Save all LlamaIndex Documents as a JSONL file"""
        
        if filename is None:
            config_name = self.dataset_config.name if self.dataset_config else 'documents'
            filename = f'{config_name}_llama_documents.jsonl'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                doc_data = {
                    'doc_id': doc.doc_id,
                    'text': doc.text,
                    'metadata': doc.metadata
                }
                f.write(json.dumps(doc_data, ensure_ascii=False, default=str) + '\n')
        
        logger.info(f"Saved {len(documents)} LlamaIndex Documents to: {filepath}")
        return filepath
    
    def save_documents_as_markdown_files(self, documents: List[Document], prefix: str = "row"):
        """Save each LlamaIndex Document as an individual markdown file"""
        
        if not documents:
            logger.warning("No documents to save as markdown files")
            return []
        
        # Create a subdirectory for markdown files
        markdown_dir = os.path.join(self.output_dir, "llama_markdown_files")
        os.makedirs(markdown_dir, exist_ok=True)
        
        saved_files = []
        
        # Determine padding based on total number of documents
        total_docs = len(documents)
        padding = len(str(total_docs))
        
        for idx, doc in enumerate(documents, 1):
            # Create enumerated filename with zero-padding
            filename = f"{prefix}_{str(idx).zfill(padding)}.md"
            filepath = os.path.join(markdown_dir, filename)
            
            # Format document content as markdown
            markdown_content = self._format_document_as_markdown(doc)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                saved_files.append(filepath)
            except Exception as e:
                logger.error(f"Failed to save markdown file {filename}: {e}")
        
        logger.info(f"Saved {len(saved_files)} markdown files to: {markdown_dir}")
        return saved_files
    
    def _format_document_as_markdown(self, doc: Document) -> str:
        """Format a LlamaIndex Document as markdown content"""
        
        # Start with a title using doc_id
        title = f"# Document ID: {doc.doc_id}"
        
        markdown_lines = [title, ""]
        
        # Add the main content with proper markdown formatting
        if doc.text:
            # Split the text into sections and format as markdown
            text_sections = doc.text.split('\n\n')
            
            for section in text_sections:
                if ':' in section:
                    # Convert "Section: content" to "## Section\ncontent"
                    parts = section.split(':', 1)
                    section_title = parts[0].strip()
                    section_content = parts[1].strip()
                    
                    markdown_lines.append(f"## {section_title}")
                    markdown_lines.append("")
                    markdown_lines.append(section_content)
                    markdown_lines.append("")
                else:
                    # Regular content
                    markdown_lines.append(section)
                    markdown_lines.append("")
        
        # Add metadata section
        if doc.metadata:
            markdown_lines.append("## Metadata")
            markdown_lines.append("")
            
            # Group metadata by type for better organization
            metadata_groups = {
                'identifiers': [],
                'demographics': [],
                'education': [],
                'career': [],
                'compensation': [],
                'assessments': [],
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
                    
                field_type = field_type_map.get(key, 'other')
                
                if field_type == 'identifier':
                    metadata_groups['identifiers'].append((key, value))
                elif field_type == 'demographic':
                    metadata_groups['demographics'].append((key, value))
                elif field_type == 'education':
                    metadata_groups['education'].append((key, value))
                elif field_type == 'career':
                    metadata_groups['career'].append((key, value))
                elif field_type == 'compensation':
                    metadata_groups['compensation'].append((key, value))
                elif field_type == 'assessment':
                    metadata_groups['assessments'].append((key, value))
                elif key in ['source', 'created_at', 'doc_type', 'config_name', 'row_index']:
                    metadata_groups['system'].append((key, value))
                else:
                    metadata_groups['other'].append((key, value))
            
            # Add metadata sections
            section_titles = {
                'identifiers': 'Identifiers',
                'demographics': 'Demographics',
                'education': 'Education',
                'career': 'Career Information',
                'compensation': 'Compensation',
                'assessments': 'Assessments',
                'other': 'Additional Information',
                'system': 'System Information'
            }
            
            for group_key, items in metadata_groups.items():
                if items:
                    markdown_lines.append(f"### {section_titles[group_key]}")
                    markdown_lines.append("")
                    
                    for key, value in items:
                        formatted_key = key.replace('_', ' ').title()
                        markdown_lines.append(f"- **{formatted_key}**: {value}")
                    
                    markdown_lines.append("")
        
        return '\n'.join(markdown_lines)

def main():
    """Main execution function with LlamaIndex Document integration"""
    
    # Auto-detect correct paths based on script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_dir = project_root / "data" / "input_docs"
    
    input_csv = None
    # Look for CSV files in the input directory
    if input_dir.exists():
        csv_files = list(input_dir.glob("*.csv"))
        if csv_files:
            # Use the first CSV file found (or specify input_dataset.csv if it exists)
            target_file = input_dir / "input_dataset.csv"
            if target_file.exists():
                input_csv = str(target_file)
            else:
                input_csv = str(csv_files[0])  # Use first CSV file found
    
    if input_csv is None:
        raise FileNotFoundError(
            "Could not find any CSV files in data/input_docs/ directory.\n"
            f"Searched directory: {input_dir}\n"
            "Please ensure your CSV file exists in data/input_docs/"
        )
    
    # Set output directory relative to project root
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using CSV file: {input_csv}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create LlamaIndex converter
    converter = LlamaIndexCSVConverter(input_csv, output_dir)
    
    # Setup configuration (auto-generate from CSV analysis)
    config = converter.setup_configuration(config_name="candidate_profiles_llama", auto_generate=True)
    
    # Process documents
    documents = converter.process_csv_to_documents(batch_size=500)
    
    # Save documents as JSONL with LlamaIndex format
    converter.save_documents_as_jsonl(documents)
    
    # Save documents as individual markdown files
    markdown_files = converter.save_documents_as_markdown_files(documents)
    
    logger.info("LlamaIndex Document conversion completed successfully!")
    logger.info(f"Total LlamaIndex Documents created: {len(documents)}")
    logger.info(f"Configuration: {config.name}")
    logger.info(f"JSONL output saved to: {output_dir}")
    logger.info(f"Markdown files saved: {len(markdown_files)} files in {output_dir}/llama_markdown_files/")

if __name__ == "__main__":
    main() 