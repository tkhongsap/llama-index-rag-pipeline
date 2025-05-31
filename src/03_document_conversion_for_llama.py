import pandas as pd
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import logging
from pathlib import Path
from llama_index.core import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaIndexDocumentConverter:
    """Optimized CSV to LlamaIndex Document converter"""
    
    def __init__(self, input_csv_path: str, output_dir: str):
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _clean_value(self, value: Any) -> Any:
        """Clean and normalize values for metadata"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip()
        return value
    
    def _normalize_metadata_key(self, key: str) -> str:
        """Normalize column name to valid metadata key"""
        return key.lower().replace(' ', '_').replace('-', '_')
    
    def _extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata from CSV row"""
        metadata = {}
        
        for column, value in row.items():
            cleaned_value = self._clean_value(value)
            if cleaned_value is not None:
                key = self._normalize_metadata_key(column)
                
                # Type conversion for LlamaIndex compatibility
                if isinstance(cleaned_value, (int, float)):
                    metadata[key] = float(cleaned_value)
                elif str(cleaned_value).lower() in ['true', 'false']:
                    metadata[key] = str(cleaned_value).lower() == 'true'
                else:
                    metadata[key] = str(cleaned_value)
        
        return metadata
    
    def _generate_document_text(self, row: pd.Series) -> str:
        """Generate document text from CSV row"""
        text_parts = []
        
        for column, value in row.items():
            cleaned_value = self._clean_value(value)
            if cleaned_value is not None:
                text_parts.append(f"{column}: {cleaned_value}")
        
        return "\n".join(text_parts)
    
    def _get_document_id(self, row: pd.Series, row_index: int) -> str:
        """Extract document ID from row or generate one"""
        for id_column in ['Id', 'ID', 'id', 'No', 'no']:
            if id_column in row and not pd.isna(row[id_column]):
                return str(row[id_column])
        return f"row_{row_index}"
    
    def convert_row_to_document(self, row: pd.Series, row_index: int) -> Document:
        """Convert single CSV row to LlamaIndex Document"""
        try:
            # Extract metadata and text
            metadata = self._extract_metadata(row)
            text_content = self._generate_document_text(row)
            doc_id = self._get_document_id(row, row_index)
            
            # Add processing metadata
            metadata.update({
                'source': 'csv_import',
                'created_at': datetime.now().isoformat(),
                'doc_type': 'csv_record',
                'row_index': row_index
            })
            
            # Create LlamaIndex Document
            doc = Document(text=text_content, doc_id=doc_id)
            doc.metadata = metadata
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to convert row {row_index}: {e}")
            raise
    
    def process_csv_to_documents(self, batch_size: int = 1000) -> List[Document]:
        """Process CSV file and convert to LlamaIndex Documents"""
        logger.info(f"Starting conversion: {self.input_csv_path}")
        
        documents = []
        global_row_index = 0
        
        try:
            for chunk in pd.read_csv(self.input_csv_path, chunksize=batch_size):
                logger.info(f"Processing {len(chunk)} rows")
                
                for _, row in chunk.iterrows():
                    try:
                        global_row_index += 1
                        document = self.convert_row_to_document(row, global_row_index)
                        documents.append(document)
                    except Exception as e:
                        logger.warning(f"Skipping row {global_row_index}: {e}")
                        continue
            
            logger.info(f"Successfully converted {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process CSV: {e}")
            raise
    
    def save_documents_as_jsonl(self, documents: List[Document], filename: str = None) -> str:
        """Save documents as JSONL file"""
        if not filename:
            filename = 'llama_documents.jsonl'
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for doc in documents:
                    doc_data = {
                        'doc_id': doc.doc_id,
                        'text': doc.text,
                        'metadata': doc.metadata
                    }
                    f.write(json.dumps(doc_data, ensure_ascii=False, default=str) + '\n')
            
            logger.info(f"Saved {len(documents)} documents to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save JSONL: {e}")
            raise
    
    def save_documents_as_markdown(self, documents: List[Document], prefix: str = "doc") -> List[str]:
        """Save documents as individual markdown files"""
        if not documents:
            logger.warning("No documents to save")
            return []
        
        markdown_dir = os.path.join(self.output_dir, "llama_markdown_files")
        os.makedirs(markdown_dir, exist_ok=True)
        
        saved_files = []
        padding = len(str(len(documents)))
        
        try:
            for idx, doc in enumerate(documents, 1):
                filename = f"{prefix}_{str(idx).zfill(padding)}.md"
                filepath = os.path.join(markdown_dir, filename)
                
                markdown_content = self._format_as_markdown(doc)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                saved_files.append(filepath)
            
            logger.info(f"Saved {len(saved_files)} markdown files to: {markdown_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save markdown files: {e}")
            raise
    
    def _format_as_markdown(self, doc: Document) -> str:
        """Format document as markdown optimized for LlamaIndex pipeline"""
        lines = []
        
        # Document header with clear identification
        doc_title = f"Candidate Profile {doc.doc_id}"
        lines.extend([f"# {doc_title}", ""])
        
        # Extract key information for better chunking
        metadata = doc.metadata
        
        # Profile Summary Section (for high-level retrieval)
        lines.extend(["## Profile Summary", ""])
        summary_parts = []
        
        # Add key identifiers
        if metadata.get('id'):
            summary_parts.append(f"Profile ID: {metadata['id']}")
        if metadata.get('position'):
            summary_parts.append(f"Current Position: {metadata['position']}")
        if metadata.get('experience'):
            summary_parts.append(f"Experience Level: {metadata['experience']}")
        if metadata.get('education_level'):
            summary_parts.append(f"Education: {metadata['education_level']}")
        
        if summary_parts:
            lines.append(" | ".join(summary_parts))
            lines.append("")
        
        # Demographics Section (semantic chunking)
        demo_fields = ['age', 'age2', 'province', 'region', 'state___province']
        demo_content = []
        for field in demo_fields:
            if metadata.get(field):
                formatted_field = field.replace('_', ' ').title()
                demo_content.append(f"**{formatted_field}**: {metadata[field]}")
        
        if demo_content:
            lines.extend(["## Demographics", ""])
            lines.extend(demo_content)
            lines.append("")
        
        # Education Section (semantic chunking)
        edu_fields = ['education_level', 'major', 'degree', 'institute', 'education_from__month_year_', 'education_to__month_year_']
        edu_content = []
        for field in edu_fields:
            if metadata.get(field):
                formatted_field = field.replace('_', ' ').title()
                edu_content.append(f"**{formatted_field}**: {metadata[field]}")
        
        if edu_content:
            lines.extend(["## Education Background", ""])
            lines.extend(edu_content)
            lines.append("")
        
        # Career Section (semantic chunking)
        career_fields = ['position', 'companyname', 'industry', 'job_family', 'job_sub_family', 
                        'work_from__month_year_', 'work_to__month_year____present', 'yos_y', 'freshgraduate']
        career_content = []
        for field in career_fields:
            if metadata.get(field):
                formatted_field = field.replace('_', ' ').title()
                career_content.append(f"**{formatted_field}**: {metadata[field]}")
        
        if career_content:
            lines.extend(["## Career Information", ""])
            lines.extend(career_content)
            lines.append("")
        
        # Compensation Section (semantic chunking)
        comp_fields = ['salaryexpectation', 'bonus', 'currencytype', 'currencytype2']
        comp_content = []
        for field in comp_fields:
            if metadata.get(field):
                formatted_field = field.replace('_', ' ').title()
                comp_content.append(f"**{formatted_field}**: {metadata[field]}")
        
        if comp_content:
            lines.extend(["## Compensation Details", ""])
            lines.extend(comp_content)
            lines.append("")
        
        # Skills and Assessment Section (semantic chunking)
        skill_fields = ['testtype', 'score', '30focus']
        skill_content = []
        for field in skill_fields:
            if metadata.get(field):
                formatted_field = field.replace('_', ' ').title()
                skill_content.append(f"**{formatted_field}**: {metadata[field]}")
        
        if skill_content:
            lines.extend(["## Skills and Assessments", ""])
            lines.extend(skill_content)
            lines.append("")
        
        # Additional Information Section (for remaining fields)
        processed_fields = set(['id', 'position', 'experience', 'education_level', 'age', 'age2', 
                               'province', 'region', 'state___province', 'major', 'degree', 'institute',
                               'education_from__month_year_', 'education_to__month_year_', 'companyname',
                               'industry', 'job_family', 'job_sub_family', 'work_from__month_year_',
                               'work_to__month_year____present', 'yos_y', 'freshgraduate', 'salaryexpectation',
                               'bonus', 'currencytype', 'currencytype2', 'testtype', 'score', '30focus',
                               'source', 'created_at', 'doc_type', 'row_index'])
        
        additional_content = []
        for key, value in metadata.items():
            if key not in processed_fields and value is not None:
                formatted_key = key.replace('_', ' ').title()
                additional_content.append(f"**{formatted_key}**: {value}")
        
        if additional_content:
            lines.extend(["## Additional Information", ""])
            lines.extend(additional_content)
            lines.append("")
        
        # Document Metadata Section (for system tracking)
        lines.extend(["## Document Metadata", ""])
        system_fields = ['source', 'created_at', 'doc_type', 'row_index']
        for field in system_fields:
            if metadata.get(field):
                formatted_field = field.replace('_', ' ').title()
                lines.append(f"**{formatted_field}**: {metadata[field]}")
        
        return '\n'.join(lines)


def main():
    """Main execution function"""
    try:
        # Auto-detect paths
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        input_dir = project_root / "data" / "input_docs"
        
        # Find CSV file
        input_csv = None
        if input_dir.exists():
            csv_files = list(input_dir.glob("*.csv"))
            if csv_files:
                target_file = input_dir / "input_dataset.csv"
                input_csv = str(target_file if target_file.exists() else csv_files[0])
        
        if not input_csv:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")
        
        output_dir = str(project_root / "data" / "output_docs")
        
        logger.info(f"Input CSV: {input_csv}")
        logger.info(f"Output directory: {output_dir}")
        
        # Process documents
        converter = LlamaIndexDocumentConverter(input_csv, output_dir)
        documents = converter.process_csv_to_documents(batch_size=500)
        
        # Save outputs
        converter.save_documents_as_jsonl(documents)
        converter.save_documents_as_markdown(documents)
        
        logger.info("Conversion completed successfully!")
        logger.info(f"Total documents: {len(documents)}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main() 