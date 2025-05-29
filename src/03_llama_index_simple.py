import pandas as pd
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Document class that mimics LlamaIndex Document if import fails
class SimpleDocument:
    """Simple document class that mimics LlamaIndex Document structure"""
    
    def __init__(self, text: str, doc_id: str = None, metadata: Dict[str, Any] = None):
        self.text = text
        self.doc_id = doc_id or f"doc_{id(self)}"
        self.metadata = metadata or {}

# Try to import LlamaIndex Document, fall back to SimpleDocument
try:
    from llama_index.core import Document
    logger.info("Using LlamaIndex Document class")
    DocumentClass = Document
except ImportError:
    try:
        from llama_index import Document
        logger.info("Using LlamaIndex Document class (legacy import)")
        DocumentClass = Document
    except ImportError:
        logger.warning("LlamaIndex not available, using SimpleDocument fallback")
        DocumentClass = SimpleDocument

class LlamaIndexCSVConverter:
    """CSV to LlamaIndex Document converter with markdown output"""
    
    def __init__(self, input_csv_path: str, output_dir: str):
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def clean_value(self, value: Any) -> Any:
        """Clean and normalize values for metadata"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip()
        return value
    
    def convert_row_to_document(self, row: pd.Series, row_index: int) -> DocumentClass:
        """Convert a single CSV row to a LlamaIndex Document following the user's pattern"""
        
        # Extract key fields for text content (following user's example format)
        text_parts = []
        metadata = {}
        
        # Process each column in the row
        for column, value in row.items():
            cleaned_value = self.clean_value(value)
            if cleaned_value is not None:
                # Add to text content
                text_parts.append(f"{column}: {cleaned_value}")
                
                # Add to metadata with proper typing
                if isinstance(cleaned_value, (int, float)):
                    metadata[column.lower().replace(' ', '_')] = float(cleaned_value)
                elif str(cleaned_value).lower() in ['true', 'false']:
                    metadata[column.lower().replace(' ', '_')] = str(cleaned_value).lower() == 'true'
                else:
                    metadata[column.lower().replace(' ', '_')] = str(cleaned_value)
        
        # Create text content
        text_content = "\n".join(text_parts)
        
        # Get document ID from row data or use row index
        doc_id = None
        if 'Id' in row and not pd.isna(row['Id']):
            doc_id = str(row['Id'])
        elif 'ID' in row and not pd.isna(row['ID']):
            doc_id = str(row['ID'])
        elif 'No' in row and not pd.isna(row['No']):
            doc_id = str(row['No'])
        else:
            doc_id = f"row_{row_index}"
        
        # Add processing metadata
        metadata.update({
            'source': 'csv_import',
            'created_at': datetime.now().isoformat(),
            'doc_type': 'csv_record',
            'row_index': row_index
        })
        
        # Create Document following the user's pattern
        if DocumentClass == SimpleDocument:
            doc = DocumentClass(text=text_content, doc_id=doc_id, metadata=metadata)
        else:
            # LlamaIndex Document
            doc = DocumentClass(text=text_content, doc_id=doc_id)
            doc.metadata = metadata
        
        return doc
    
    def process_csv_to_documents(self, batch_size: int = 1000) -> List[DocumentClass]:
        """Process entire CSV file and convert to Documents"""
        
        logger.info(f"Starting conversion of CSV: {self.input_csv_path}")
        
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
        
        logger.info(f"Converted {len(documents)} rows to Documents")
        return documents
    
    def save_documents_as_jsonl(self, documents: List[DocumentClass], filename: str = None):
        """Save all Documents as a JSONL file"""
        
        if filename is None:
            filename = 'llama_documents.jsonl'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                doc_data = {
                    'doc_id': doc.doc_id,
                    'text': doc.text,
                    'metadata': doc.metadata
                }
                f.write(json.dumps(doc_data, ensure_ascii=False, default=str) + '\n')
        
        logger.info(f"Saved {len(documents)} Documents to: {filepath}")
        return filepath
    
    def save_documents_as_markdown_files(self, documents: List[DocumentClass], prefix: str = "row"):
        """Save each Document as an individual markdown file"""
        
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
    
    def _format_document_as_markdown(self, doc: DocumentClass) -> str:
        """Format a Document as markdown content"""
        
        # Start with a title using doc_id
        title = f"# Document ID: {doc.doc_id}"
        
        markdown_lines = [title, ""]
        
        # Add the main content
        if doc.text:
            markdown_lines.append("## Content")
            markdown_lines.append("")
            
            # Format each line of the text content
            for line in doc.text.split('\n'):
                if line.strip():
                    if ':' in line:
                        # Format as bullet point for field-value pairs
                        parts = line.split(':', 1)
                        field = parts[0].strip()
                        value = parts[1].strip()
                        markdown_lines.append(f"- **{field}**: {value}")
                    else:
                        markdown_lines.append(f"- {line.strip()}")
            
            markdown_lines.append("")
        
        # Add metadata section
        if doc.metadata:
            markdown_lines.append("## Metadata")
            markdown_lines.append("")
            
            # Group metadata into categories
            system_metadata = {}
            business_metadata = {}
            
            for key, value in doc.metadata.items():
                if value is None:
                    continue
                    
                if key in ['source', 'created_at', 'doc_type', 'row_index']:
                    system_metadata[key] = value
                else:
                    business_metadata[key] = value
            
            # Add business metadata first
            if business_metadata:
                markdown_lines.append("### Record Information")
                markdown_lines.append("")
                
                for key, value in business_metadata.items():
                    formatted_key = key.replace('_', ' ').title()
                    markdown_lines.append(f"- **{formatted_key}**: {value}")
                
                markdown_lines.append("")
            
            # Add system metadata
            if system_metadata:
                markdown_lines.append("### System Information")
                markdown_lines.append("")
                
                for key, value in system_metadata.items():
                    formatted_key = key.replace('_', ' ').title()
                    markdown_lines.append(f"- **{formatted_key}**: {value}")
                
                markdown_lines.append("")
        
        return '\n'.join(markdown_lines)

def main():
    """Main execution function"""
    
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
    logger.info(f"Document class: {DocumentClass.__name__}")
    
    # Create converter
    converter = LlamaIndexCSVConverter(input_csv, output_dir)
    
    # Process documents
    documents = converter.process_csv_to_documents(batch_size=500)
    
    # Save documents as JSONL
    converter.save_documents_as_jsonl(documents, "candidate_profiles_llama_simple.jsonl")
    
    # Save documents as individual markdown files
    markdown_files = converter.save_documents_as_markdown_files(documents)
    
    logger.info("LlamaIndex Document conversion completed successfully!")
    logger.info(f"Total Documents created: {len(documents)}")
    logger.info(f"JSONL output saved to: {output_dir}")
    logger.info(f"Markdown files saved: {len(markdown_files)} files in {output_dir}/llama_markdown_files/")

if __name__ == "__main__":
    main() 