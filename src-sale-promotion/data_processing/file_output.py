"""
File output manager for saving documents in JSONL and markdown formats.

Handles dual output: JSONL for Phase 2 and markdown for human review.
"""

import os
import json
import logging
from typing import List
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .models import SimpleDocument, DatasetConfig
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from models import SimpleDocument, DatasetConfig

logger = logging.getLogger(__name__)


class FileOutputManager:
    """
    Handles file output operations including JSONL and Markdown generation.
    
    Generates dual outputs:
    1. JSONL file (primary, for Phase 2)
    2. Individual markdown files (secondary, for human QA)
    """
    
    def __init__(self, output_dir: str, dataset_config: DatasetConfig):
        """
        Initialize file output manager.
        
        Args:
            output_dir: Base output directory
            dataset_config: Dataset configuration
        """
        self.output_dir = output_dir
        self.dataset_config = dataset_config
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "documents", "markdown_files"), exist_ok=True)
    
    def save_documents_as_jsonl(
        self,
        documents: List[SimpleDocument],
        filename: str = "documents.jsonl"
    ) -> str:
        """
        Save all documents as a single JSONL file (primary output for Phase 2).
        
        Args:
            documents: List of SimpleDocument objects
            filename: Output filename
        
        Returns:
            Path to saved JSONL file
        """
        filepath = os.path.join(self.output_dir, "documents", filename)
        
        logger.info(f"Saving {len(documents)} documents to JSONL: {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                # Use SimpleDocument.to_dict() method
                doc_data = doc.to_dict()
                
                # Write as single-line JSON
                json_line = json.dumps(doc_data, ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"✓ Saved {len(documents)} documents to JSONL")
        return filepath
    
    def save_documents_as_markdown_files(
        self,
        documents: List[SimpleDocument],
        prefix: str = "promotion"
    ) -> List[str]:
        """
        Save each document as an individual markdown file (for human QA).
        
        Args:
            documents: List of SimpleDocument objects
            prefix: Filename prefix (default: "promotion")
        
        Returns:
            List of paths to saved markdown files
        """
        if not documents:
            logger.warning("No documents to save as markdown files")
            return []
        
        markdown_dir = os.path.join(self.output_dir, "documents", "markdown_files")
        os.makedirs(markdown_dir, exist_ok=True)
        
        logger.info(f"Saving {len(documents)} documents as markdown files")
        
        saved_files = []
        
        # Determine padding based on total number of documents
        total_docs = len(documents)
        padding = 3  # Fixed 3-digit padding per PRD requirement (promotion_001.md to promotion_071.md)
        
        for idx, doc in enumerate(documents, 1):
            # Create filename: promotion_001.md, promotion_002.md, etc.
            filename = f"{prefix}_{str(idx).zfill(padding)}.md"
            filepath = os.path.join(markdown_dir, filename)
            
            # Use SimpleDocument.to_markdown() method
            markdown_content = doc.to_markdown()
            
            # Write markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            saved_files.append(filepath)
            
            # Log progress for every 10 files
            if idx % 10 == 0:
                logger.info(f"  Saved {idx}/{total_docs} markdown files")
        
        logger.info(f"✓ Saved {len(saved_files)} markdown files to {markdown_dir}")
        return saved_files
    
    def validate_jsonl_output(self, jsonl_path: str) -> bool:
        """
        Validate JSONL output format.
        
        Args:
            jsonl_path: Path to JSONL file
        
        Returns:
            True if valid
        """
        logger.info(f"Validating JSONL output: {jsonl_path}")
        
        if not os.path.exists(jsonl_path):
            logger.error(f"JSONL file not found: {jsonl_path}")
            return False
        
        try:
            line_count = 0
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Parse JSON
                    data = json.loads(line)
                    
                    # Check required fields
                    if 'id' not in data:
                        logger.error(f"Line {line_num}: Missing 'id' field")
                        return False
                    
                    if 'text' not in data:
                        logger.error(f"Line {line_num}: Missing 'text' field")
                        return False
                    
                    if 'metadata' not in data:
                        logger.error(f"Line {line_num}: Missing 'metadata' field")
                        return False
                    
                    # Check text is non-empty
                    if not data['text']:
                        logger.error(f"Line {line_num}: Empty 'text' field")
                        return False
                    
                    # Check metadata has sufficient fields
                    if len(data['metadata']) < 15:
                        logger.warning(f"Line {line_num}: Only {len(data['metadata'])} metadata fields (expected 15+)")
                    
                    line_count += 1
            
            logger.info(f"✓ JSONL validation passed: {line_count} documents")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def get_output_summary(self) -> dict:
        """
        Get summary of output files.
        
        Returns:
            Dictionary with output file information
        """
        summary = {
            'output_dir': self.output_dir,
            'jsonl_file': os.path.join(self.output_dir, "documents", "documents.jsonl"),
            'markdown_dir': os.path.join(self.output_dir, "documents", "markdown_files"),
            'files_exist': {}
        }
        
        # Check JSONL file
        jsonl_path = summary['jsonl_file']
        summary['files_exist']['jsonl'] = os.path.exists(jsonl_path)
        if summary['files_exist']['jsonl']:
            summary['jsonl_size_kb'] = os.path.getsize(jsonl_path) / 1024
        
        # Check markdown directory
        markdown_dir = summary['markdown_dir']
        if os.path.exists(markdown_dir):
            markdown_files = [f for f in os.listdir(markdown_dir) if f.endswith('.md')]
            summary['files_exist']['markdown'] = len(markdown_files) > 0
            summary['markdown_file_count'] = len(markdown_files)
        else:
            summary['files_exist']['markdown'] = False
            summary['markdown_file_count'] = 0
        
        return summary

