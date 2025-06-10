"""
Document loader module for iLand Thai land deed documents.
Handles loading markdown files and creating LlamaIndex Document objects.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List
from llama_index.core import Document

# Handle both module import and direct script execution
try:
    from .metadata_extractor import iLandMetadataExtractor
except ImportError:
    from metadata_extractor import iLandMetadataExtractor


class iLandDocumentLoader:
    """Loads iLand markdown documents with structured metadata extraction."""
    
    def __init__(self):
        self.metadata_extractor = iLandMetadataExtractor()
    
    def load_documents_from_files(self, file_paths: List[Path]) -> List[Document]:
        """Load documents with structured metadata extraction."""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self._load_single_document(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
        
        return documents
    
    def _load_single_document(self, file_path: Path) -> Document:
        """Load a single markdown document with metadata extraction."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract structured metadata
        metadata = self.metadata_extractor.extract_from_content(content)
        
        # Add file-level metadata
        metadata.update({
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'creation_date': datetime.fromtimestamp(file_path.stat().st_ctime).strftime('%Y-%m-%d'),
            'last_modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
        })
        
        # Extract document ID from filename or content
        filename_match = re.search(r'deed_([a-f0-9-]+)\.md$', file_path.name)
        if filename_match:
            metadata['filename_deed_id'] = filename_match.group(1)
        
        # Add document type classification
        metadata['document_type'] = 'land_deed_record'
        metadata['content_type'] = self.metadata_extractor.classify_content_types(content)
        
        # Derive additional metadata for better filtering
        metadata.update(self.metadata_extractor.derive_categories(metadata))
        
        return Document(text=content, metadata=metadata) 