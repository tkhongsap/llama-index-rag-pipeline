"""
Document loader module for sales promotion documents.
Handles loading JSONL files and converting SimpleDocument to LlamaIndex Document objects.

Following PRD Section 6.1: document_loader.py specification
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document


class PromotionDocumentLoader:
    """
    Loads sales promotion documents from JSONL file.
    
    Converts Phase 1 SimpleDocument format (JSONL) to LlamaIndex Document objects.
    Validates document structure and metadata completeness.
    """
    
    def __init__(self):
        """Initialize document loader."""
        self.documents_loaded = 0
        self.validation_errors = []
    
    def load_documents_from_jsonl(
        self,
        jsonl_path: Path
    ) -> List[Document]:
        """
        Load all promotion documents from JSONL file.
        
        Args:
            jsonl_path: Path to documents.jsonl file
            
        Returns:
            List of LlamaIndex Document objects
            
        Raises:
            FileNotFoundError: If JSONL file doesn't exist
            ValueError: If validation fails
        """
        # Validate file exists
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        print(f"📂 Loading documents from: {jsonl_path}")
        
        documents = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON
                    doc_data = json.loads(line.strip())
                    
                    # Validate structure
                    self._validate_document_structure(doc_data, line_num)
                    
                    # Convert to LlamaIndex Document
                    doc = self._convert_to_llamaindex_document(doc_data)
                    documents.append(doc)
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_num}: Invalid JSON - {e}"
                    self.validation_errors.append(error_msg)
                    raise ValueError(error_msg)
                
                except ValueError as e:
                    error_msg = f"Line {line_num}: {e}"
                    self.validation_errors.append(error_msg)
                    raise
        
        self.documents_loaded = len(documents)
        
        # Validate expected count
        if len(documents) != 71:
            raise ValueError(
                f"Expected 71 documents, but loaded {len(documents)}"
            )
        
        print(f"✅ Successfully loaded {len(documents)} promotion documents")
        return documents
    
    def _validate_document_structure(
        self,
        doc_data: dict,
        line_num: int
    ) -> None:
        """
        Validate document has required fields.
        
        Args:
            doc_data: Document dictionary
            line_num: Line number in JSONL file
            
        Raises:
            ValueError: If validation fails
        """
        # Check required top-level fields
        required_fields = ['id', 'text', 'metadata']
        for field in required_fields:
            if field not in doc_data:
                raise ValueError(f"Missing required field: '{field}'")
        
        # Validate metadata field count
        metadata = doc_data['metadata']
        if len(metadata) < 25:
            raise ValueError(
                f"Insufficient metadata fields: {len(metadata)} "
                f"(expected 25+)"
            )
        
        # Validate text is non-empty
        if not doc_data['text'] or len(doc_data['text']) == 0:
            raise ValueError("Empty 'text' field")
    
    def _convert_to_llamaindex_document(
        self,
        doc_data: dict
    ) -> Document:
        """
        Convert SimpleDocument (JSONL) to LlamaIndex Document.
        
        Args:
            doc_data: Dictionary from JSONL line
            
        Returns:
            LlamaIndex Document object
        """
        return Document(
            text=doc_data['text'],
            metadata=doc_data['metadata'],
            doc_id=doc_data['id']
        )
    
    def validate_jsonl_file(self, jsonl_path: Path) -> Dict[str, Any]:
        """
        Validate JSONL file without loading full documents.
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            Validation report dictionary
        """
        report = {
            "file_exists": jsonl_path.exists(),
            "line_count": 0,
            "valid_json_lines": 0,
            "errors": []
        }
        
        if not report["file_exists"]:
            report["errors"].append("File not found")
            return report
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                report["line_count"] += 1
                
                try:
                    doc_data = json.loads(line.strip())
                    
                    # Quick validation
                    if 'id' in doc_data and 'text' in doc_data and 'metadata' in doc_data:
                        report["valid_json_lines"] += 1
                    else:
                        report["errors"].append(
                            f"Line {line_num}: Missing required fields"
                        )
                
                except json.JSONDecodeError as e:
                    report["errors"].append(
                        f"Line {line_num}: Invalid JSON - {e}"
                    )
        
        report["is_valid"] = (
            report["line_count"] == 71 and
            report["valid_json_lines"] == 71 and
            len(report["errors"]) == 0
        )
        
        return report

