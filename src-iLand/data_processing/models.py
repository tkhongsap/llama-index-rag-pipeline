import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


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