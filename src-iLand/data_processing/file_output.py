import os
import json
import re
import logging
from typing import List, Dict, Any
from datetime import datetime
from .models import SimpleDocument, DatasetConfig

logger = logging.getLogger(__name__)


class FileOutputManager:
    """Handles file output operations including JSONL and Markdown generation"""
    
    def __init__(self, output_dir: str, dataset_config: DatasetConfig):
        self.output_dir = output_dir
        self.dataset_config = dataset_config
    
    def save_documents_as_jsonl(self, documents: List[SimpleDocument], filename: str = None) -> str:
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

    def save_documents_as_markdown_files(self, documents: List[SimpleDocument], prefix: str = "deed", batch_size: int = 1000) -> List[str]:
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