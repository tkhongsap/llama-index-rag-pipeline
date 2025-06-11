"""
Standalone section-based parser for Thai land deed documents.
Independent implementation to avoid import dependencies.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


class StandaloneLandDeedSectionParser:
    """Standalone section-aware parser for structured Thai land deed documents."""
    
    def __init__(
        self, 
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_section_size: int = 50
    ):
        """
        Initialize section parser.
        
        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks
            min_section_size: Minimum size for standalone sections
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_size = min_section_size
        
        # Create standard sentence splitter as fallback
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Section patterns for Thai land deed documents (updated to match actual text format)
        self.section_patterns = {
            "deed_info": r"## ข้อมูลโฉนด \(Deed Information\)(.*?)(?=##|\Z)",
            "location": r"## ที่ตั้ง \(Location\)(.*?)(?=##|\Z)",
            "geolocation": r"## พิกัดภูมิศาสตร์ \(Geolocation\)(.*?)(?=##|\Z)",
            "land_details": r"## รายละเอียดที่ดิน \(Land Details\)(.*?)(?=##|\Z)",
            "area_measurements": r"## ขนาดพื้นที่ \(Area Measurements\)(.*?)(?=##|\Z)",
            "classification": r"## การจำแนกประเภท \(Classification\)(.*?)(?=##|\Z)",
            "dates": r"## วันที่สำคัญ \(Important Dates\)(.*?)(?=##|\Z)",
            "financial": r"## ข้อมูลการเงิน \(Financial Information\)(.*?)(?=##|\Z)",
            "additional": r"## ข้อมูลเพิ่มเติม \(Additional Information\)(.*?)(?=##|\Z)",
            "metadata": r"## ข้อมูลเมตา \(Document Metadata\)(.*?)(?=##|\Z)"
        }
        
        # Enable generic section parsing as well
        self.enable_generic_section_parsing = True
    
    def parse_document_to_sections(
        self, 
        document_text: str,
        metadata: Dict[str, Any]
    ) -> List[TextNode]:
        """
        Parse a document into section-based chunks.
        
        Args:
            document_text: Full text of the document
            metadata: Document metadata
            
        Returns:
            List of TextNode objects with section-aware metadata
        """
        nodes = []
        
        # Create key info chunk first (always most important)
        key_info_chunk = self._create_key_info_chunk(document_text, metadata)
        nodes.append(key_info_chunk)
        
        # Parse sections from the structured text
        section_chunks = self._extract_section_chunks(document_text, metadata)
        nodes.extend(section_chunks)
        
        # If no sections found, fall back to sentence splitting
        if len(nodes) == 1:  # Only key info chunk
            logger.warning(f"No sections found in document {metadata.get('doc_id', 'unknown')}, using sentence splitting")
            fallback_nodes = self._fallback_to_sentence_splitting(document_text, metadata)
            nodes.extend(fallback_nodes)
        
        return nodes
    
    def _create_key_info_chunk(
        self, 
        document_text: str, 
        metadata: Dict[str, Any]
    ) -> TextNode:
        """Create a composite chunk with most important information for retrieval."""
        
        # Extract key searchable information
        key_elements = []
        
        # Document ID and type
        if 'deed_id' in metadata and metadata['deed_id']:
            key_elements.append(f"โฉนดเลขที่: {metadata['deed_id']}")
        
        if 'deed_serial_no' in metadata and metadata['deed_serial_no']:
            key_elements.append(f"เลขที่อนุกรม: {metadata['deed_serial_no']}")
            
        if 'deed_type' in metadata and metadata['deed_type']:
            key_elements.append(f"ประเภท: {metadata['deed_type']}")
        
        # Location information
        if 'location_hierarchy' in metadata and metadata['location_hierarchy']:
            key_elements.append(f"ที่ตั้ง: {metadata['location_hierarchy']}")
        elif 'province' in metadata and metadata['province']:
            location_parts = [str(metadata['province'])]
            if 'district' in metadata and metadata['district']:
                location_parts.append(str(metadata['district']))
            if 'subdistrict' in metadata and metadata['subdistrict']:
                location_parts.append(str(metadata['subdistrict']))
            key_elements.append(f"ที่ตั้ง: {' > '.join(location_parts)}")
        
        # Coordinates
        if 'coordinates_formatted' in metadata and metadata['coordinates_formatted']:
            key_elements.append(f"พิกัด: {metadata['coordinates_formatted']}")
        elif 'longitude' in metadata and 'latitude' in metadata and metadata['longitude'] and metadata['latitude']:
            key_elements.append(f"พิกัด: {metadata['latitude']}, {metadata['longitude']}")
        
        # Area information
        if 'area_formatted' in metadata and metadata['area_formatted']:
            key_elements.append(f"เนื้อที่: {metadata['area_formatted']}")
        elif 'deed_rai' in metadata or 'deed_ngan' in metadata or 'deed_wa' in metadata:
            area_parts = []
            if metadata.get('deed_rai'):
                area_parts.append(f"{metadata['deed_rai']} ไร่")
            if metadata.get('deed_ngan'):
                area_parts.append(f"{metadata['deed_ngan']} งาน")
            if metadata.get('deed_wa'):
                area_parts.append(f"{metadata['deed_wa']} ตารางวา")
            if area_parts:
                key_elements.append(f"เนื้อที่: {' '.join(area_parts)}")
        
        # Land category
        if 'land_main_category' in metadata and metadata['land_main_category']:
            key_elements.append(f"หมวดหมู่ที่ดิน: {metadata['land_main_category']}")
            
        # Deed group type
        if 'deed_group_type' in metadata and metadata['deed_group_type']:
            key_elements.append(f"ประเภทกลุ่ม: {metadata['deed_group_type']}")
        
        # Create key info text
        key_info_text = "\n".join(key_elements) if key_elements else "ข้อมูลสำคัญไม่พร้อมใช้งาน"
        
        # Enhanced metadata for key info chunk
        key_metadata = {
            **metadata,
            "chunk_type": "key_info",
            "section": "key_info",
            "chunk_index": 0,
            "is_primary_chunk": True
        }
        
        return TextNode(
            text=key_info_text,
            metadata=key_metadata
        )
    
    def _extract_section_chunks(
        self, 
        document_text: str, 
        base_metadata: Dict[str, Any]
    ) -> List[TextNode]:
        """Extract individual section chunks from structured document text."""
        section_nodes = []
        chunk_index = 1  # Start after key info chunk
        
        logger.info(f"🔍 Looking for sections in document {base_metadata.get('deed_id', 'unknown')}")
        
        # Try specific patterns first
        found_sections = set()
        for section_name, pattern in self.section_patterns.items():
            match = re.search(pattern, document_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                section_content = match.group(1).strip()
                logger.info(f"✅ Found section '{section_name}': {len(section_content)} chars")
                found_sections.add(section_name)
                
                # Skip nearly empty sections
                if len(section_content) < self.min_section_size:
                    logger.info(f"⚠️ Skipping section '{section_name}': too small ({len(section_content)} < {self.min_section_size})")
                    continue
                
                # If section is too large, split it further
                if len(section_content) > self.chunk_size * 2:
                    sub_chunks = self._split_large_section(
                        section_content, 
                        section_name, 
                        base_metadata, 
                        chunk_index
                    )
                    section_nodes.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                else:
                    # Create single section chunk
                    section_metadata = {
                        **base_metadata,
                        "chunk_type": "section",
                        "section": section_name,
                        "chunk_index": chunk_index,
                        "section_size": len(section_content)
                    }
                    
                    section_title = self._get_section_title(section_name)
                    section_node = TextNode(
                        text=f"## {section_title}\n{section_content}",
                        metadata=section_metadata
                    )
                    section_nodes.append(section_node)
                    chunk_index += 1
            else:
                logger.info(f"❌ Section '{section_name}' not found")
        
        # If generic parsing is enabled and we have few sections, try to catch all ## headers
        if self.enable_generic_section_parsing and len(section_nodes) < 3:
            logger.info("🔄 Enabling generic section parsing to catch all ## headers")
            generic_sections = self._extract_generic_sections(document_text, base_metadata, found_sections, chunk_index)
            section_nodes.extend(generic_sections)
        
        return section_nodes
    
    def _extract_generic_sections(
        self, 
        document_text: str, 
        base_metadata: Dict[str, Any], 
        found_sections: set, 
        start_chunk_index: int
    ) -> List[TextNode]:
        """Extract any sections that start with ## that weren't caught by specific patterns."""
        generic_sections = []
        chunk_index = start_chunk_index
        
        # Find all ## headers in the document
        header_pattern = r"## ([^\n]+)\n(.*?)(?=##|\Z)"
        matches = re.finditer(header_pattern, document_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            header_text = match.group(1).strip()
            content = match.group(2).strip()
            
            # Skip if content is too small
            if len(content) < self.min_section_size:
                logger.info(f"⚠️ Skipping generic section '{header_text}': too small ({len(content)} < {self.min_section_size})")
                continue
            
            # Create a section name from the header
            section_name = self._create_section_name_from_header(header_text)
            
            # Skip if this section was already found by specific patterns
            if section_name in found_sections:
                continue
            
            logger.info(f"✅ Found generic section '{section_name}' ('{header_text}'): {len(content)} chars")
            
            # If section is too large, split it further
            if len(content) > self.chunk_size * 2:
                sub_chunks = self._split_large_section(
                    content, 
                    section_name, 
                    base_metadata, 
                    chunk_index
                )
                generic_sections.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                # Create single section chunk
                section_metadata = {
                    **base_metadata,
                    "chunk_type": "section",
                    "section": section_name,
                    "chunk_index": chunk_index,
                    "section_size": len(content),
                    "original_header": header_text,
                    "parsing_method": "generic"
                }
                
                section_node = TextNode(
                    text=f"## {header_text}\n{content}",
                    metadata=section_metadata
                )
                generic_sections.append(section_node)
                chunk_index += 1
        
        logger.info(f"📊 Generic parsing found {len(generic_sections)} additional sections")
        return generic_sections
    
    def _create_section_name_from_header(self, header_text: str) -> str:
        """Create a section name from header text."""
        # Extract the Thai part before parentheses
        thai_part = re.split(r'\s*\(', header_text)[0].strip()
        
        # Convert to lowercase and replace spaces/special chars with underscores
        section_name = re.sub(r'[^\w\u0E00-\u0E7F]+', '_', thai_part.lower())
        section_name = re.sub(r'_+', '_', section_name).strip('_')
        
        # If empty or too short, use full header
        if len(section_name) < 3:
            section_name = re.sub(r'[^\w\u0E00-\u0E7F]+', '_', header_text.lower())
            section_name = re.sub(r'_+', '_', section_name).strip('_')
        
        return section_name if section_name else "unknown_section"
    
    def _split_large_section(
        self, 
        section_content: str, 
        section_name: str, 
        base_metadata: Dict[str, Any], 
        start_chunk_index: int
    ) -> List[TextNode]:
        """Split large sections using sentence splitter while preserving section context."""
        
        # Create temporary document for sentence splitting
        from llama_index.core.schema import Document
        temp_doc = Document(text=section_content)
        sub_chunks = self.sentence_splitter.get_nodes_from_documents([temp_doc])
        
        section_nodes = []
        for i, sub_chunk in enumerate(sub_chunks):
            sub_metadata = {
                **base_metadata,
                "chunk_type": "section_part",
                "section": section_name,
                "chunk_index": start_chunk_index + i,
                "sub_chunk_index": i,
                "total_sub_chunks": len(sub_chunks),
                "section_size": len(section_content)
            }
            
            # Prepend section title for context
            section_title = self._get_section_title(section_name)
            chunk_text = f"## {section_title}\n{sub_chunk.text}"
            
            section_node = TextNode(
                text=chunk_text,
                metadata=sub_metadata
            )
            section_nodes.append(section_node)
        
        return section_nodes
    
    def _fallback_to_sentence_splitting(
        self, 
        document_text: str, 
        base_metadata: Dict[str, Any]
    ) -> List[TextNode]:
        """Fallback to sentence splitting if no sections found."""
        
        from llama_index.core.schema import Document
        temp_doc = Document(text=document_text)
        chunks = self.sentence_splitter.get_nodes_from_documents([temp_doc])
        
        fallback_nodes = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "chunk_type": "fallback",
                "section": "content",
                "chunk_index": i + 1,  # After key info chunk
                "fallback_chunk": True
            }
            
            fallback_node = TextNode(
                text=chunk.text,
                metadata=chunk_metadata
            )
            fallback_nodes.append(fallback_node)
        
        return fallback_nodes
    
    def _get_section_title(self, section_name: str) -> str:
        """Get Thai section title from section name."""
        title_map = {
            "deed_info": "ข้อมูลโฉนด (Deed Information)",
            "location": "ที่ตั้ง (Location)",
            "geolocation": "พิกัดภูมิศาสตร์ (Geolocation)",
            "land_details": "รายละเอียดที่ดิน (Land Details)",
            "area_measurements": "ขนาดพื้นที่ (Area Measurements)",
            "classification": "การจำแนกประเภท (Classification)",
            "dates": "วันที่สำคัญ (Important Dates)",
            "financial": "ข้อมูลการเงิน (Financial Information)",
            "additional": "ข้อมูลเพิ่มเติม (Additional Information)"
        }
        return title_map.get(section_name, section_name)
    
    def get_chunking_statistics(self, nodes: List[TextNode]) -> Dict[str, Any]:
        """Generate statistics about the chunking results."""
        
        stats = {
            "total_chunks": len(nodes),
            "chunk_types": {},
            "sections": {},
            "average_chunk_size": 0,
            "size_distribution": {
                "small": 0,  # < 200 chars
                "medium": 0,  # 200-800 chars
                "large": 0   # > 800 chars
            }
        }
        
        total_size = 0
        
        for node in nodes:
            chunk_type = node.metadata.get("chunk_type", "unknown")
            section = node.metadata.get("section", "unknown")
            chunk_size = len(node.text)
            
            # Count chunk types
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
            
            # Count sections
            stats["sections"][section] = stats["sections"].get(section, 0) + 1
            
            # Size statistics
            total_size += chunk_size
            if chunk_size < 200:
                stats["size_distribution"]["small"] += 1
            elif chunk_size <= 800:
                stats["size_distribution"]["medium"] += 1
            else:
                stats["size_distribution"]["large"] += 1
        
        if nodes:
            stats["average_chunk_size"] = total_size / len(nodes)
        
        return stats


# Simple document class to avoid dependencies
class SimpleDocument:
    """Simple document class that mimics LlamaIndex Document structure"""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.id = metadata.get('doc_id', self._generate_id())
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content"""
        import hashlib
        content = f"{self.text}{str(self.metadata)}"
        return hashlib.md5(content.encode()).hexdigest()[:16] 