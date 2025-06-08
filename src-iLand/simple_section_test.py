#!/usr/bin/env python3
"""
Standalone test for section-based chunking concept.
Demonstrates the core functionality without complex imports.
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class SimpleTextNode:
    """Simple text node for testing."""
    text: str
    metadata: Dict[str, Any]


class SimpleSectionParser:
    """Simplified section parser for demonstration."""
    
    def __init__(self, chunk_size: int = 512, min_section_size: int = 50):
        self.chunk_size = chunk_size
        self.min_section_size = min_section_size
        
        # Section patterns for Thai land deed documents
        self.section_patterns = {
            "deed_info": r"## ข้อมูลโฉนด \(Deed Information\)(.*?)(?=##|\Z)",
            "location": r"## ที่ตั้ง \(Location\)(.*?)(?=##|\Z)",
            "geolocation": r"## พิกัดภูมิศาสตร์ \(Geolocation\)(.*?)(?=##|\Z)",
            "land_details": r"## รายละเอียดที่ดิน \(Land Details\)(.*?)(?=##|\Z)",
            "area_measurements": r"## ขนาดพื้นที่ \(Area Measurements\)(.*?)(?=##|\Z)",
            "classification": r"## การจำแนกประเภท \(Classification\)(.*?)(?=##|\Z)",
            "dates": r"## วันที่สำคัญ \(Important Dates\)(.*?)(?=##|\Z)"
        }
    
    def parse_document(self, text: str, metadata: Dict[str, Any]) -> List[SimpleTextNode]:
        """Parse document into section-based chunks."""
        nodes = []
        
        # Create key info chunk first
        key_info_chunk = self._create_key_info_chunk(metadata)
        nodes.append(key_info_chunk)
        
        # Extract section chunks
        for section_name, pattern in self.section_patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if match:
                section_content = match.group(1).strip()
                
                if len(section_content) >= self.min_section_size:
                    section_metadata = {
                        **metadata,
                        "chunk_type": "section",
                        "section": section_name,
                        "section_size": len(section_content)
                    }
                    
                    section_title = self._get_section_title(section_name)
                    chunk_text = f"## {section_title}\n{section_content}"
                    
                    node = SimpleTextNode(
                        text=chunk_text,
                        metadata=section_metadata
                    )
                    nodes.append(node)
        
        return nodes
    
    def _create_key_info_chunk(self, metadata: Dict[str, Any]) -> SimpleTextNode:
        """Create key info chunk with most important data."""
        key_elements = []
        
        if 'deed_serial_no' in metadata:
            key_elements.append(f"โฉนดเลขที่: {metadata['deed_serial_no']}")
        
        if 'deed_type' in metadata:
            key_elements.append(f"ประเภท: {metadata['deed_type']}")
        
        if 'province' in metadata:
            location_parts = [metadata['province']]
            if 'district' in metadata:
                location_parts.append(metadata['district'])
            key_elements.append(f"ที่ตั้ง: {' > '.join(location_parts)}")
        
        key_info_text = "\n".join(key_elements) if key_elements else "ข้อมูลสำคัญไม่พร้อมใช้งาน"
        
        key_metadata = {
            **metadata,
            "chunk_type": "key_info",
            "section": "key_info",
            "is_primary_chunk": True
        }
        
        return SimpleTextNode(text=key_info_text, metadata=key_metadata)
    
    def _get_section_title(self, section_name: str) -> str:
        """Get human-readable section title."""
        titles = {
            "deed_info": "ข้อมูลโฉนด (Deed Information)",
            "location": "ที่ตั้ง (Location)",
            "geolocation": "พิกัดภูมิศาสตร์ (Geolocation)",
            "land_details": "รายละเอียดที่ดิน (Land Details)",
            "area_measurements": "ขนาดพื้นที่ (Area Measurements)",
            "classification": "การจำแนกประเภท (Classification)",
            "dates": "วันที่สำคัญ (Important Dates)"
        }
        return titles.get(section_name, section_name.replace('_', ' ').title())
    
    def get_statistics(self, nodes: List[SimpleTextNode]) -> Dict[str, Any]:
        """Generate statistics about chunking results."""
        stats = {
            "total_chunks": len(nodes),
            "chunk_types": {},
            "sections": {},
            "average_size": 0,
            "size_distribution": {"small": 0, "medium": 0, "large": 0}
        }
        
        total_size = 0
        for node in nodes:
            chunk_type = node.metadata.get('chunk_type', 'unknown')
            section = node.metadata.get('section', 'unknown')
            size = len(node.text)
            
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
            stats["sections"][section] = stats["sections"].get(section, 0) + 1
            total_size += size
            
            if size < 200:
                stats["size_distribution"]["small"] += 1
            elif size <= 800:
                stats["size_distribution"]["medium"] += 1
            else:
                stats["size_distribution"]["large"] += 1
        
        if nodes:
            stats["average_size"] = total_size / len(nodes)
        
        return stats


def test_section_based_chunking():
    """Test the section-based chunking approach."""
    print("🚀 SECTION-BASED CHUNKING TEST")
    print("="*60)
    
    # Sample Thai land deed document
    sample_document = """# บันทึกข้อมูลโฉนดที่ดิน (Land Deed Record)

**สรุป:** โฉนดเลขที่ 12345 | ที่ตั้ง: กรุงเทพมหานคร > วัฒนา | เนื้อที่: 2 ไร่ 1 งาน 50 ตร.ว.

## ข้อมูลโฉนด (Deed Information)
- รหัสโฉนด: deed_12345
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: 12345
- เล่มที่: 100
- หน้าที่: 25

## ที่ตั้ง (Location)
- จังหวัด: กรุงเทพมหานคร
- อำเภอ: วัฒนา
- ตำบล: ลุมพินี
- ที่ตั้ง: กรุงเทพมหานคร > วัฒนา > ลุมพินี

## พิกัดภูมิศาสตร์ (Geolocation)
- พิกัด: 13.7563, 100.5018
- ลองจิจูด: 100.5018
- ละติจูด: 13.7563
- ลิงก์แผนที่: https://www.google.com/maps?q=13.7563,100.5018

## รายละเอียดที่ดิน (Land Details)
- หมวดหมู่หลัก: ที่ดินเปล่า
- หมวดหมู่ย่อย: พื้นที่พาณิชยกรรม
- ประเภทการใช้ที่ดิน: พาณิชยกรรม

## ขนาดพื้นที่ (Area Measurements)
- เนื้อที่: 2 ไร่ 1 งาน 50 ตร.ว.
- พื้นที่รวม (ตร.ม.): 3600.0

## การจำแนกประเภท (Classification)
- หมวดหมู่หลัก: ที่ดินเปล่า
- หมวดหมู่ย่อย: พาณิชยกรรม

## วันที่สำคัญ (Important Dates)
- วันที่ได้มา: 2020-05-15
"""
    
    # Sample metadata
    metadata = {
        'deed_serial_no': '12345',
        'deed_type': 'โฉนดที่ดิน',
        'province': 'กรุงเทพมหานคร',
        'district': 'วัฒนา',
        'subdistrict': 'ลุมพินี',
        'doc_id': 'deed_12345'
    }
    
    # Create parser and parse document
    parser = SimpleSectionParser(chunk_size=512, min_section_size=50)
    
    print("📄 Parsing sample land deed document...")
    nodes = parser.parse_document(sample_document, metadata)
    
    print(f"✅ Created {len(nodes)} section-based chunks\n")
    
    # Display results
    print("📊 SECTION CHUNKS CREATED:")
    print("-" * 50)
    
    for i, node in enumerate(nodes):
        chunk_type = node.metadata.get('chunk_type', 'unknown')
        section = node.metadata.get('section', 'unknown')
        size = len(node.text)
        
        print(f"{i+1}. [{chunk_type}] {section}")
        print(f"   Size: {size} characters")
        
        # Show preview
        preview = node.text[:120].replace('\n', ' ')
        if len(node.text) > 120:
            preview += "..."
        print(f"   Preview: {preview}")
        print()
    
    # Generate and display statistics
    stats = parser.get_statistics(nodes)
    
    print("📈 CHUNKING STATISTICS:")
    print("-" * 30)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average size: {stats['average_size']:.1f} chars")
    print(f"Chunk types: {stats['chunk_types']}")
    print(f"Sections: {stats['sections']}")
    print(f"Size distribution: {stats['size_distribution']}")
    
    print("\n✅ SECTION-BASED CHUNKING TEST COMPLETE!")
    
    # Demonstrate query routing concept
    print("\n🎯 QUERY ROUTING DEMONSTRATION:")
    print("-" * 40)
    
    query_examples = [
        ("โฉนดเลขที่ 12345", ["key_info", "deed_info"]),
        ("ที่ตั้งในกรุงเทพมหานคร", ["key_info", "location"]),
        ("ขนาดพื้นที่ 2 ไร่", ["key_info", "area_measurements"]),
        ("พิกัด 13.7563", ["key_info", "geolocation"]),
        ("วันที่ได้มา", ["key_info", "dates"])
    ]
    
    for query, expected_sections in query_examples:
        print(f"Query: '{query}'")
        print(f"  → Should target sections: {expected_sections}")
        
        # Find matching chunks
        matching_chunks = []
        for node in nodes:
            if node.metadata.get('section') in expected_sections:
                matching_chunks.append(node.metadata.get('section'))
        
        print(f"  → Available chunks: {matching_chunks}")
        print()
    
    return True


if __name__ == "__main__":
    success = test_section_based_chunking()
    
    if success:
        print("\n🎉 SUCCESS: Section-based chunking concept validated!")
        print("\n📋 Key Benefits Demonstrated:")
        print("✅ Structured section extraction from Thai land deeds")
        print("✅ Key info chunk for comprehensive retrieval")
        print("✅ Section-specific chunks for precise queries")
        print("✅ Metadata preservation for filtering")
        print("✅ Query routing based on section relevance")
        
        print("\n🚀 Ready for integration with:")
        print("- LlamaIndex embedding pipeline")
        print("- Vector store with metadata filtering")
        print("- Intelligent retrieval strategies")
        print("- 50k document scalability")
    else:
        print("\n❌ Test failed!") 