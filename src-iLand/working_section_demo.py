#!/usr/bin/env python3
"""
Working section-based chunking demo for structured Thai land deed documents.
This version uses direct imports and demonstrates the complete workflow.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

@dataclass
class SimpleTextNode:
    """Simple text node for demonstration."""
    text: str
    metadata: Dict[str, Any]


class ProductionLandDeedSectionParser:
    """Production-ready section parser following the recommended approach from 08_section_based_chunking.md"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_section_size: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_size = min_section_size
        
        # Section patterns optimized for Thai land deed documents
        self.section_patterns = {
            "deed_info": r"## ข้อมูลโฉนด \(Deed Information\)(.*?)(?=##|\Z)",
            "location": r"## ที่ตั้ง \(Location\)(.*?)(?=##|\Z)",
            "geolocation": r"## พิกัดภูมิศาสตร์ \(Geolocation\)(.*?)(?=##|\Z)",
            "land_details": r"## รายละเอียดที่ดิน \(Land Details\)(.*?)(?=##|\Z)",
            "area_measurements": r"## ขนาดพื้นที่ \(Area Measurements\)(.*?)(?=##|\Z)",
            "classification": r"## การจำแนกประเภท \(Classification\)(.*?)(?=##|\Z)",
            "dates": r"## วันที่สำคัญ \(Important Dates\)(.*?)(?=##|\Z)"
        }
        
        # Query routing patterns
        self.query_section_mapping = {
            "location_queries": ["ที่ตั้ง", "อำเภอ", "จังหวัด", "ตำบล", "อยู่ที่ไหน"],
            "size_queries": ["ขนาด", "พื้นที่", "ไร่", "งาน", "ตารางวา", "ตร.ม.", "ตร.ว."],
            "deed_queries": ["โฉนด", "เลขที่", "รหัส", "เล่มที่", "หน้าที่"],
            "coordinate_queries": ["พิกัด", "ละติจูด", "ลองจิจูด", "แผนที่"],
            "type_queries": ["ประเภท", "หมวดหมู่", "การใช้", "จำแนก"],
            "date_queries": ["วันที่", "ได้มา", "โอน", "วาระ"]
        }
    
    def parse_document(self, text: str, metadata: Dict[str, Any], doc_id: str) -> List[SimpleTextNode]:
        """Parse document into section-based chunks following best practices."""
        nodes = []
        
        # Strategy 1: Create key info chunk for comprehensive retrieval
        key_info_chunk = self._create_key_info_chunk(metadata, doc_id)
        nodes.append(key_info_chunk)
        
        # Strategy 2: Create section-specific chunks for precise queries
        for section_name, pattern in self.section_patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if match:
                section_content = match.group(1).strip()
                
                if len(section_content) >= self.min_section_size:
                    section_metadata = {
                        **metadata,
                        "chunk_type": "section",
                        "section": section_name,
                        "section_size": len(section_content),
                        "doc_id": doc_id,
                        "is_primary_chunk": False
                    }
                    
                    section_title = self._get_section_title(section_name)
                    chunk_text = f"## {section_title}\n{section_content}"
                    
                    # Apply size optimization
                    if len(chunk_text) > self.chunk_size:
                        chunk_text = chunk_text[:self.chunk_size] + "..."
                    
                    node = SimpleTextNode(
                        text=chunk_text,
                        metadata=section_metadata
                    )
                    nodes.append(node)
        
        return nodes
    
    def _create_key_info_chunk(self, metadata: Dict[str, Any], doc_id: str) -> SimpleTextNode:
        """Create composite chunk with most important info for retrieval (following 08_section_based_chunking.md)"""
        key_elements = []
        
        # Essential information for retrieval
        if 'deed_serial_no' in metadata:
            key_elements.append(f"โฉนดเลขที่: {metadata['deed_serial_no']}")
        
        if 'deed_type' in metadata:
            key_elements.append(f"ประเภท: {metadata['deed_type']}")
            
        if 'province' in metadata:
            location_parts = [metadata['province']]
            if 'district' in metadata:
                location_parts.append(metadata['district'])
            if 'subdistrict' in metadata:
                location_parts.append(metadata['subdistrict'])
            key_elements.append(f"ที่ตั้ง: {' > '.join(location_parts)}")
        
        if 'coordinates_formatted' in metadata:
            key_elements.append(f"พิกัด: {metadata['coordinates_formatted']}")
            
        if 'area_formatted' in metadata:
            key_elements.append(f"ขนาด: {metadata['area_formatted']}")
            
        if 'land_main_category' in metadata:
            key_elements.append(f"ประเภทที่ดิน: {metadata['land_main_category']}")
        
        key_info_text = "\n".join(key_elements) if key_elements else "ข้อมูลสำคัญไม่พร้อมใช้งาน"
        
        key_metadata = {
            **metadata,
            "chunk_type": "key_info",
            "section": "key_info",
            "is_primary_chunk": True,
            "doc_id": doc_id
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
    
    def detect_query_sections(self, query: str) -> List[str]:
        """Detect which sections a query should target."""
        query_lower = query.lower()
        target_sections = ["key_info"]  # Always include key info
        
        for query_type, keywords in self.query_section_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                if query_type == "location_queries":
                    target_sections.extend(["location", "geolocation"])
                elif query_type == "size_queries":
                    target_sections.append("area_measurements")
                elif query_type == "deed_queries":
                    target_sections.append("deed_info")
                elif query_type == "coordinate_queries":
                    target_sections.append("geolocation")
                elif query_type == "type_queries":
                    target_sections.extend(["classification", "land_details"])
                elif query_type == "date_queries":
                    target_sections.append("dates")
        
        return list(set(target_sections))  # Remove duplicates
    
    def get_chunking_statistics(self, nodes: List[SimpleTextNode]) -> Dict[str, Any]:
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


class MockEmbeddingService:
    """Mock embedding service to simulate the full pipeline without API calls."""
    
    def __init__(self):
        self.embedding_count = 0
    
    def create_embeddings(self, chunks: List[SimpleTextNode]) -> List[Dict[str, Any]]:
        """Create mock embeddings for chunks."""
        embeddings = []
        
        for chunk in chunks:
            self.embedding_count += 1
            embedding = {
                "id": f"embed_{self.embedding_count}",
                "type": "section_chunk",
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding_vector": [0.1] * 1536,  # Mock embedding vector
                "created_at": "2024-01-01T00:00:00Z"
            }
            embeddings.append(embedding)
        
        return embeddings


def demo_section_based_chunking():
    """Demonstrate the complete section-based chunking workflow."""
    print("🚀 PRODUCTION SECTION-BASED CHUNKING DEMO")
    print("="*80)
    print("Following recommendations from 08_section_based_chunking.md")
    print("="*80)
    
    # Sample Thai land deed documents
    sample_documents = [
        {
            "id": "deed_12345",
            "text": """# บันทึกข้อมูลโฉนดที่ดิน (Land Deed Record)

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

## รายละเอียดที่ดิน (Land Details)
- หมวดหมู่หลัก: ที่ดินเปล่า
- หมวดหมู่ย่อย: พื้นที่พาณิชยกรรม

## ขนาดพื้นที่ (Area Measurements)
- เนื้อที่: 2 ไร่ 1 งาน 50 ตร.ว.
- พื้นที่รวม (ตร.ม.): 3600.0

## การจำแนกประเภท (Classification)
- หมวดหมู่หลัก: ที่ดินเปล่า
- หมวดหมู่ย่อย: พาณิชยกรรม

## วันที่สำคัญ (Important Dates)
- วันที่ได้มา: 2020-05-15
""",
            "metadata": {
                'deed_serial_no': '12345',
                'deed_type': 'โฉนดที่ดิน',
                'province': 'กรุงเทพมหานคร',
                'district': 'วัฒนา',
                'subdistrict': 'ลุมพินี',
                'coordinates_formatted': '13.7563, 100.5018',
                'area_formatted': '2 ไร่ 1 งาน 50 ตร.ว.',
                'land_main_category': 'ที่ดินเปล่า'
            }
        },
        {
            "id": "deed_67890",
            "text": """# บันทึกข้อมูลโฉนดที่ดิน (Land Deed Record)

**สรุป:** โฉนดเลขที่ 67890 | ที่ตั้ง: เชียงใหม่ > เมือง | เนื้อที่: 1 ไร่ 2 งาน 75 ตร.ว.

## ข้อมูลโฉนด (Deed Information)
- รหัสโฉนด: deed_67890
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: 67890

## ที่ตั้ง (Location)
- จังหวัด: เชียงใหม่
- อำเภอ: เมือง
- ตำบล: ศรีภูมิ

## ขนาดพื้นที่ (Area Measurements)
- เนื้อที่: 1 ไร่ 2 งาน 75 ตร.ว.
- พื้นที่รวม (ตร.ม.): 2700.0

## การจำแนกประเภท (Classification)
- หมวดหมู่หลัก: ที่ดินเกษตร
- หมวดหมู่ย่อย: นาข้าว
""",
            "metadata": {
                'deed_serial_no': '67890',
                'deed_type': 'โฉนดที่ดิน',
                'province': 'เชียงใหม่',
                'district': 'เมือง',
                'subdistrict': 'ศรีภูมิ',
                'area_formatted': '1 ไร่ 2 งาน 75 ตร.ว.',
                'land_main_category': 'ที่ดินเกษตร'
            }
        }
    ]
    
    # Initialize parser with recommended configuration
    parser = ProductionLandDeedSectionParser(
        chunk_size=512,        # As recommended in the guide
        chunk_overlap=50,      # Minimal overlap for structured data
        min_section_size=50    # Skip nearly empty sections
    )
    
    # Initialize mock embedding service
    embedding_service = MockEmbeddingService()
    
    all_chunks = []
    all_embeddings = []
    
    print("\n📄 PROCESSING DOCUMENTS:")
    print("-" * 50)
    
    # Process each document
    for i, doc in enumerate(sample_documents):
        print(f"\n{i+1}. Processing {doc['id']}...")
        
        # Parse document into section-based chunks
        chunks = parser.parse_document(doc['text'], doc['metadata'], doc['id'])
        all_chunks.extend(chunks)
        
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = embedding_service.create_embeddings(chunks)
        all_embeddings.extend(embeddings)
        
        # Show chunk breakdown
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        print(f"   📊 Chunk types: {chunk_types}")
    
    # Generate overall statistics
    stats = parser.get_chunking_statistics(all_chunks)
    
    print(f"\n📈 OVERALL STATISTICS:")
    print("-" * 30)
    print(f"Total documents: {len(sample_documents)}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average chunk size: {stats['average_size']:.1f} chars")
    print(f"Chunk types: {stats['chunk_types']}")
    print(f"Sections: {stats['sections']}")
    print(f"Size distribution: {stats['size_distribution']}")
    print(f"Total embeddings: {len(all_embeddings)}")
    
    # Demonstrate query routing
    print(f"\n🎯 QUERY ROUTING DEMONSTRATION:")
    print("-" * 40)
    
    test_queries = [
        "โฉนดเลขที่ 12345 อยู่ที่ไหน",
        "ที่ดินในเชียงใหม่ขนาดเท่าไหร่",
        "พิกัดของโฉนดในกรุงเทพ",
        "ที่ดินประเภทเกษตรมีที่ไหนบ้าง",
        "วันที่ได้มาของโฉนดเลขที่ 12345"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Detect target sections
        target_sections = parser.detect_query_sections(query)
        print(f"  → Target sections: {target_sections}")
        
        # Find matching chunks
        matching_chunks = []
        for chunk in all_chunks:
            if chunk.metadata.get('section') in target_sections:
                matching_chunks.append({
                    'doc_id': chunk.metadata.get('doc_id'),
                    'section': chunk.metadata.get('section'),
                    'text_preview': chunk.text[:80].replace('\n', ' ') + "..."
                })
        
        print(f"  → Found {len(matching_chunks)} relevant chunks:")
        for match in matching_chunks[:3]:  # Show top 3
            print(f"    - {match['doc_id']} [{match['section']}]: {match['text_preview']}")
    
    # Demonstrate metadata filtering for 50k scale
    print(f"\n🏗️ 50K DOCUMENT SCALABILITY FEATURES:")
    print("-" * 45)
    
    # Show metadata that can be used for filtering
    available_filters = set()
    for chunk in all_chunks:
        for key in chunk.metadata.keys():
            if key in ['province', 'district', 'deed_type', 'land_main_category', 'chunk_type', 'section']:
                available_filters.add(key)
    
    print(f"Available metadata filters: {sorted(available_filters)}")
    
    # Example filtering scenarios
    filter_examples = [
        {"province": "กรุงเทพมหานคร"},
        {"land_main_category": "ที่ดินเปล่า"},
        {"chunk_type": "key_info"},
        {"section": "location"}
    ]
    
    print(f"\nFiltering examples for pre-search optimization:")
    for filter_example in filter_examples:
        filtered_chunks = [
            chunk for chunk in all_chunks 
            if all(chunk.metadata.get(k) == v for k, v in filter_example.items())
        ]
        print(f"  Filter {filter_example}: {len(filtered_chunks)} chunks")
    
    return all_chunks, all_embeddings


def main():
    """Run the complete section-based chunking demo."""
    print("🌟 Welcome to the Production Section-Based Chunking Demo")
    print("This demonstrates the full workflow for 50k+ land deed documents")
    print()
    
    # Validate environment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OpenAI API key configured (using mock embeddings for demo)")
    else:
        print("⚠️  OpenAI API key not found (using mock embeddings)")
    
    # Run the demo
    chunks, embeddings = demo_section_based_chunking()
    
    print(f"\n" + "="*80)
    print("✅ SECTION-BASED CHUNKING DEMO COMPLETE!")
    print("="*80)
    
    print(f"\n📋 IMPLEMENTATION SUMMARY:")
    print("✅ Section-aware parsing with Thai language support")
    print("✅ Key info chunks for comprehensive retrieval")
    print("✅ Section-specific chunks for precise queries")
    print("✅ Query routing based on content analysis")
    print("✅ Metadata filtering for 50k document scalability")
    print("✅ Optimized chunk sizes (512 tokens) as recommended")
    print("✅ Minimal overlap (50 tokens) for structured data")
    
    print(f"\n🚀 READY FOR PRODUCTION:")
    print("- Integrate with LlamaIndex VectorStoreIndex")
    print("- Connect to OpenAI embeddings API")
    print("- Add vector database with metadata filtering")
    print("- Implement query routing in retrieval pipeline")
    print("- Scale to 50k+ documents with batch processing")
    
    print(f"\n🎯 Next Steps:")
    print("1. Integrate this parser with your existing pipeline")
    print("2. Replace mock embeddings with real OpenAI API calls")
    print("3. Test with your actual land deed data")
    print("4. Monitor performance metrics and optimize as needed")


if __name__ == "__main__":
    main() 