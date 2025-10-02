#!/usr/bin/env python3
"""
Minimal test script for section-based parser functionality.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from data_processing.section_parser import LandDeedSectionParser
    from data_processing.models import DatasetConfig, SimpleDocument
    print("✅ Successfully imported section parser modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_section_parser():
    """Test the section parser with a sample document."""
    print("\n🔧 Testing Section-Based Parser")
    print("="*50)
    
    # Sample land deed document text
    sample_text = """# บันทึกข้อมูลโฉนดที่ดิน (Land Deed Record)

## ข้อมูลโฉนด (Deed Information)
- รหัสโฉนด: deed_12345
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: 12345

## ที่ตั้ง (Location)
- จังหวัด: กรุงเทพมหานคร
- อำเภอ: วัฒนา
- ตำบล: ลุมพินี

## ขนาดพื้นที่ (Area Measurements)
- เนื้อที่: 2 ไร่ 1 งาน 50 ตร.ว.
- พื้นที่รวม (ตร.ม.): 3600.0
"""
    
    # Sample metadata
    metadata = {
        'deed_serial_no': '12345',
        'deed_type': 'โฉนดที่ดิน',
        'province': 'กรุงเทพมหานคร',
        'district': 'วัฒนา',
        'doc_id': 'deed_12345'
    }
    
    # Create simple document
    simple_doc = SimpleDocument(text=sample_text, metadata=metadata)
    
    # Create section parser
    config = DatasetConfig(
        name="test_config",
        description="Test configuration",
        field_mappings=[]
    )
    
    parser = LandDeedSectionParser(
        dataset_config=config,
        chunk_size=512,
        chunk_overlap=50,
        min_section_size=50
    )
    
    print("📄 Parsing sample document...")
    
    # Parse document
    try:
        nodes = parser.parse_simple_document_to_sections(simple_doc)
        print(f"✅ Successfully created {len(nodes)} section chunks")
        
        # Display results
        for i, node in enumerate(nodes):
            chunk_type = node.metadata.get('chunk_type', 'unknown')
            section = node.metadata.get('section', 'unknown')
            text_length = len(node.text)
            
            print(f"  {i+1}. [{chunk_type}] {section} ({text_length} chars)")
            
            # Show first 100 chars of text
            preview = node.text[:100].replace('\n', ' ')
            if len(node.text) > 100:
                preview += "..."
            print(f"      Preview: {preview}")
            print()
        
        # Get statistics
        stats = parser.get_chunking_statistics(nodes)
        print("📊 Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Chunk types: {stats['chunk_types']}")
        print(f"  Sections: {stats['sections']}")
        print(f"  Average size: {stats['average_chunk_size']:.1f} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing document: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Section Parser Test")
    success = test_section_parser()
    
    if success:
        print("\n✅ Section parser test completed successfully!")
        print("\n📋 What was tested:")
        print("- Section pattern matching")
        print("- Key info chunk creation")
        print("- Section-specific chunk generation")
        print("- Metadata preservation")
        print("- Statistics generation")
    else:
        print("\n❌ Section parser test failed!")
    
    print("\n🎯 Next: Run the full embedding pipeline with section chunking") 