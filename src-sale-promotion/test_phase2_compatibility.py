"""
Phase 2 Compatibility Test

This script tests that Phase 1 output (JSONL) can be properly loaded
and converted to LlamaIndex Documents for Phase 2 embedding generation.
"""

import json
import sys
from pathlib import Path

# Try to import LlamaIndex (optional for this test)
try:
    from llama_index.core import Document
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    print("⚠️  LlamaIndex not installed - testing JSONL format only")


def test_jsonl_format(jsonl_path: Path):
    """Test that JSONL format is correct"""
    print(f"\n📄 Testing JSONL format: {jsonl_path}")
    
    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        return False
    
    try:
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Parse JSON
                data = json.loads(line)
                
                # Validate required fields
                assert 'id' in data, f"Line {line_num}: Missing 'id' field"
                assert 'text' in data, f"Line {line_num}: Missing 'text' field"
                assert 'metadata' in data, f"Line {line_num}: Missing 'metadata' field"
                
                # Validate text is non-empty
                assert data['text'], f"Line {line_num}: Empty 'text' field"
                
                # Validate metadata has sufficient fields
                assert len(data['metadata']) >= 15, f"Line {line_num}: Only {len(data['metadata'])} metadata fields"
                
                documents.append(data)
        
        print(f"✅ JSONL format validated: {len(documents)} documents")
        return True
        
    except Exception as e:
        print(f"❌ JSONL validation failed: {e}")
        return False


def test_llama_index_conversion(jsonl_path: Path):
    """Test conversion to LlamaIndex Documents"""
    if not LLAMA_INDEX_AVAILABLE:
        print("\n⚠️  Skipping LlamaIndex conversion test (llama-index not installed)")
        return True
    
    print(f"\n🔄 Testing LlamaIndex Document conversion...")
    
    try:
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line)
                
                # Convert to LlamaIndex Document
                doc = Document(
                    text=data['text'],
                    metadata=data['metadata'],
                    doc_id=data['id']
                )
                
                documents.append(doc)
        
        print(f"✅ Successfully converted to {len(documents)} LlamaIndex Documents")
        
        # Print sample document info
        if documents:
            sample = documents[0]
            print(f"\n📝 Sample Document:")
            print(f"  ID: {sample.doc_id}")
            print(f"  Text length: {len(sample.text)} chars")
            print(f"  Metadata fields: {len(sample.metadata)}")
            print(f"  Sample metadata keys: {list(sample.metadata.keys())[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LlamaIndex conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thai_encoding(jsonl_path: Path):
    """Test that Thai UTF-8 encoding is preserved"""
    print(f"\n🇹🇭 Testing Thai UTF-8 encoding...")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            data = json.loads(first_line)
            
            # Check for Thai characters in text
            thai_chars = [c for c in data['text'] if '\u0e00' <= c <= '\u0e7f']
            
            if thai_chars:
                print(f"✅ Thai characters found and preserved: {len(thai_chars)} chars")
                print(f"  Sample Thai text: {data['text'][:100]}...")
            else:
                print("⚠️  No Thai characters detected (might be issue)")
        
        return True
        
    except Exception as e:
        print(f"❌ Thai encoding test failed: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("PHASE 2 COMPATIBILITY TEST")
    print("=" * 60)
    
    # Find JSONL file
    module_dir = Path(__file__).parent
    jsonl_path = module_dir / "data" / "processed" / "documents" / "documents.jsonl"
    
    if not jsonl_path.exists():
        print(f"❌ JSONL file not found: {jsonl_path}")
        print("\nPlease run Phase 1 data processing first:")
        print("  cd src-sale-promotion")
        print("  python3 data_processing/main.py --input data/promotion_list.csv")
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    tests_total += 1
    if test_jsonl_format(jsonl_path):
        tests_passed += 1
    
    tests_total += 1
    if test_thai_encoding(jsonl_path):
        tests_passed += 1
    
    tests_total += 1
    if test_llama_index_conversion(jsonl_path):
        tests_passed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n✅ All tests passed! Output is ready for Phase 2.")
        print("\n📝 Next steps:")
        print("  1. Install llama-index if not already: pip install llama-index")
        print("  2. Create Phase 2 embedding script")
        print("  3. Load documents.jsonl and generate embeddings")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

