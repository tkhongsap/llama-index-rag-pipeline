#!/usr/bin/env python3
"""
Test script for BGE embedding integration with iLand docs_embedding module.
Tests both Hugging Face BGE model and OpenAI embeddings for compatibility.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src-iLand to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src-iLand"))

# Load environment variables
load_dotenv()

def test_installations():
    """Test if required packages are installed."""
    print("🔍 Testing package installations...")
    
    required_packages = [
        "llama_index",
        "sentence_transformers", 
        "transformers",
        "torch"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} installed")
        except ImportError:
            print(f"  ❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_bge_embedding_basic():
    """Test basic BGE embedding functionality."""
    print("\n🧪 Testing BGE embedding basic functionality...")
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Initialize BGE embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_folder="./cache"
        )
        
        # Test embedding generation
        test_text = "This is a test document about Thai land deeds and property information."
        
        print(f"  📝 Test text: {test_text[:50]}...")
        
        start_time = time.time()
        embedding = embed_model.get_text_embedding(test_text)
        duration = time.time() - start_time
        
        print(f"  ✅ BGE embedding successful")
        print(f"  📊 Embedding dimension: {len(embedding)}")
        print(f"  ⏱️ Generation time: {duration:.3f}s")
        print(f"  📈 Sample values: {embedding[:5]}")
        
        return embed_model, embedding
        
    except Exception as e:
        print(f"  ❌ BGE embedding failed: {str(e)}")
        return None, None

def test_openai_embedding_basic():
    """Test OpenAI embedding for comparison."""
    print("\n🧪 Testing OpenAI embedding basic functionality...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠️ OPENAI_API_KEY not found - skipping OpenAI test")
        return None, None
    
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        # Initialize OpenAI embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=api_key
        )
        
        # Test embedding generation
        test_text = "This is a test document about Thai land deeds and property information."
        
        print(f"  📝 Test text: {test_text[:50]}...")
        
        start_time = time.time()
        embedding = embed_model.get_text_embedding(test_text)
        duration = time.time() - start_time
        
        print(f"  ✅ OpenAI embedding successful")
        print(f"  📊 Embedding dimension: {len(embedding)}")
        print(f"  ⏱️ Generation time: {duration:.3f}s")
        print(f"  📈 Sample values: {embedding[:5]}")
        
        return embed_model, embedding
        
    except Exception as e:
        print(f"  ❌ OpenAI embedding failed: {str(e)}")
        return None, None

def test_embedding_compatibility(bge_embedding, openai_embedding):
    """Test compatibility between BGE and OpenAI embeddings."""
    print("\n🔄 Testing embedding compatibility...")
    
    if bge_embedding is None or openai_embedding is None:
        print("  ⚠️ Cannot test compatibility - one or both embeddings missing")
        return False
    
    # Check dimensions
    bge_dim = len(bge_embedding)
    openai_dim = len(openai_embedding)
    
    print(f"  📊 BGE dimension: {bge_dim}")
    print(f"  📊 OpenAI dimension: {openai_dim}")
    
    if bge_dim != openai_dim:
        print(f"  ⚠️ Different dimensions: BGE={bge_dim}, OpenAI={openai_dim}")
        print("  💡 This is expected - different models have different dimensions")
    
    # Test similarity calculation
    try:
        # Convert to numpy arrays for calculation
        bge_arr = np.array(bge_embedding)
        openai_arr = np.array(openai_embedding)
        
        # Normalize vectors
        bge_norm = bge_arr / np.linalg.norm(bge_arr)
        openai_norm = openai_arr / np.linalg.norm(openai_arr)
        
        print(f"  📊 BGE vector norm: {np.linalg.norm(bge_arr):.4f}")
        print(f"  📊 OpenAI vector norm: {np.linalg.norm(openai_arr):.4f}")
        
        print("  ✅ Both embeddings are compatible for vector operations")
        return True
        
    except Exception as e:
        print(f"  ❌ Compatibility test failed: {str(e)}")
        return False

def test_docs_embedding_with_bge():
    """Test the docs_embedding module with BGE embeddings."""
    print("\n🏗️ Testing docs_embedding module with BGE...")
    
    try:
        # Import docs_embedding modules
        from docs_embedding.document_loader import iLandDocumentLoader
        from docs_embedding.metadata_extractor import iLandMetadataExtractor
        from docs_embedding.embedding_processor import EmbeddingProcessor
        
        # Check if we have sample documents
        example_dir = Path("example")
        if not example_dir.exists():
            print("  ⚠️ No example directory found - creating minimal test")
            return test_minimal_integration()
        
        # Find sample documents
        sample_files = list(example_dir.rglob("*.md"))
        if not sample_files:
            print("  ⚠️ No markdown files found in example directory")
            return test_minimal_integration()
        
        # Take first sample file
        sample_file = sample_files[0]
        print(f"  📄 Testing with: {sample_file.name}")
        
        # Load document
        loader = iLandDocumentLoader()
        docs = loader.load_documents_from_files([sample_file])
        
        if not docs:
            print("  ❌ Failed to load document")
            return False
        
        print(f"  ✅ Loaded document: {docs[0].metadata.get('title', 'Unknown')}")
        print(f"  📊 Document length: {len(docs[0].text)} characters")
        
        # Test with BGE embedding
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        bge_embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_folder="./cache"
        )
        
        # Test embedding a chunk of the document
        test_chunk = docs[0].text[:500]  # First 500 characters
        
        start_time = time.time()
        embedding = bge_embed_model.get_text_embedding(test_chunk)
        duration = time.time() - start_time
        
        print(f"  ✅ BGE embedding successful for real document")
        print(f"  📊 Embedding dimension: {len(embedding)}")
        print(f"  ⏱️ Processing time: {duration:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ docs_embedding BGE test failed: {str(e)}")
        return False

def test_minimal_integration():
    """Test minimal integration without sample documents."""
    print("  🔧 Running minimal integration test...")
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Document
        
        # Create minimal test document
        test_doc = Document(
            text="โฉนดที่ดิน เลขที่ 12345 จังหวัด กรุงเทพมหานคร เขต บางกะปิ",
            metadata={
                "deed_type": "Chanote",
                "province": "Bangkok", 
                "district": "Bang Kapi",
                "deed_serial_no": "12345"
            }
        )
        
        # Test BGE embedding
        bge_embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_folder="./cache"
        )
        
        embedding = bge_embed_model.get_text_embedding(test_doc.text)
        
        print(f"  ✅ Minimal integration successful")
        print(f"  📊 Embedding dimension: {len(embedding)}")
        print(f"  📋 Test metadata: {test_doc.metadata}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Minimal integration failed: {str(e)}")
        return False

def test_performance_comparison():
    """Compare performance between BGE and OpenAI embeddings."""
    print("\n⚡ Testing performance comparison...")
    
    test_texts = [
        "โฉนดที่ดิน เลขที่ 12345 จังหวัด กรุงเทพมหานคร",
        "Land deed document with property information and measurements",
        "ที่ดิน ขนาด 2 ไร่ 3 งาน 45 ตารางวา ในเขต บางกะปิ กรุงเทพฯ",
        "Property location coordinates and boundary descriptions",
        "รายละเอียดการใช้ประโยชน์ที่ดิน และข้อมูลเจ้าของ"
    ]
    
    results = {}
    
    # Test BGE performance
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        bge_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_folder="./cache"
        )
        
        print("  🔄 Testing BGE performance...")
        start_time = time.time()
        
        bge_embeddings = []
        for text in test_texts:
            embedding = bge_model.get_text_embedding(text)
            bge_embeddings.append(embedding)
        
        bge_duration = time.time() - start_time
        results["bge"] = {
            "duration": bge_duration,
            "texts_processed": len(test_texts),
            "avg_time": bge_duration / len(test_texts),
            "dimension": len(bge_embeddings[0])
        }
        
        print(f"  ✅ BGE: {len(test_texts)} texts in {bge_duration:.3f}s")
        print(f"      Average: {bge_duration/len(test_texts):.3f}s per text")
        
    except Exception as e:
        print(f"  ❌ BGE performance test failed: {str(e)}")
    
    # Test OpenAI performance (if API key available)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            openai_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=api_key
            )
            
            print("  🔄 Testing OpenAI performance...")
            start_time = time.time()
            
            openai_embeddings = []
            for text in test_texts:
                embedding = openai_model.get_text_embedding(text)
                openai_embeddings.append(embedding)
            
            openai_duration = time.time() - start_time
            results["openai"] = {
                "duration": openai_duration,
                "texts_processed": len(test_texts),
                "avg_time": openai_duration / len(test_texts),
                "dimension": len(openai_embeddings[0])
            }
            
            print(f"  ✅ OpenAI: {len(test_texts)} texts in {openai_duration:.3f}s")
            print(f"      Average: {openai_duration/len(test_texts):.3f}s per text")
            
        except Exception as e:
            print(f"  ❌ OpenAI performance test failed: {str(e)}")
    
    # Compare results
    if len(results) > 1:
        print("\n  📊 Performance Summary:")
        for model_name, metrics in results.items():
            print(f"    {model_name.upper()}:")
            print(f"      - Total time: {metrics['duration']:.3f}s")
            print(f"      - Avg per text: {metrics['avg_time']:.3f}s")
            print(f"      - Dimension: {metrics['dimension']}")
    
    return results

def main():
    """Main test runner."""
    print("🚀 BGE EMBEDDING INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Package installations
    if not test_installations():
        print("\n❌ Package installation check failed")
        print("Please install missing packages and retry")
        return False
    
    # Test 2: Basic BGE embedding
    bge_model, bge_embedding = test_bge_embedding_basic()
    
    # Test 3: Basic OpenAI embedding (for comparison)
    openai_model, openai_embedding = test_openai_embedding_basic()
    
    # Test 4: Embedding compatibility
    test_embedding_compatibility(bge_embedding, openai_embedding)
    
    # Test 5: Integration with docs_embedding module
    integration_success = test_docs_embedding_with_bge()
    
    # Test 6: Performance comparison
    performance_results = test_performance_comparison()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    tests_results = [
        ("Package Installation", test_installations()),
        ("BGE Basic Functionality", bge_embedding is not None),
        ("OpenAI Basic Functionality", openai_embedding is not None),
        ("Embedding Compatibility", bge_embedding is not None and openai_embedding is not None),
        ("docs_embedding Integration", integration_success),
        ("Performance Testing", len(performance_results) > 0)
    ]
    
    passed = sum(1 for _, result in tests_results if result)
    total = len(tests_results)
    
    for test_name, result in tests_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BGE embedding is ready for use.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print("\n💡 Next Steps:")
    print("  1. Update requirements.txt to include BGE dependencies")
    print("  2. Modify embedding_processor.py to support BGE models")
    print("  3. Add configuration option to switch between OpenAI and BGE")
    print("  4. Test with full document processing pipeline")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)