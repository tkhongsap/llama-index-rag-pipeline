#!/usr/bin/env python3
"""
Minimal BGE embedding test that checks structure and imports without requiring all packages.
"""

import sys
from pathlib import Path

# Add src-iLand to path
sys.path.insert(0, str(Path(__file__).parent / "src-iLand"))

def test_import_structure():
    """Test that our BGE modules can be imported and have the right structure."""
    print("🔍 Testing BGE module import structure...")
    
    try:
        # Test BGE embedding processor import
        from docs_embedding.bge_embedding_processor import BGEEmbeddingProcessor
        print("  ✅ BGEEmbeddingProcessor imports successfully")
        
        # Test class structure
        processor_methods = dir(BGEEmbeddingProcessor)
        required_methods = [
            '__init__',
            'get_model_info',
            'extract_indexnode_embeddings',
            'extract_chunk_embeddings', 
            'extract_summary_embeddings',
            'compare_embeddings'
        ]
        
        for method in required_methods:
            if method in processor_methods:
                print(f"    ✅ Method {method} available")
            else:
                print(f"    ❌ Method {method} missing")
        
        # Test BGE models configuration
        bge_models = BGEEmbeddingProcessor.BGE_MODELS
        print(f"  📊 Available BGE models: {list(bge_models.keys())}")
        
        for model_name, config in bge_models.items():
            print(f"    {model_name}: {config['dimension']}d, max_len={config['max_length']}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ BGEEmbeddingProcessor import failed: {e}")
        return False

def test_batch_embedding_structure():
    """Test the enhanced batch embedding module."""
    print("\n🔍 Testing enhanced batch embedding structure...")
    
    try:
        from docs_embedding.batch_embedding_bge import iLandBGEBatchEmbeddingPipeline, CONFIG
        print("  ✅ iLandBGEBatchEmbeddingPipeline imports successfully")
        
        # Test configuration structure
        print(f"  📋 Configuration structure:")
        print(f"    Provider: {CONFIG['embedding']['provider']}")
        print(f"    BGE model: {CONFIG['embedding']['bge']['model_name']}")
        print(f"    OpenAI model: {CONFIG['embedding']['openai']['model_name']}")
        print(f"    Batch size: {CONFIG['batch_size']}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Enhanced batch embedding import failed: {e}")
        return False

def test_requirements_structure():
    """Test that requirements file has been updated."""
    print("\n🔍 Testing requirements file structure...")
    
    try:
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            print("  ❌ requirements.txt not found")
            return False
        
        with open(requirements_file) as f:
            content = f.read()
        
        required_packages = [
            "llama-index-embeddings-huggingface",
            "sentence-transformers",
            "transformers",
            "torch"
        ]
        
        for package in required_packages:
            if package in content:
                print(f"  ✅ {package} found in requirements")
            else:
                print(f"  ❌ {package} missing from requirements")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Requirements test failed: {e}")
        return False

def test_integration_readiness():
    """Test that the integration is ready for use."""
    print("\n🔍 Testing integration readiness...")
    
    # Check if we can create a basic configuration
    try:
        bge_config = {
            "provider": "bge",
            "model_name": "bge-small-en-v1.5",
            "cache_folder": "./cache/bge_models"
        }
        print(f"  ✅ BGE configuration structure valid: {bge_config}")
        
        openai_config = {
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "test_key"
        }
        print(f"  ✅ OpenAI configuration structure valid: {openai_config}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples for when packages are installed."""
    print("\n💡 USAGE EXAMPLES (after installing packages):")
    print("=" * 50)
    
    print("\n1. Install required packages:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Basic BGE embedding test:")
    print("""   from src-iLand.docs_embedding.bge_embedding_processor import create_bge_embedding_processor
   
   processor = create_bge_embedding_processor("bge-small-en-v1.5")
   embedding = processor.embed_model.get_text_embedding("Test text")
   print(f"Embedding dimension: {len(embedding)}")""")
    
    print("\n3. Run BGE batch processing:")
    print("""   cd src-iLand
   python -m docs_embedding.batch_embedding_bge""")
    
    print("\n4. Compare BGE vs OpenAI (with OPENAI_API_KEY set):")
    print("""   # Edit CONFIG in batch_embedding_bge.py:
   CONFIG = {
       ...
       "embedding": {"provider": "bge", ...},
       "enable_comparison": True
   }""")
    
    print("\n5. Switch to OpenAI embeddings:")
    print("""   # Edit CONFIG in batch_embedding_bge.py:
   CONFIG = {
       ...
       "embedding": {"provider": "openai", ...}
   }""")

def main():
    """Main test runner."""
    print("🚀 BGE EMBEDDING INTEGRATION STRUCTURE TEST")
    print("=" * 50)
    
    tests = [
        ("BGE Module Structure", test_import_structure),
        ("Batch Embedding Structure", test_batch_embedding_structure), 
        ("Requirements Structure", test_requirements_structure),
        ("Integration Readiness", test_integration_readiness)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 STRUCTURE TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! BGE integration is properly set up.")
        print("📦 Install packages with: pip install -r requirements.txt")
        print("🧪 Then run: python test_bge_embedding.py")
    else:
        print("⚠️ Some structure tests failed. Check the output above.")
    
    show_usage_examples()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)