"""Test script for Production RAG features in iLand batch embedding pipeline."""

from pathlib import Path
import sys

# Add docs_embedding directory to path so `batch_embedding` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src-iLand' / 'docs_embedding'))

from batch_embedding import (
    iLandBatchEmbeddingPipeline,
    create_iland_production_query_engine,
    CONFIG
)

def test_path_configuration():
    """Test that paths are correctly configured."""
    print("🔧 TESTING PATH CONFIGURATION:")
    print("-" * 40)
    
    data_dir = CONFIG["data_dir"]
    output_dir = CONFIG["output_dir"]
    
    print(f"📁 Data directory: {data_dir}")
    print(f"✅ Data directory exists: {data_dir.exists()}")
    
    print(f"📁 Output directory parent: {output_dir.parent}")
    print(f"✅ Output parent exists: {output_dir.parent.exists()}")
    
    if data_dir.exists():
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"📂 Found {len(subdirs)} data subdirectories: {[d.name for d in subdirs]}")
        
        # Count markdown files
        md_files = list(data_dir.glob("**/*.md"))
        print(f"📄 Found {len(md_files)} markdown files")
    
    return data_dir.exists()

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print(f"\n🚀 TESTING PIPELINE INITIALIZATION:")
    print("-" * 40)
    
    try:
        pipeline = iLandBatchEmbeddingPipeline()
        print("✅ Pipeline initialized successfully")
        print(f"🔧 Configuration loaded:")
        print(f"   • Sentence Window: {pipeline.config['enable_sentence_window']}")
        print(f"   • Hierarchical Retrieval: {pipeline.config['enable_hierarchical_retrieval']}")
        print(f"   • Query Router: {pipeline.config['enable_query_router']}")
        print(f"   • Batch Size: {pipeline.config['batch_size']}")
        return True
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_file_discovery():
    """Test file discovery functionality."""
    print(f"\n📂 TESTING FILE DISCOVERY:")
    print("-" * 40)
    
    try:
        pipeline = iLandBatchEmbeddingPipeline()
        batches = pipeline.get_markdown_files_in_batches()
        
        print(f"✅ File discovery successful")
        print(f"📦 Created {len(batches)} batches")
        
        total_files = sum(len(batch) for batch in batches)
        print(f"📄 Total files: {total_files}")
        
        if batches:
            first_batch = batches[0]
            print(f"📋 First batch contains {len(first_batch)} files:")
            for i, file_path in enumerate(first_batch[:3], 1):
                print(f"   {i}. {file_path.name}")
            if len(first_batch) > 3:
                print(f"   ... and {len(first_batch) - 3} more files")
        
        return len(batches) > 0
    except Exception as e:
        print(f"❌ File discovery failed: {e}")
        return False

def demonstrate_production_features():
    """Demonstrate what production features are available."""
    print(f"\n🎯 PRODUCTION RAG FEATURES AVAILABLE:")
    print("=" * 50)
    
    features = {
        "✅ Sentence Window Index": "Fine-grained retrieval with context windows",
        "✅ Hierarchical Retriever": "Thai metadata filtering and structured search", 
        "✅ Query Router": "Automatic routing between factual, summary, and comparison queries",
        "✅ Document Summary Index": "Hierarchical document-to-chunk retrieval",
        "✅ Production Query Engine": "Unified interface with all optimizations"
    }
    
    for feature, description in features.items():
        print(f"{feature}")
        print(f"   → {description}")
        print()
    
    print("🚀 READY FOR PRODUCTION USE!")
    print("-" * 30)
    print("To run the full pipeline:")
    print("   python batch_embedding.py")
    print()
    print("To create a query engine programmatically:")
    print("   from batch_embedding import create_iland_production_query_engine")
    print("   query_engine = create_iland_production_query_engine()")
    print("   response = query_engine.query('Your question here')")

def main():
    """Run all tests."""
    print("🧪 iLAND PRODUCTION RAG - SYSTEM TESTS")
    print("=" * 60)
    
    # Test 1: Path configuration
    paths_ok = test_path_configuration()
    
    # Test 2: Pipeline initialization  
    init_ok = test_pipeline_initialization()
    
    # Test 3: File discovery
    files_ok = test_file_discovery()
    
    # Show features
    demonstrate_production_features()
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print("-" * 20)
    print(f"📁 Path Configuration: {'✅ PASS' if paths_ok else '❌ FAIL'}")
    print(f"🚀 Pipeline Init: {'✅ PASS' if init_ok else '❌ FAIL'}")
    print(f"📂 File Discovery: {'✅ PASS' if files_ok else '❌ FAIL'}")
    
    all_tests_pass = paths_ok and init_ok and files_ok
    
    if all_tests_pass:
        print(f"\n🎉 ALL TESTS PASSED! Ready for production embedding!")
    else:
        print(f"\n⚠️ Some tests failed. Check configuration.")
    
    return all_tests_pass

if __name__ == "__main__":
    main() 