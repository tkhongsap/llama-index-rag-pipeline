"""
example_usage.py - Examples of using the iLand embedding loading module

This file demonstrates various ways to use the modular iLand embedding loading system.
Updated to use the new modular structure following coding rules.
"""

# Import from the new modular structure
from .models import EmbeddingConfig, FilterConfig
from .embedding_loader import iLandEmbeddingLoader
from .index_reconstructor import iLandIndexReconstructor
from .validation import validate_iland_embeddings, generate_validation_report
from .utils import (
    load_latest_iland_embeddings,
    load_all_latest_iland_embeddings,
    create_iland_index_from_latest_batch,
    get_iland_batch_summary
)

def example_basic_loading():
    """Example 1: Basic loading of iLand embeddings."""
    print("📚 Example 1: Basic Loading")
    print("-" * 40)
    
    # Load chunk embeddings from latest batch
    embeddings, batch_path = load_latest_iland_embeddings("chunks")
    print(f"✅ Loaded {len(embeddings)} chunk embeddings from {batch_path.name}")
    
    # Validate the embeddings
    stats = validate_iland_embeddings(embeddings)
    print(f"📊 Found {len(stats['thai_metadata']['provinces'])} provinces")
    print(f"📋 Found {len(stats['thai_metadata']['deed_types'])} deed types")

def example_multi_batch_loading():
    """Example 2: Loading from multiple sub-batches."""
    print("\n📚 Example 2: Multi-batch Loading")
    print("-" * 40)
    
    # Load all chunk embeddings from all sub-batches
    all_embeddings, batch_path = load_all_latest_iland_embeddings("chunks")
    print(f"✅ Loaded {len(all_embeddings)} total chunk embeddings from all sub-batches")
    
    # Get batch summary
    summary = get_iland_batch_summary()
    print(f"📊 Batch summary: {summary['latest_batch_stats']['total_embeddings']} total embeddings")

def example_filtering():
    """Example 3: Filtering embeddings by various criteria."""
    print("\n📚 Example 3: Filtering Embeddings")
    print("-" * 40)
    
    # Initialize components
    config = EmbeddingConfig()
    loader = iLandEmbeddingLoader(config)
    
    # Load embeddings
    embeddings, _ = load_all_latest_iland_embeddings("chunks")
    print(f"📄 Starting with {len(embeddings)} embeddings")
    
    # Filter by province
    bangkok_embeddings = loader.filter_embeddings_by_province(embeddings, "กรุงเทพมหานคร")
    print(f"🌏 Bangkok embeddings: {len(bangkok_embeddings)}")
    
    # Filter by deed type
    chanote_embeddings = loader.filter_embeddings_by_deed_type(embeddings, "chanote")
    print(f"📋 Chanote deed embeddings: {len(chanote_embeddings)}")
    
    # Filter by area range
    small_plots = loader.filter_embeddings_by_area_range(embeddings, max_area_rai=5.0)
    print(f"📏 Small plots (≤5 rai): {len(small_plots)}")

def example_index_creation():
    """Example 4: Creating indices with different configurations."""
    print("\n📚 Example 4: Index Creation")
    print("-" * 40)
    
    config = EmbeddingConfig()
    
    if not config.api_key:
        print("⚠️ Skipping index creation - no API key available")
        return
    
    try:
        # Create basic index
        index = create_iland_index_from_latest_batch(
            use_chunks=True,
            max_embeddings=50  # Limit for example
        )
        print("✅ Created basic chunk index")
        
        # Create filtered index
        filtered_index = create_iland_index_from_latest_batch(
            use_chunks=True,
            province_filter="กรุงเทพมหานคร",
            max_embeddings=20
        )
        print("✅ Created Bangkok-filtered index")
        
        # Test query
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("ที่ดินประเภทใดมีมากที่สุด?")
        print(f"🔍 Query response: {str(response)[:100]}...")
        
    except Exception as e:
        print(f"❌ Index creation failed: {e}")

def example_advanced_filtering():
    """Example 5: Advanced filtering with FilterConfig."""
    print("\n📚 Example 5: Advanced Filtering")
    print("-" * 40)
    
    # Create filter configuration
    filter_config = FilterConfig(
        provinces=["กรุงเทพมหานคร", "เชียงใหม่"],
        deed_types=["chanote", "nor_sor_3"],
        max_embeddings=100
    )
    
    # Initialize loader
    config = EmbeddingConfig()
    loader = iLandEmbeddingLoader(config)
    
    # Load and filter embeddings
    embeddings, _ = load_all_latest_iland_embeddings("chunks")
    filtered_embeddings = loader.apply_filter_config(embeddings, filter_config)
    
    print(f"📊 Filtered from {len(embeddings)} to {len(filtered_embeddings)} embeddings")
    print(f"🔧 Filter configuration: {filter_config.to_dict()}")

def example_validation_report():
    """Example 6: Generate comprehensive validation report."""
    print("\n📚 Example 6: Validation Report")
    print("-" * 40)
    
    # Load embeddings
    embeddings, _ = load_all_latest_iland_embeddings("chunks")
    
    # Generate comprehensive report
    report = generate_validation_report(embeddings)
    print(report)

def example_class_based_usage():
    """Example 7: Using classes directly for more control."""
    print("\n📚 Example 7: Class-based Usage")
    print("-" * 40)
    
    # Initialize configuration
    config = EmbeddingConfig()
    
    # Initialize loader
    loader = iLandEmbeddingLoader(config)
    
    # Get available batches
    batches = loader.get_available_iland_batches()
    print(f"📁 Found {len(batches)} available batches")
    
    # Load specific embedding type
    result = loader.load_specific_embedding_type("chunks", "batch_1")
    print(f"📦 Loaded {result.count} embeddings from {result.metadata['sub_batch']}")
    
    # Initialize reconstructor (if API key available)
    if config.api_key:
        reconstructor = iLandIndexReconstructor(config)
        
        # Create index with limited embeddings
        limited_embeddings = result.embeddings[:10]
        index = reconstructor.create_vector_index_from_embeddings(limited_embeddings)
        print(f"🏗️ Created index with {len(limited_embeddings)} nodes")

# ---------- MAIN EXECUTION --------------------------------------------------

if __name__ == "__main__":
    print("🚀 iLAND EMBEDDING LOADING EXAMPLES")
    print("=" * 80)
    
    try:
        example_basic_loading()
        example_multi_batch_loading()
        example_filtering()
        example_index_creation()
        example_advanced_filtering()
        example_validation_report()
        example_class_based_usage()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc() 