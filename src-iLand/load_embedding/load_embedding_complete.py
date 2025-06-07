"""
load_embedding_complete.py - Complete embedding loading with validation, filtering, and indexing

This module provides comprehensive functionality for loading iLand embeddings with:
- Full validation and quality analysis
- Advanced filtering by province, deed type, and area
- Index creation and testing capabilities
- Production-ready embedding loading pipeline
"""

import os
from .models import EmbeddingConfig
from .embedding_loader import iLandEmbeddingLoader
from .index_reconstructor import iLandIndexReconstructor
from .validation import validate_iland_embeddings

# ---------- MAIN DEMONSTRATION FUNCTION -------------------------------------

def demonstrate_iland_loading(config: EmbeddingConfig = None):
    """Demonstrate loading iLand embeddings and creating indices."""
    print("🔄 iLAND EMBEDDING LOADER DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize components
        if config is None:
            config = EmbeddingConfig()
        
        loader = iLandEmbeddingLoader(config)
        
        # Show available iLand batches
        batches = loader.get_available_iland_batches()
        print(f"\n📁 Found {len(batches)} iLand embedding batches:")
        for batch in batches[-5:]:  # Show last 5
            print(f"   • {batch.name}")
        
        # Use latest batch
        latest_batch = loader.get_latest_iland_batch()
        if not latest_batch:
            print("❌ No iLand embedding batches found!")
            return
        
        print(f"\n📊 Using latest iLand batch: {latest_batch.name}")
        
        # Load batch statistics
        stats = loader.load_batch_statistics(latest_batch)
        if stats:
            print(f"   • Dataset type: {stats.get('dataset_type', 'N/A')}")
            print(f"   • Total batches: {stats.get('total_batches', 'N/A')}")
            print(f"   • Total embeddings: {stats['grand_totals']['total_embeddings']}")
            print(f"   • Chunks: {stats['grand_totals']['chunk_embeddings']}")
            print(f"   • IndexNodes: {stats['grand_totals']['indexnode_embeddings']}")
            print(f"   • Summaries: {stats['grand_totals']['summary_embeddings']}")
            print(f"   • Unique metadata fields: {stats['metadata_analysis']['total_unique_metadata_fields']}")
        
        # Load embeddings from ALL sub-batches
        print("\n🔄 Loading iLand embeddings from all sub-batches...")
        all_batch_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        # Count total embeddings across all sub-batches
        total_chunks = sum(len(emb_types.get("chunks", [])) for emb_types in all_batch_embeddings.values())
        total_summaries = sum(len(emb_types.get("summaries", [])) for emb_types in all_batch_embeddings.values())
        total_indexnodes = sum(len(emb_types.get("indexnodes", [])) for emb_types in all_batch_embeddings.values())
        
        print(f"   • Found {len(all_batch_embeddings)} sub-batches")
        print(f"   • Total chunks: {total_chunks}")
        print(f"   • Total summaries: {total_summaries}")
        print(f"   • Total indexnodes: {total_indexnodes}")
        
        # Get chunk embeddings from first available sub-batch for validation
        chunk_embeddings = []
        for sub_batch, emb_types in all_batch_embeddings.items():
            if emb_types.get("chunks"):
                chunk_embeddings = emb_types["chunks"]
                print(f"   • Using chunks from {sub_batch} for demo ({len(chunk_embeddings)} embeddings)")
                break
        
        # Validate iLand embeddings
        print("\n📋 Validating loaded iLand embeddings...")
        validation_stats = validate_iland_embeddings(chunk_embeddings)
        print(f"   • Total loaded: {validation_stats['total_count']}")
        print(f"   • Has text: {validation_stats['has_text']}")
        print(f"   • Has vectors: {validation_stats['has_vectors']}")
        print(f"   • Embedding dimensions: {validation_stats['embedding_dims']}")
        print(f"   • Average text length: {validation_stats['avg_text_length']:.0f} chars")
        
        # Thai-specific metadata
        thai_meta = validation_stats['thai_metadata']
        print(f"   🌏 Provinces found: {len(thai_meta['provinces'])} ({', '.join(list(thai_meta['provinces'])[:3])}...)")
        print(f"   📋 Deed types: {len(thai_meta['deed_types'])} types")
        print(f"   🏞️ Land categories: {len(thai_meta['land_categories'])} categories")
        print(f"   🏠 Ownership types: {len(thai_meta['ownership_types'])} types")
        print(f"   📏 Deeds with area info: {thai_meta['deed_with_area']}")
        print(f"   📍 Deeds with location: {thai_meta['deed_with_location']}")
        
        if validation_stats['issues']:
            print(f"   ⚠️ Found {len(validation_stats['issues'])} issues")
        
        # Create index (only if API key is available)
        if config.api_key and chunk_embeddings:
            try:
                print("\n🔄 Creating VectorStoreIndex from iLand embeddings...")
                reconstructor = iLandIndexReconstructor(config)
                index = reconstructor.create_vector_index_from_embeddings(
                    chunk_embeddings[:10]  # Use first 10 for demo
                )
                
                # Test the index with Thai land deed query
                print("\n🧪 Testing index with Thai land deed query...")
                query_engine = index.as_query_engine(similarity_top_k=3)
                response = query_engine.query("ที่ดินในจังหวัดใดมีพื้นที่มากที่สุด?")  # Which province has the largest land area?
                print(f"📝 Thai query response preview: {str(response)[:200]}...")
                
                # Test filtering by province
                if thai_meta['provinces']:
                    test_province = list(thai_meta['provinces'])[0]
                    print(f"\n🌏 Testing province filter for: {test_province}")
                    filtered_embeddings = loader.filter_embeddings_by_province(chunk_embeddings, test_province)
                    print(f"   • Filtered to {len(filtered_embeddings)} embeddings for {test_province}")
                    
                    if filtered_embeddings:
                        province_index = reconstructor.create_province_specific_index(
                            chunk_embeddings, test_province, show_progress=False
                        )
                        print(f"   ✅ Created province-specific index for {test_province}")
                        
            except Exception as index_error:
                print(f"⚠️ Could not create index: {index_error}")
        else:
            print("\n⚠️ Skipping index creation - no API key or embeddings available")
            print("   Set OPENAI_API_KEY environment variable to test index functionality")
        
        print("\n✅ iLand embedding loading and index reconstruction successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def demonstrate_filtering_capabilities(config: EmbeddingConfig = None):
    """Demonstrate filtering capabilities of iLand embeddings."""
    print("\n🔍 iLAND FILTERING CAPABILITIES DEMONSTRATION")
    print("=" * 80)
    
    try:
        if config is None:
            config = EmbeddingConfig()
        
        loader = iLandEmbeddingLoader(config)
        
        # Load all chunk embeddings
        result = loader.load_all_embeddings_of_type("chunks")
        embeddings = result.embeddings
        
        print(f"\n📊 Loaded {len(embeddings)} total chunk embeddings")
        
        # Validate to get metadata distribution
        validation_stats = validate_iland_embeddings(embeddings)
        thai_meta = validation_stats['thai_metadata']
        
        # Demonstrate province filtering
        if thai_meta['provinces']:
            test_provinces = list(thai_meta['provinces'])[:3]  # Test with first 3 provinces
            print(f"\n🌏 Testing province filtering with: {test_provinces}")
            
            for province in test_provinces:
                filtered = loader.filter_embeddings_by_province(embeddings, province)
                print(f"   • {province}: {len(filtered)} embeddings")
        
        # Demonstrate deed type filtering
        if thai_meta['deed_types']:
            test_deed_types = list(thai_meta['deed_types'])[:3]  # Test with first 3 deed types
            print(f"\n📋 Testing deed type filtering with: {test_deed_types}")
            
            for deed_type in test_deed_types:
                filtered = loader.filter_embeddings_by_deed_type(embeddings, deed_type)
                print(f"   • {deed_type}: {len(filtered)} embeddings")
        
        # Demonstrate area filtering
        print(f"\n📏 Testing area filtering...")
        area_filtered_small = loader.filter_embeddings_by_area_range(embeddings, max_area_rai=5.0)
        area_filtered_large = loader.filter_embeddings_by_area_range(embeddings, min_area_rai=10.0)
        print(f"   • Small plots (≤5 rai): {len(area_filtered_small)} embeddings")
        print(f"   • Large plots (≥10 rai): {len(area_filtered_large)} embeddings")
        
        print("\n✅ Filtering demonstration completed!")
        
    except Exception as e:
        print(f"\n❌ Error in filtering demonstration: {str(e)}")

def demonstrate_index_creation_options(config: EmbeddingConfig = None):
    """Demonstrate different index creation options."""
    print("\n🏗️ iLAND INDEX CREATION OPTIONS DEMONSTRATION")
    print("=" * 80)
    
    try:
        if config is None:
            config = EmbeddingConfig()
        
        if not config.api_key:
            print("⚠️ Skipping index creation demo - no API key available")
            return
        
        loader = iLandEmbeddingLoader(config)
        reconstructor = iLandIndexReconstructor(config)
        
        # Load different types of embeddings
        chunks_result = loader.load_all_embeddings_of_type("chunks")
        summaries_result = loader.load_all_embeddings_of_type("summaries")
        indexnodes_result = loader.load_all_embeddings_of_type("indexnodes")
        
        chunks = chunks_result.embeddings[:20]  # Limit for demo
        summaries = summaries_result.embeddings[:10]
        indexnodes = indexnodes_result.embeddings[:10]
        
        print(f"\n📊 Using limited embeddings for demo:")
        print(f"   • Chunks: {len(chunks)}")
        print(f"   • Summaries: {len(summaries)}")
        print(f"   • IndexNodes: {len(indexnodes)}")
        
        # Create different types of indices
        print(f"\n🔄 Creating different index types...")
        
        # 1. Chunks-only index
        chunks_index = reconstructor.create_vector_index_from_embeddings(chunks, show_progress=False)
        print(f"   ✅ Chunks-only index: {len(chunks)} nodes")
        
        # 2. Combined index
        if summaries and indexnodes:
            combined_index = reconstructor.create_combined_iland_index(
                chunks, summaries, indexnodes, show_progress=False
            )
            print(f"   ✅ Combined index: {len(chunks) + len(summaries) + len(indexnodes)} nodes")
        
        # 3. Province-specific index
        validation_stats = validate_iland_embeddings(chunks)
        if validation_stats['thai_metadata']['provinces']:
            test_province = list(validation_stats['thai_metadata']['provinces'])[0]
            province_index = reconstructor.create_province_specific_index(
                chunks, test_province, show_progress=False
            )
            print(f"   ✅ Province-specific index for {test_province}")
        
        print("\n✅ Index creation options demonstration completed!")
        
    except Exception as e:
        print(f"\n❌ Error in index creation demonstration: {str(e)}")

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    # Run demonstration
    config = EmbeddingConfig()
    demonstrate_iland_loading(config) 