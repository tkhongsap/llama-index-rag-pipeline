#!/usr/bin/env python3
"""
Example usage of the multi-model embedding system for iLand documents.
Demonstrates how to switch between BGE-M3 and OpenAI providers.
"""

import os
from pathlib import Path
from batch_embedding import iLandBatchEmbeddingPipeline, CONFIG

def example_bge_m3_usage():
    """Example: Using BGE-M3 as primary provider with OpenAI fallback."""
    print("\n🎯 EXAMPLE 1: BGE-M3 with OpenAI Fallback")
    print("=" * 60)
    
    # Configuration for BGE-M3 as primary
    config = CONFIG.copy()
    config.update({
        "embedding_provider": "BGE_M3",
        "embedding_config": {
            "default_provider": "BGE_M3",
            "providers": {
                "BGE_M3": {
                    "model_name": "BAAI/bge-m3",
                    "device": "auto",  # Will use CUDA if available, otherwise CPU
                    "batch_size": 32,
                    "normalize": True,
                    "trust_remote_code": True,
                    "max_length": 8192
                },
                "OPENAI": {
                    "model_name": "text-embedding-3-small",
                    "api_key_env": "OPENAI_API_KEY",
                    "batch_size": 20,
                    "retry_attempts": 3,
                    "timeout": 30
                }
            },
            "fallback_enabled": True,
            "fallback_order": ["BGE_M3", "OPENAI"]
        }
    })
    
    # Create pipeline
    try:
        pipeline = iLandBatchEmbeddingPipeline(config)
        print("✅ Pipeline initialized successfully with BGE-M3")
        
        # Get active provider info
        if hasattr(pipeline.embedding_processor, 'get_active_provider_info'):
            provider_info = pipeline.embedding_processor.get_active_provider_info()
            print(f"Active provider: {provider_info}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure BGE-M3 dependencies are installed: pip install transformers sentence-transformers torch FlagEmbedding")

def example_openai_only_usage():
    """Example: Using OpenAI only (no fallback)."""
    print("\n🎯 EXAMPLE 2: OpenAI Only (Legacy Mode)")
    print("=" * 60)
    
    # Configuration for OpenAI only
    config = CONFIG.copy()
    config.update({
        "embedding_provider": "OPENAI",
        "embedding_config": {
            "default_provider": "OPENAI",
            "providers": {
                "OPENAI": {
                    "model_name": "text-embedding-3-small",
                    "api_key_env": "OPENAI_API_KEY",
                    "batch_size": 20,
                    "retry_attempts": 3,
                    "timeout": 30
                }
            },
            "fallback_enabled": False,
            "fallback_order": ["OPENAI"]
        }
    })
    
    # Create pipeline
    try:
        pipeline = iLandBatchEmbeddingPipeline(config)
        print("✅ Pipeline initialized successfully with OpenAI")
        
        # Get active provider info
        if hasattr(pipeline.embedding_processor, 'get_active_provider_info'):
            provider_info = pipeline.embedding_processor.get_active_provider_info()
            print(f"Active provider: {provider_info}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure OPENAI_API_KEY is set in your environment")

def example_auto_detection():
    """Example: Auto-detect optimal provider based on available resources."""
    print("\n🎯 EXAMPLE 3: Auto-Detection")
    print("=" * 60)
    
    from embedding_config import get_embedding_config
    
    try:
        # Auto-detect configuration
        embedding_config = get_embedding_config()
        optimal_provider = embedding_config.auto_detect_optimal_provider()
        
        print(f"🎯 Auto-detected optimal provider: {optimal_provider}")
        
        # Use the auto-detected configuration
        config = CONFIG.copy()
        config.update({
            "embedding_provider": optimal_provider,
            "embedding_config": embedding_config.get_full_config()
        })
        
        pipeline = iLandBatchEmbeddingPipeline(config)
        print("✅ Pipeline initialized with auto-detected configuration")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_environment_configuration():
    """Example: Using environment variables for configuration."""
    print("\n🎯 EXAMPLE 4: Environment Variable Configuration")
    print("=" * 60)
    
    # Set environment variables (normally done in shell/docker)
    os.environ["EMBEDDING_PROVIDER"] = "BGE_M3"
    os.environ["BGE_M3_DEVICE"] = "cpu"  # Force CPU usage
    os.environ["BGE_M3_BATCH_SIZE"] = "16"
    os.environ["EMBEDDING_FALLBACK_ENABLED"] = "true"
    
    # Create pipeline with environment configuration
    try:
        pipeline = iLandBatchEmbeddingPipeline(CONFIG)
        print("✅ Pipeline initialized with environment configuration")
        
        # Show configuration
        if hasattr(pipeline.embedding_processor, 'get_active_provider_info'):
            provider_info = pipeline.embedding_processor.get_active_provider_info()
            print(f"Active provider: {provider_info}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_legacy_compatibility():
    """Example: Legacy configuration format (backward compatibility)."""
    print("\n🎯 EXAMPLE 5: Legacy Configuration Compatibility")
    print("=" * 60)
    
    # Old-style configuration (should still work)
    legacy_config = {
        "data_dir": Path("../example"),
        "output_dir": Path("../data/embedding"),
        "embedding_model": "text-embedding-3-small",  # Old format
        "batch_size": 20,
        "chunk_size": 1024,
        # ... other legacy settings
    }
    
    try:
        pipeline = iLandBatchEmbeddingPipeline(legacy_config)
        print("✅ Pipeline initialized with legacy configuration")
        print("ℹ️ Legacy configuration automatically converted to new format")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_performance_comparison():
    """Example: Compare performance between providers."""
    print("\n🎯 EXAMPLE 6: Performance Comparison")
    print("=" * 60)
    
    # Sample texts for testing
    sample_texts = [
        "ที่ดินประเภทชาโนด เนื้อที่ 100 ตารางวา ตั้งอยู่ในเขตกรุงเทพมหานคร",
        "ที่ดินเกษตรกรรม เนื้อที่ 5 ไร่ จังหวัดเชียงใหม่",
        "ที่ดินพาณิชกรรม เนื้อที่ 50 ตารางวา ใกล้ห้างสรรพสินค้า"
    ]
    
    # Test BGE-M3
    try:
        from multi_model_embedding_processor import MultiModelEmbeddingProcessor
        from embedding_config import get_embedding_config
        
        # BGE-M3 configuration
        bge_config = get_embedding_config("BGE_M3")
        bge_processor = MultiModelEmbeddingProcessor(bge_config.get_full_config())
        
        print("🔄 Testing BGE-M3 performance...")
        import time
        start_time = time.time()
        
        bge_embeddings = bge_processor.embedding_manager.embed_documents_with_fallback(sample_texts)
        bge_duration = time.time() - start_time
        
        print(f"✅ BGE-M3: {len(bge_embeddings)} embeddings in {bge_duration:.2f}s")
        print(f"   Embedding dimension: {len(bge_embeddings[0]) if bge_embeddings else 0}")
        
        # Get metrics
        metrics = bge_processor.get_provider_metrics()
        print(f"   Metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ BGE-M3 test failed: {str(e)}")
    
    # Test OpenAI (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_config = get_embedding_config("OPENAI")
            openai_processor = MultiModelEmbeddingProcessor(openai_config.get_full_config())
            
            print("🔄 Testing OpenAI performance...")
            start_time = time.time()
            
            openai_embeddings = openai_processor.embedding_manager.embed_documents_with_fallback(sample_texts)
            openai_duration = time.time() - start_time
            
            print(f"✅ OpenAI: {len(openai_embeddings)} embeddings in {openai_duration:.2f}s")
            print(f"   Embedding dimension: {len(openai_embeddings[0]) if openai_embeddings else 0}")
            
        except Exception as e:
            print(f"❌ OpenAI test failed: {str(e)}")
    else:
        print("⚠️ Skipping OpenAI test (no API key)")

def main():
    """Run all examples."""
    print("🚀 MULTI-MODEL EMBEDDING SYSTEM EXAMPLES")
    print("=" * 80)
    print("This script demonstrates various ways to use the new multi-model embedding system.")
    print("Make sure you have the required dependencies installed:")
    print("  pip install transformers sentence-transformers torch FlagEmbedding")
    print("")
    
    # Run examples
    example_bge_m3_usage()
    example_openai_only_usage()
    example_auto_detection()
    example_environment_configuration()
    example_legacy_compatibility()
    example_performance_comparison()
    
    print("\n✨ SUMMARY")
    print("=" * 40)
    print("✅ BGE-M3: Local processing, no API costs, multilingual support")
    print("✅ OpenAI: Cloud processing, high quality, requires API key")
    print("✅ Fallback: Automatic switching if primary provider fails")
    print("✅ Backward compatibility: Legacy configurations still work")
    print("✅ Environment configuration: Easy deployment configuration")

if __name__ == "__main__":
    main() 