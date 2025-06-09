#!/usr/bin/env python3
"""
BGE Embedding Runner - Easy configuration selection and execution.
"""

import sys
import argparse
from pathlib import Path

# Add src-iLand to path
sys.path.insert(0, str(Path(__file__).parent / "src-iLand"))

def main():
    parser = argparse.ArgumentParser(description="Run BGE embedding processing with different configurations")
    parser.add_argument(
        "--config", 
        type=str, 
        default="bge_multilingual",
        help="Configuration to use (default: bge_multilingual for Thai documents)"
    )
    parser.add_argument(
        "--list-configs", 
        action="store_true", 
        help="List all available configurations"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show configuration without running"
    )
    
    args = parser.parse_args()
    
    try:
        from docs_embedding.embedding_config import get_config, list_configs, ALL_CONFIGS
        from docs_embedding.batch_embedding_bge import iLandBGEBatchEmbeddingPipeline
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the required packages:")
        print("  pip install -r requirements.txt")
        print("Or run: ./install_and_test_bge.sh")
        return 1
    
    # List configurations if requested
    if args.list_configs:
        list_configs()
        return 0
    
    # Get configuration
    try:
        config = get_config(args.config)
        print(f"📋 Using configuration: {args.config}")
        print(f"   Provider: {config['embedding']['provider']}")
        
        if config['embedding']['provider'] == 'bge':
            model_name = config['embedding']['bge']['model_name']
            print(f"   BGE Model: {model_name}")
        else:
            model_name = config['embedding']['openai']['model_name']
            print(f"   OpenAI Model: {model_name}")
        
        print(f"   Comparison: {'enabled' if config.get('enable_comparison', False) else 'disabled'}")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n📋 Available configurations:")
        for name in ALL_CONFIGS.keys():
            print(f"  - {name}")
        return 1
    
    # Dry run - show config and exit
    if args.dry_run:
        print("\n📊 Configuration details:")
        print(f"  Data directory: {config['data_dir']}")
        print(f"  Output directory: {config['output_dir']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Chunk size: {config['chunk_size']}")
        return 0
    
    # Run the pipeline
    try:
        print(f"\n🚀 Starting BGE embedding pipeline...")
        pipeline = iLandBGEBatchEmbeddingPipeline(config)
        pipeline.run()
        print(f"\n✅ Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())