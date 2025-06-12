#!/usr/bin/env python3
"""
CLI script to run the BGE-based PostgreSQL embedding pipeline

This script uses BGE models for local embedding generation (no API calls)
with the same sophisticated processing as the local version, including 
rich metadata extraction and section-based chunking.
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='BGE-based PostgreSQL Embedding Pipeline for iLand Thai Land Deeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents with default BGE multilingual model (recommended for Thai)
  python run_postgres_embedding.py
  
  # Process only 10 documents for testing
  python run_postgres_embedding.py --limit 10
  
  # Use specific BGE model
  python run_postgres_embedding.py --model bge-large-en-v1.5
  
  # Use custom cache folder and enable OpenAI fallback
  python run_postgres_embedding.py --cache-folder /path/to/cache --enable-openai-fallback

Available BGE models:
  - bge-small-en-v1.5: Fast, 384 dimensions
  - bge-base-en-v1.5: Balanced, 768 dimensions  
  - bge-large-en-v1.5: High quality, 1024 dimensions
  - bge-m3: Multilingual (Thai support), 1024 dimensions (DEFAULT)
        """
    )
    
    parser.add_argument(
        '--limit', 
        type=int, 
        help='Limit number of documents to process (default: all)'
    )
    
    parser.add_argument(
        '--model', 
        default='bge-m3',
        choices=['bge-small-en-v1.5', 'bge-base-en-v1.5', 'bge-large-en-v1.5', 'bge-m3'],
        help='BGE model to use (default: bge-m3 for Thai support)'
    )
    
    parser.add_argument(
        '--cache-folder',
        default='./cache/bge_models',
        help='Cache folder for BGE models (default: ./cache/bge_models)'
    )
    
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=512,
        help='Chunk size for text splitting (default: 512)'
    )
    
    parser.add_argument(
        '--chunk-overlap', 
        type=int, 
        default=50,
        help='Chunk overlap for text splitting (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=20,
        help='Batch size for processing (default: 20)'
    )
    
    parser.add_argument(
        '--disable-section-chunking',
        action='store_true',
        help='Disable section-based chunking (use sentence splitting instead)'
    )
    
    parser.add_argument(
        '--enable-openai-fallback',
        action='store_true',
        help='Enable OpenAI fallback if BGE fails'
    )
    
    args = parser.parse_args()
    
    print("🤗 BGE PostgreSQL Embedding Pipeline for iLand")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Cache folder: {args.cache_folder}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    print(f"Batch size: {args.batch_size}")
    print(f"Section chunking: {not args.disable_section_chunking}")
    print(f"OpenAI fallback: {args.enable_openai_fallback}")
    print(f"Document limit: {args.limit or 'All'}")
    print("=" * 60)
    
    try:
        # Import BGE-based processor
        from postgres_embedding_bge import BGEPostgresEmbeddingGenerator
        
        generator = BGEPostgresEmbeddingGenerator(
            bge_model_key=args.model,
            cache_folder=args.cache_folder,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            enable_section_chunking=not args.disable_section_chunking,
            fallback_to_openai=args.enable_openai_fallback
        )
        
        result = generator.run_pipeline(limit=args.limit)
        
        if result > 0:
            print(f"\n🎉 Success! Processed {result} embeddings using BGE model: {args.model}")
            print(f"   ✅ All processing done locally (no API calls)")
            print(f"   📊 Rich metadata extraction with 30+ Thai land deed fields")
            print(f"   🔄 Section-based chunking for semantic coherence")
        else:
            print(f"\n❌ Pipeline failed or no embeddings created")
            return 1
            
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure BGE dependencies are installed:")
        print("  pip install llama-index-embeddings-huggingface sentence-transformers")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        print(f"\n❌ Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 