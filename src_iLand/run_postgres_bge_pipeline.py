#!/usr/bin/env python3
"""
Integrated PostgreSQL BGE-M3 Pipeline for iLand RAG

This script combines:
1. Data processing (CSV → Documents)
2. Section-based chunking (Documents → Chunks)
3. BGE-M3 embedding generation (Chunks → Embeddings)
4. PostgreSQL storage

Usage:
    # Process sample data
    python run_postgres_bge_pipeline.py --max-rows 10
    
    # Process all data
    python run_postgres_bge_pipeline.py
    
    # Process specific CSV file
    python run_postgres_bge_pipeline.py --input-file input_dataset_iLand_sample.csv
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import time
from typing import Optional

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our modules
from data_processing_postgres.iland_converter import iLandCSVConverter
from data_processing_postgres.main import process_documents_for_embedding
from docs_embedding_postgres.postgres_embedding_bge import PostgresBGEEmbeddingProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    input_file: str,
    max_rows: Optional[int] = None,
    batch_size: int = 500,
    db_batch_size: int = 100,
    embedding_batch_size: int = 32,
    device: str = "auto"
):
    """Run the complete pipeline from CSV to PostgreSQL embeddings."""
    
    start_time = time.time()
    
    print("=" * 80)
    print("🚀 INTEGRATED POSTGRESQL BGE-M3 PIPELINE FOR ILAND RAG")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Max rows: {max_rows or 'all'}")
    print(f"Processing: 100% local (no API calls)")
    print(f"Embedding: BGE-M3 (1024d)")
    print(f"Storage: PostgreSQL with section-based chunks")
    print("=" * 80)
    
    # Step 1: Process CSV to documents
    print("\n📄 STEP 1: Processing CSV to documents...")
    
    # Get project paths
    project_root = Path(current_dir).parent
    input_path = project_root / "data" / "input_docs" / input_file
    output_dir = str(project_root / "data" / "output_docs")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create converter
    converter = iLandCSVConverter(str(input_path), output_dir)
    
    # Setup configuration
    config = converter.setup_configuration(
        config_name="iland_deed_records", 
        auto_generate=True
    )
    
    # Process documents
    documents = converter.process_csv_to_documents(
        batch_size=batch_size, 
        max_rows=max_rows
    )
    
    print(f"✅ Processed {len(documents)} documents from CSV")
    
    # Step 2: Save documents to PostgreSQL
    print("\n💾 STEP 2: Saving documents to PostgreSQL...")
    
    inserted_count = converter.save_documents_to_database(
        documents, 
        batch_size=db_batch_size
    )
    
    print(f"✅ Inserted {inserted_count} documents into iland_md_data table")
    
    # Step 3: Setup chunks table
    print("\n🗃️ STEP 3: Setting up chunks table...")
    
    converter.db_manager.setup_chunks_table()
    print("✅ Chunks table ready for section-based storage")
    
    # Step 4: Generate section-based chunks
    print("\n✂️ STEP 4: Generating section-based chunks...")
    
    chunks = process_documents_for_embedding(documents, converter.db_manager)
    
    chunks_per_doc = len(chunks) / len(documents) if documents else 0
    print(f"✅ Generated {len(chunks)} chunks from {len(documents)} documents")
    print(f"📊 Average chunks per document: {chunks_per_doc:.1f}")
    
    # Step 5: Generate BGE-M3 embeddings
    print("\n🤗 STEP 5: Generating BGE-M3 embeddings...")
    
    # Close the converter's database connection first
    converter.db_manager.close()
    
    # Initialize embedding processor
    embedding_processor = PostgresBGEEmbeddingProcessor(
        device=device,
        batch_size=embedding_batch_size
    )
    
    # Run embedding pipeline
    embeddings_generated = embedding_processor.run_pipeline(
        batch_limit=100,
        total_limit=None  # Process all chunks
    )
    
    print(f"✅ Generated {embeddings_generated} embeddings")
    
    # Step 6: Verify results
    print("\n📊 STEP 6: Verifying results...")
    
    stats = embedding_processor.verify_embeddings()
    
    print(f"   Total chunks: {stats.get('total_chunks', 0)}")
    print(f"   Chunks with embeddings: {stats.get('chunks_with_embeddings', 0)}")
    print(f"   BGE-M3 embeddings: {stats.get('bge_m3_embeddings', 0)}")
    print(f"   Average dimension: {stats.get('avg_embedding_dim', 0):.1f}")
    print(f"   Coverage: {stats.get('chunks_with_embeddings', 0) / stats.get('total_chunks', 1) * 100:.1f}%")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📄 Documents processed: {len(documents)}")
    print(f"✂️ Chunks created: {len(chunks)} (avg {chunks_per_doc:.1f} per doc)")
    print(f"🤗 Embeddings generated: {embeddings_generated}")
    print(f"💾 Storage: PostgreSQL (iland_md_data + iland_chunks)")
    print(f"🔍 Ready for vector search with BGE-M3!")
    print("=" * 80)
    
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "embeddings": embeddings_generated,
        "chunks_per_doc": chunks_per_doc,
        "total_time": total_time
    }


def main():
    """CLI entry point for the integrated pipeline."""
    parser = argparse.ArgumentParser(
        description='Integrated PostgreSQL BGE-M3 Pipeline for iLand RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 10 sample documents
  python run_postgres_bge_pipeline.py --max-rows 10
  
  # Process all documents
  python run_postgres_bge_pipeline.py
  
  # Process specific CSV file
  python run_postgres_bge_pipeline.py --input-file my_data.csv
  
  # Use GPU for faster processing
  python run_postgres_bge_pipeline.py --device cuda

Pipeline Steps:
  1. CSV → Documents (with rich metadata extraction)
  2. Documents → PostgreSQL (iland_md_data table)
  3. Documents → Section-based chunks (6-10 per doc)
  4. Chunks → PostgreSQL (iland_chunks table)
  5. Chunks → BGE-M3 embeddings (1024d)
  6. Embeddings → PostgreSQL (update chunks)

Features:
  - 100% local processing (no API calls)
  - BGE-M3 model with Thai language support
  - Section-based chunking (vs 169 chunks with sentence splitting)
  - Rich metadata preservation
  - Efficient PostgreSQL storage
        """
    )
    
    parser.add_argument(
        '--input-file',
        default='input_dataset_iLand_sample.csv',
        help='Input CSV filename (default: input_dataset_iLand_sample.csv)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        help='Maximum number of rows to process (default: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Batch size for CSV processing (default: 500)'
    )
    parser.add_argument(
        '--db-batch-size',
        type=int,
        default=100,
        help='Batch size for database insertion (default: 100)'
    )
    parser.add_argument(
        '--embedding-batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device for BGE-M3 model (default: auto)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_complete_pipeline(
            input_file=args.input_file,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
            db_batch_size=args.db_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            device=args.device
        )
        
        print("\n🎉 Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()