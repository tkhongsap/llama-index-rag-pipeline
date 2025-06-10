#!/usr/bin/env python
"""
iLand All-in-One Processing and Embedding Script

This script provides a unified workflow for:
1. Processing Excel/CSV files to PostgreSQL markdown documents
2. Generating OpenAI embeddings from these documents
3. Storing embeddings in PostgreSQL vector store for retrieval

Usage:
    python process_and_embed_to_postgres.py [options]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure src-iLand is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import required modules
try:
    from data_processing_new.iland_converter import iLandCSVConverter
    from docs_embedding_new.postgres_embedding import PostgresEmbeddingGenerator
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def process_data(args):
    """Process data from Excel/CSV to PostgreSQL markdown documents"""
    logger.info("=== STEP 1: DATA PROCESSING ===")
    
    # Auto-detect paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "input_docs"
    
    # Look for the iLand data file
    input_filename = args.input_file if args.input_file else "input_dataset_iLand.xlsx"
    input_file = input_dir / input_filename
    
    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find iLand data file: {input_file}\n"
            f"Please ensure {input_filename} exists in data/input_docs/"
        )
    
    # Set output directory
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using iLand data file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max rows to process: {args.max_rows or 'All'}")
    logger.info(f"Database host: {args.db_host}, port: {args.db_port}")
    
    # Create iLand converter
    converter = iLandCSVConverter(str(input_file), output_dir)
    
    # Set database connection parameters
    converter.db_manager.db_host = args.db_host
    converter.db_manager.db_port = args.db_port
    converter.db_manager.db_name = args.db_name
    converter.db_manager.db_user = args.db_user
    converter.db_manager.db_password = args.db_password
    
    # Setup configuration
    config = converter.setup_configuration(config_name="iland_deed_records", auto_generate=True)
    
    # Process documents
    logger.info("Processing dataset in batches...")
    documents = converter.process_csv_to_documents(
        batch_size=args.batch_size, 
        max_rows=args.max_rows
    )
    
    # Save documents as JSONL for backup
    jsonl_path = converter.save_documents_as_jsonl(documents)
    
    # Save documents to PostgreSQL database
    logger.info("Saving documents to PostgreSQL database...")
    inserted_count = converter.save_documents_to_database(
        documents, 
        batch_size=args.db_batch_size
    )
    
    # Print summary statistics
    converter.print_summary_statistics(documents)
    
    logger.info("iLand dataset conversion completed successfully!")
    logger.info(f"Total documents created: {len(documents)}")
    logger.info(f"JSONL output: {jsonl_path}")
    logger.info(f"Database insertion: {inserted_count} documents inserted into {args.source_table} table")
    
    return inserted_count


def generate_embeddings(args, document_count):
    """Generate embeddings from PostgreSQL documents and store in vector database"""
    logger.info("\n=== STEP 2: EMBEDDING GENERATION ===")
    
    # Create embedding generator
    generator = PostgresEmbeddingGenerator(
        source_db_host=args.db_host,
        source_db_user=args.db_user,
        source_db_password=args.db_password,
        source_db_name=args.db_name,
        source_db_port=args.db_port,
        source_table_name=args.source_table,
        dest_db_host=args.db_host,
        dest_db_user=args.db_user,
        dest_db_password=args.db_password,
        dest_db_name=args.db_name,
        dest_db_port=args.db_port,
        dest_table_name=args.dest_table,
        embed_model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.api_batch_size
    )
    
    # Process only the documents we just inserted
    logger.info(f"Generating embeddings for {document_count} documents...")
    if args.max_rows and args.max_rows < document_count:
        document_count = args.max_rows
    
    # Run embedding pipeline
    inserted_count = generator.run_pipeline(limit=document_count)
    
    if inserted_count > 0:
        logger.info(f"Success! {inserted_count} nodes with embeddings inserted into PostgreSQL vector store")
        return True
    else:
        logger.error("Embedding generation failed or no documents were processed")
        return False


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process iLand data and generate embeddings for RAG'
    )
    
    # Data processing arguments
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum number of rows to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for processing (default: 500)')
    parser.add_argument('--db-batch-size', type=int, default=100,
                        help='Batch size for database insertion (default: 100)')
    parser.add_argument('--db-host', type=str, default=os.getenv("DB_HOST", "10.4.102.11"),
                        help=f'Database host (default: {os.getenv("DB_HOST", "10.4.102.11")})')
    parser.add_argument('--db-port', type=int, default=int(os.getenv("DB_PORT", "5432")),
                        help=f'Database port (default: {os.getenv("DB_PORT", "5432")})')
    parser.add_argument('--db-name', type=str, default=os.getenv("DB_NAME", "iland-vector-dev"),
                        help=f'Database name (default: {os.getenv("DB_NAME", "iland-vector-dev")})')
    parser.add_argument('--db-user', type=str, default=os.getenv("DB_USER", "vector_user_dev"),
                        help=f'Database user (default: {os.getenv("DB_USER", "vector_user_dev")})')
    parser.add_argument('--db-password', type=str, default=os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
                        help='Database password (default: from .env)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Custom input filename (default: input_dataset_iLand.xlsx)')
    parser.add_argument('--source-table', type=str, default=os.getenv("SOURCE_TABLE", "iland_md_data"),
                        help=f'Source table name (default: {os.getenv("SOURCE_TABLE", "iland_md_data")})')
    
    # Embedding arguments
    parser.add_argument('--model', type=str, default=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                        help=f'OpenAI embedding model (default: {os.getenv("EMBED_MODEL", "text-embedding-3-small")})')
    parser.add_argument('--chunk-size', type=int, default=int(os.getenv("CHUNK_SIZE", "512")),
                        help=f'Chunk size for text splitting (default: {os.getenv("CHUNK_SIZE", "512")})')
    parser.add_argument('--chunk-overlap', type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")),
                        help=f'Chunk overlap for text splitting (default: {os.getenv("CHUNK_OVERLAP", "50")})')
    parser.add_argument('--api-batch-size', type=int, default=int(os.getenv("API_BATCH_SIZE", "20")),
                        help=f'Batch size for API calls (default: {os.getenv("API_BATCH_SIZE", "20")})')
    parser.add_argument('--dest-table', type=str, default=os.getenv("DEST_TABLE", "iland_embeddings"),
                        help=f'Destination table name for embeddings (default: {os.getenv("DEST_TABLE", "iland_embeddings")})')
    
    # Processing control
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip data processing step (only generate embeddings)')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step (only process data)')
    
    args = parser.parse_args()
    
    # Print header
    logger.info("=" * 80)
    logger.info("iLAND RAG PIPELINE: DATA PROCESSING & EMBEDDING GENERATION")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Step 1: Process data (unless skipped)
        document_count = 0
        if not args.skip_processing:
            document_count = process_data(args)
        else:
            logger.info("Skipping data processing step as requested")
            # If we're skipping processing but doing embeddings, use max_rows as document count
            document_count = args.max_rows or 1000
        
        # Step 2: Generate embeddings (unless skipped)
        if not args.skip_embeddings:
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY not found in environment variables. Set it in .env file.")
                return 1
                
            success = generate_embeddings(args, document_count)
            if not success:
                return 1
        else:
            logger.info("Skipping embedding generation step as requested")
        
        # Print completion message
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 