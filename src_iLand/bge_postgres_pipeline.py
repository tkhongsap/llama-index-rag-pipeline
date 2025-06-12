#!/usr/bin/env python
"""
iLand BGE Embedding with PostgreSQL Storage

This script provides a unified workflow for:
1. Processing Excel/CSV files to PostgreSQL markdown documents
2. Generating BGE embeddings from these documents
3. Storing embeddings in PostgreSQL vector store in 4 separate tables:
   - iland_chunks: Chunked document content
   - iland_summaries: Document summaries
   - iland_indexnodes: Document index nodes
   - iland_combined: Combined embeddings for faster search

Usage:
    python bge_postgres_pipeline.py [options]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
from typing import Tuple
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
    from data_processing_postgres.iland_converter import iLandCSVConverter
    from docs_embedding_postgres.embeddings_manager import EmbeddingsManager
    from docs_embedding_postgres.db_utils import PostgresManager
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
    
    if args.filter_province:
        logger.info(f"Filtering data for province: {args.filter_province}")
    
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
        max_rows=args.max_rows,
        filter_province=args.filter_province
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


def generate_embeddings(args, document_count) -> bool:
    """Generate BGE embeddings and store in 4 separate PostgreSQL tables"""
    logger.info("\n=== STEP 2: BGE EMBEDDING GENERATION ===")
    
    # Create managers
    embeddings_manager = EmbeddingsManager(
        bge_model=args.bge_model,
        cache_folder=args.cache_folder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.embed_batch_size
    )
    
    db_manager = PostgresManager(
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        db_host=args.db_host,
        db_port=args.db_port,
        source_table=args.source_table,
        chunks_table=args.chunks_table,
        summaries_table=args.summaries_table,
        indexnodes_table=args.indexnodes_table,
        combined_table=args.combined_table
    )
    
    try:
        # Connect to database
        if not db_manager.connect():
            logger.error("Failed to connect to database. Aborting.")
            return False
        
        # Setup database tables
        vector_dimension = embeddings_manager.get_model_dimension()
        if not db_manager.setup_tables(vector_dimension):
            logger.error("Failed to setup database tables. Aborting.")
            return False
        
        # Fetch documents from database
        logger.info(f"Fetching up to {document_count} documents from database...")
        documents = db_manager.fetch_documents(limit=document_count)
        if not documents:
            logger.error("No documents found in the database. Aborting.")
            return False
        
        # Process documents and generate embeddings
        logger.info(f"Processing {len(documents)} documents to generate embeddings...")
        chunk_embeddings, summary_embeddings, indexnode_embeddings = embeddings_manager.process_documents(documents)
        
        # Save embeddings to database
        logger.info("Saving embeddings to database...")
        chunks_count, summaries_count, indexnodes_count, combined_count = db_manager.save_all_embeddings(
            chunk_embeddings, summary_embeddings, indexnode_embeddings
        )
        
        total_count = chunks_count + summaries_count + indexnodes_count
        
        if total_count > 0:
            logger.info(f"Success! Embeddings created and saved to PostgreSQL tables:")
            logger.info(f"  - {chunks_count} chunks saved to {args.chunks_table}")
            logger.info(f"  - {summaries_count} summaries saved to {args.summaries_table}")
            logger.info(f"  - {indexnodes_count} index nodes saved to {args.indexnodes_table}")
            logger.info(f"  - {combined_count} combined entries saved to {args.combined_table}")
            return True
        else:
            logger.error("Embedding generation failed or no documents were processed")
            return False
            
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        return False
    finally:
        # Close database connection
        db_manager.close()


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process iLand data and generate BGE embeddings for RAG'
    )
    
    # Data processing arguments
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum number of rows to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for processing (default: 500)')
    parser.add_argument('--db-batch-size', type=int, default=100,
                        help='Batch size for database insertion (default: 100)')
    parser.add_argument('--db-host', type=str, default=os.getenv("DB_HOST"),
                        help=f'Database host (default: {os.getenv("DB_HOST")})')
    parser.add_argument('--db-port', type=int, default=int(os.getenv("DB_PORT")),
                        help=f'Database port (default: {os.getenv("DB_PORT")})')
    parser.add_argument('--db-name', type=str, default=os.getenv("DB_NAME", "iland-vector-dev"),
                        help=f'Database name (default: {os.getenv("DB_NAME", "iland-vector-dev")})')
    parser.add_argument('--db-user', type=str, default=os.getenv("DB_USER", "vector_user_dev"),
                        help=f'Database user (default: {os.getenv("DB_USER", "vector_user_dev")})')
    parser.add_argument('--db-password', type=str, default=os.getenv("DB_PASSWORD"),
                        help='Database password (default: from .env)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Custom input filename (default: input_dataset_iLand.xlsx)')
    parser.add_argument('--source-table', type=str, default=os.getenv("SOURCE_TABLE", "iland_md_data"),
                        help=f'Source table name (default: {os.getenv("SOURCE_TABLE", "iland_md_data")})')
    parser.add_argument('--filter-province', type=str, default="ชัยนาท",
                        help='Filter data by province name (default: "ชัยนาท")')
    
    # BGE model arguments
    parser.add_argument('--bge-model', type=str, default=os.getenv("BGE_MODEL", "bge-small-en-v1.5"),
                        help=f'BGE model name (default: {os.getenv("BGE_MODEL", "bge-small-en-v1.5")})')
    parser.add_argument('--cache-folder', type=str, default=os.getenv("CACHE_FOLDER", "./cache/bge_models"),
                        help=f'BGE model cache folder (default: {os.getenv("CACHE_FOLDER", "./cache/bge_models")})')
    
    # Embedding arguments
    parser.add_argument('--chunk-size', type=int, default=int(os.getenv("CHUNK_SIZE", "512")),
                        help=f'Chunk size for text splitting (default: {os.getenv("CHUNK_SIZE", "512")})')
    parser.add_argument('--chunk-overlap', type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")),
                        help=f'Chunk overlap for text splitting (default: {os.getenv("CHUNK_OVERLAP", "50")})')
    parser.add_argument('--embed-batch-size', type=int, default=int(os.getenv("API_BATCH_SIZE", "20")),
                        help=f'Batch size for embedding generation (default: {os.getenv("API_BATCH_SIZE", "20")})')
                        
    # Output table names
    parser.add_argument('--chunks-table', type=str, default=os.getenv("CHUNKS_TABLE", "iland_chunks"),
                        help=f'Chunks table name (default: {os.getenv("CHUNKS_TABLE", "iland_chunks")})')
    parser.add_argument('--summaries-table', type=str, default=os.getenv("SUMMARIES_TABLE", "iland_summaries"),
                        help=f'Summaries table name (default: {os.getenv("SUMMARIES_TABLE", "iland_summaries")})')
    parser.add_argument('--indexnodes-table', type=str, default=os.getenv("INDEXNODES_TABLE", "iland_indexnodes"),
                        help=f'Index nodes table name (default: {os.getenv("INDEXNODES_TABLE", "iland_indexnodes")})')
    parser.add_argument('--combined-table', type=str, default=os.getenv("COMBINED_TABLE", "iland_combined"),
                        help=f'Combined table name (default: {os.getenv("COMBINED_TABLE", "iland_combined")})')
    
    # Processing control
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip data processing step (only generate embeddings)')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step (only process data)')
    parser.add_argument('--no-province-filter', action='store_true',
                        help='Disable province filtering (process all provinces)')
    
    args = parser.parse_args()
    
    # If no-province-filter flag is set, set filter_province to None
    if args.no_province_filter:
        args.filter_province = None
    
    # Print header
    logger.info("=" * 80)
    logger.info("iLAND BGE RAG PIPELINE: DATA PROCESSING & EMBEDDING GENERATION")
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
        
        # # Step 2: Generate embeddings (unless skipped)
        # if not args.skip_embeddings:                
        #     success = generate_embeddings(args, document_count)
        #     if not success:
        #         return 1
        # else:
        #     logger.info("Skipping embedding generation step as requested")
        
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