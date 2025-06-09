"""
Main execution script for iLand dataset processing

This script provides a simple interface to process iLand CSV files into documents
using the refactored modular components.
"""

import logging
import argparse
from pathlib import Path
import sys
import os

# Handle both relative and absolute imports
try:
    from .iland_converter import iLandCSVConverter
except ImportError:
    # Add parent directory to path to enable running as standalone script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from data_processing_new.iland_converter import iLandCSVConverter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main execution function for iLand dataset processing"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process iLand CSV data and save to PostgreSQL database')
    parser.add_argument('--max-rows', type=int, default=None, 
                        help='Maximum number of rows to process from CSV (default: all rows)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for processing (default: 500)')
    parser.add_argument('--db-batch-size', type=int, default=100,
                        help='Batch size for database insertion (default: 100)')
    parser.add_argument('--db-host', type=str, default='10.4.102.11',
                        help='Database host (default: 10.4.102.11)')
    parser.add_argument('--db-port', type=int, default=5432,
                        help='Database port (default: 5432)')
    parser.add_argument('--csv-file', type=str, default=None,
                        help='Custom CSV filename (default: input_dataset.csv)')
    args = parser.parse_args()
    
    # Auto-detect correct paths based on script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up two levels from src-iLand/data_processing/
    input_dir = project_root / "data" / "input_docs"
    
    # Look for the iLand CSV file
    csv_filename = args.csv_file if args.csv_file else "input_dataset.csv"
    input_csv = input_dir / csv_filename
    
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find iLand CSV file: {input_csv}\n"
            f"Please ensure {csv_filename} exists in data/input_docs/"
        )
    
    # Set output directory
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using iLand CSV file: {input_csv}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max rows to process: {args.max_rows or 'All'}")
    logger.info(f"Database host: {args.db_host}, port: {args.db_port}")
    
    # Create iLand converter
    converter = iLandCSVConverter(str(input_csv), output_dir)
    
    # Set database connection parameters
    converter.db_manager.db_host = args.db_host
    converter.db_manager.db_port = args.db_port
    
    # Setup configuration (auto-generate from CSV analysis)
    config = converter.setup_configuration(config_name="iland_deed_records", auto_generate=True)
    
    # Process documents in smaller batches due to large dataset
    logger.info("Processing large dataset in batches...")
    documents = converter.process_csv_to_documents(batch_size=args.batch_size, max_rows=args.max_rows)
    
    # Save documents as JSONL
    jsonl_path = converter.save_documents_as_jsonl(documents)
    
    # Save documents to PostgreSQL database instead of markdown files
    logger.info("Saving documents to PostgreSQL database...")
    inserted_count = converter.save_documents_to_database(documents, batch_size=args.db_batch_size)
    
    # Print summary statistics
    converter.print_summary_statistics(documents)
    
    logger.info("iLand dataset conversion completed successfully!")
    logger.info(f"Total documents created: {len(documents)}")
    logger.info(f"Configuration: {config.name}")
    logger.info(f"JSONL output: {jsonl_path}")
    logger.info(f"Database insertion: {inserted_count} documents inserted into iland_md_data table")


if __name__ == "__main__":
    main() 