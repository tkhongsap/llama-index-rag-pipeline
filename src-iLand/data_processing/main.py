"""
Main execution script for iLand dataset processing

This script provides a simple interface to process iLand CSV files into documents
using the refactored modular components.
"""

import logging
from pathlib import Path
from .iland_converter import iLandCSVConverter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main execution function for iLand dataset processing"""
    
    # Auto-detect correct paths based on script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up two levels from src-iLand/data_processing/
    input_dir = project_root / "data" / "input_docs"
    
    # Look for the iLand CSV file
    input_csv = input_dir / "input_dataset_iLand.csv"
    
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find iLand CSV file: {input_csv}\n"
            "Please ensure input_dataset_iLand.csv exists in data/input_docs/"
        )
    
    # Set output directory
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using iLand CSV file: {input_csv}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create iLand converter
    converter = iLandCSVConverter(str(input_csv), output_dir)
    
    # Setup configuration (auto-generate from CSV analysis)
    config = converter.setup_configuration(config_name="iland_deed_records", auto_generate=True)
    
    # Process documents in smaller batches due to large dataset
    logger.info("Processing large dataset in batches...")
    documents = converter.process_csv_to_documents(batch_size=500)
    
    # Save documents as JSONL
    jsonl_path = converter.save_documents_as_jsonl(documents)
    
    # Save documents as individual markdown files
    markdown_files = converter.save_documents_as_markdown_files(documents, batch_size=1000)
    
    # Print summary statistics
    converter.print_summary_statistics(documents)
    
    logger.info("iLand dataset conversion completed successfully!")
    logger.info(f"Total documents created: {len(documents)}")
    logger.info(f"Configuration: {config.name}")
    logger.info(f"JSONL output: {jsonl_path}")
    logger.info(f"Markdown files: {len(markdown_files)} files in {output_dir}/iland_markdown_files/")


if __name__ == "__main__":
    main() 