#!/usr/bin/env python3
"""
Standalone script to run iLand data processing

This script can be run directly without import issues.
It adds the necessary paths and imports to work standalone.
"""

import sys
import os
import logging
import warnings
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add parent directories to path if needed
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import our modules
try:
    from iland_converter import iLandCSVConverter
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the same directory.")
    sys.exit(1)

# Suppress pandas date parsing warnings to reduce noise
warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed individually')

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
        logger.error(f"Could not find iLand CSV file: {input_csv}")
        logger.error("Please ensure input_dataset_iLand.csv exists in data/input_docs/")
        return False
    
    # Set output directory
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using iLand CSV file: {input_csv}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Create iLand converter using the refactored components
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
        
        return True
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 