#!/usr/bin/env python3
"""
Standalone data processing script for iLand dataset processing.
This version fixes the relative import issues by using absolute imports.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

# Now import the modules directly
try:
    from iland_converter import iLandCSVConverter
    from models import DatasetConfig
    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are in the same directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main execution function for iLand dataset processing"""
    
    print("🚀 Starting iLand Data Processing Pipeline")
    print("="*60)
    
    # Auto-detect correct paths based on script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up two levels from src-iLand/data_processing/
    input_dir = project_root / "data" / "input_docs"
    
    # Look for the iLand CSV file
    input_csv = input_dir / "input_dataset_iLand.csv"
    
    if not input_csv.exists():
        # Try alternative locations
        alternative_paths = [
            project_root / "input_dataset_iLand.csv",
            project_root / "data" / "input_dataset_iLand.csv",
            script_dir / "input_dataset_iLand.csv"
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                input_csv = alt_path
                break
        else:
            print(f"❌ Could not find iLand CSV file")
            print(f"Checked locations:")
            print(f"  - {input_dir / 'input_dataset_iLand.csv'}")
            for alt_path in alternative_paths:
                print(f"  - {alt_path}")
            print("\nPlease ensure input_dataset_iLand.csv exists in one of these locations.")
            return False
    
    # Set output directory
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using iLand CSV file: {input_csv}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Create iLand converter
        print("🔧 Initializing iLand converter...")
        converter = iLandCSVConverter(str(input_csv), output_dir)
        
        # Setup configuration (auto-generate from CSV analysis)
        print("⚙️ Setting up configuration...")
        config = converter.setup_configuration(config_name="iland_deed_records", auto_generate=True)
        
        # Process documents in smaller batches due to large dataset
        print("📄 Processing documents in batches...")
        logger.info("Processing large dataset in batches...")
        documents = converter.process_csv_to_documents(batch_size=500)
        
        # Save documents as JSONL
        print("💾 Saving documents as JSONL...")
        jsonl_path = converter.save_documents_as_jsonl(documents)
        
        # Save documents as individual markdown files
        print("📝 Saving documents as markdown files...")
        markdown_files = converter.save_documents_as_markdown_files(documents, batch_size=1000)
        
        # Print summary statistics
        print("\n📊 PROCESSING SUMMARY:")
        print("="*40)
        converter.print_summary_statistics(documents)
        
        logger.info("iLand dataset conversion completed successfully!")
        print(f"\n✅ SUCCESS! iLand dataset conversion completed!")
        print(f"📋 Total documents created: {len(documents)}")
        print(f"⚙️ Configuration: {config.name}")
        print(f"💾 JSONL output: {jsonl_path}")
        print(f"📝 Markdown files: {len(markdown_files)} files in {output_dir}/iland_markdown_files/")
        
        print(f"\n🎯 Next Steps:")
        print(f"- Your structured documents are ready for section-based chunking")
        print(f"- No need to reprocess - use the new section_parser.py with these documents")
        print(f"- Test section chunking with: python working_section_demo.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"❌ Error during processing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Data processing completed successfully!")
    else:
        print("\n💥 Data processing failed!")
        sys.exit(1) 