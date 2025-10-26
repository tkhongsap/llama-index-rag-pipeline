"""
Main pipeline orchestrator for sales promotion data processing.

Coordinates the full workflow: CSV → SimpleDocuments → dual output (JSONL + markdown)
"""

import logging
import argparse
from pathlib import Path
from typing import List
import sys

# Handle both relative and absolute imports
try:
    from .models import PromotionDocument, SimpleDocument
    from .config_manager import ConfigManager
    from .processors.csv_processor import CSVProcessor
    from .processors.metadata_extractor import MetadataExtractor
    from .processors.thai_normalizer import ThaiNormalizer
    from .file_output import FileOutputManager
    from .statistics_generator import StatisticsGenerator
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from models import PromotionDocument, SimpleDocument
    from config_manager import ConfigManager
    from processors.csv_processor import CSVProcessor
    from processors.metadata_extractor import MetadataExtractor
    from processors.thai_normalizer import ThaiNormalizer
    from file_output import FileOutputManager
    from statistics_generator import StatisticsGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromotionDataProcessor:
    """
    Main pipeline orchestrator for promotion data processing.
    
    Workflow:
    1. Load CSV with Thai text
    2. Extract base metadata (10 fields)
    3. Derive metadata (15+ fields) using utilities
    4. Normalize Thai text
    5. Create PromotionDocument → SimpleDocument
    6. Save JSONL + markdown files
    7. Generate statistics and reports
    """
    
    def __init__(self, config_path: str = None, output_dir: str = None):
        """
        Initialize the processor.
        
        Args:
            config_path: Path to YAML config (None = use default)
            output_dir: Output directory (None = use default)
        """
        # Determine paths
        if output_dir is None:
            module_dir = Path(__file__).parent.parent
            output_dir = str(module_dir / "data" / "processed")
        
        self.output_dir = output_dir
        
        # Initialize components
        self.config_manager = ConfigManager(output_dir)
        self.config = self.config_manager.load_config(config_path)
        self.config_manager.validate_config(self.config)
        
        # Initialize processors
        self.csv_processor = CSVProcessor(self.config)
        self.metadata_extractor = MetadataExtractor(self.config)
        self.thai_normalizer = ThaiNormalizer()
        self.file_output = FileOutputManager(output_dir, self.config)
        self.statistics = StatisticsGenerator()
        
        logger.info(f"✓ Processor initialized")
        logger.info(f"  Config: {self.config.name} v{self.config.version}")
        logger.info(f"  Output: {output_dir}")
    
    def process(self, csv_path: str) -> List[SimpleDocument]:
        """
        Main processing pipeline: CSV → SimpleDocuments.
        
        Args:
            csv_path: Path to input CSV file
        
        Returns:
            List of SimpleDocument objects
        """
        logger.info("\n" + "=" * 60)
        logger.info("SALES PROMOTION DATA PROCESSING PIPELINE - PHASE 1")
        logger.info("=" * 60)
        
        self.statistics.start_timer()
        
        # Step 1: Load CSV
        logger.info("\n[1/7] Loading CSV...")
        df = self.csv_processor.load_csv(csv_path)
        self.csv_processor.validate_csv_structure(df)
        
        # Step 2: Process each row
        logger.info("\n[2/7] Processing rows...")
        promotion_docs = []
        failed_rows = []
        
        for idx, row in df.iterrows():
            try:
                # Extract base metadata (10 fields)
                base_metadata = self.csv_processor.extract_fields(row)
                
                # Derive metadata (15+ fields)
                derived_metadata = self.metadata_extractor.derive_metadata(base_metadata)
                
                # Normalize Thai text
                base_metadata['product_name'] = self.thai_normalizer.normalize_text(
                    base_metadata.get('product_name', '')
                )
                base_metadata['buy_unit'] = self.thai_normalizer.normalize_unit(
                    base_metadata.get('buy_unit', '')
                )
                base_metadata['free_unit'] = self.thai_normalizer.normalize_unit(
                    base_metadata.get('free_unit', '')
                )
                
                # Create PromotionDocument
                promo_doc = PromotionDocument(
                    document_id=f"PROMO_{str(idx+1).zfill(3)}",
                    **base_metadata,
                    **derived_metadata
                )
                
                promotion_docs.append(promo_doc)
                
                # Log progress every 10 rows
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Processed {idx+1}/{len(df)} rows")
                
            except Exception as e:
                logger.error(f"Failed to process row {idx}: {e}")
                failed_rows.append({'row': idx, 'error': str(e)})
        
        logger.info(f"✓ Processed {len(promotion_docs)}/{len(df)} rows successfully")
        
        # Step 3: Convert to SimpleDocuments
        logger.info("\n[3/7] Converting to SimpleDocuments...")
        simple_documents = [doc.to_simple_document() for doc in promotion_docs]
        logger.info(f"✓ Created {len(simple_documents)} SimpleDocument objects")
        
        # Step 4: Save JSONL output (primary)
        logger.info("\n[4/7] Saving JSONL output...")
        jsonl_path = self.file_output.save_documents_as_jsonl(simple_documents)
        self.file_output.validate_jsonl_output(jsonl_path)
        
        # Step 5: Save markdown files (secondary)
        logger.info("\n[5/7] Saving markdown files...")
        markdown_files = self.file_output.save_documents_as_markdown_files(simple_documents)
        
        # Step 6: Generate statistics
        logger.info("\n[6/7] Generating statistics...")
        self.statistics.stop_timer()
        stats = self.statistics.generate_statistics(
            simple_documents,
            self.config.name,
            self.config.version
        )
        
        # Step 7: Save reports
        logger.info("\n[7/7] Saving reports...")
        self.config_manager.save_config_snapshot(self.config)
        self.config_manager.save_processing_report(stats)
        
        if failed_rows:
            self.config_manager.save_error_report(failed_rows)
        
        # Generate README
        self.config_manager.save_readme(stats)
        
        # Print summary
        self.statistics.print_summary(stats)
        
        logger.info("\n✅ PROCESSING COMPLETE!")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  JSONL file: {jsonl_path}")
        logger.info(f"  Markdown files: {len(markdown_files)} files")
        
        return simple_documents


def main():
    """
    Main entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Sales Promotion Data Processing Pipeline - Phase 1"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: src-sale-promotion/data/processed)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config YAML (default: config/promotion_fields.yaml)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input CSV path
    if args.input:
        csv_path = args.input
    else:
        # Auto-detect: look for promotion_list.csv
        module_dir = Path(__file__).parent.parent
        csv_path = str(module_dir / "data" / "promotion_list.csv")
    
    # Check CSV exists
    if not Path(csv_path).exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.error("Please specify --input or place promotion_list.csv in data/ directory")
        sys.exit(1)
    
    # Initialize processor
    try:
        processor = PromotionDataProcessor(
            config_path=args.config,
            output_dir=args.output
        )
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Dry run: just validate config
    if args.dry_run:
        logger.info("✓ Dry run successful - configuration is valid")
        sys.exit(0)
    
    # Process data
    try:
        documents = processor.process(csv_path)
        logger.info(f"\n🎉 Successfully processed {len(documents)} promotions!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

