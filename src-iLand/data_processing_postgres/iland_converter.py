import pandas as pd
import os
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .models import SimpleDocument, DatasetConfig
    from .csv_analyzer import CSVAnalyzer
    from .config_manager import ConfigManager
    from .document_processor import DocumentProcessor
    from .file_output import FileOutputManager
    from .statistics_generator import StatisticsGenerator
    from .db_manager import DatabaseManager
except ImportError:
    from models import SimpleDocument, DatasetConfig
    from csv_analyzer import CSVAnalyzer
    from config_manager import ConfigManager
    from document_processor import DocumentProcessor
    from file_output import FileOutputManager
    from statistics_generator import StatisticsGenerator
    from db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class iLandCSVConverter:
    """Main converter class that orchestrates CSV to Document conversion for iLand dataset"""
    
    def __init__(self, input_file_path: str, output_dir: str, config_path: str = None):
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        self.config_path = config_path
        self.dataset_config = None
        self.encoding = 'utf-8'
        self.is_excel = input_file_path.lower().endswith(('.xlsx', '.xls'))
        
        # Initialize components
        self.csv_analyzer = CSVAnalyzer()
        self.config_manager = ConfigManager(output_dir)
        self.document_processor = None  # Will be initialized after config setup
        self.file_output_manager = None  # Will be initialized after config setup
        self.statistics_generator = None  # Will be initialized after config setup
        self.db_manager = DatabaseManager()  # Initialize database manager
    
    def setup_configuration(self, config_name: str = None, auto_generate: bool = True) -> DatasetConfig:
        """Setup configuration for the CSV conversion"""
        
        if self.config_path and os.path.exists(self.config_path):
            logger.info(f"Loading existing configuration: {self.config_path}")
            self.dataset_config = self.config_manager.load_config(self.config_path)
        elif auto_generate:
            logger.info("Auto-generating configuration from iLand data file analysis")
            analysis = self.csv_analyzer.analyze_csv_structure(self.input_file_path)
            
            # Update encoding based on analysis
            self.encoding = analysis['encoding_used']
            
            if config_name is None:
                config_name = "iland_deed_records"
            
            self.dataset_config = self.csv_analyzer.create_config_from_analysis(analysis, config_name)
            
            # Save the generated config for future use
            config_path = self.config_manager.save_config(self.dataset_config)
            logger.info(f"Generated configuration saved to: {config_path}")
            
            # Also save the analysis report
            self.config_manager.save_analysis_report(analysis)
        else:
            raise ValueError("No configuration provided and auto_generate is False")
        
        # Initialize other components now that we have config
        self.document_processor = DocumentProcessor(self.dataset_config)
        self.file_output_manager = FileOutputManager(self.output_dir, self.dataset_config)
        self.statistics_generator = StatisticsGenerator(self.dataset_config)
        
        return self.dataset_config
    
    def _read_data_in_chunks(self, batch_size: int = 1000, filter_province: Optional[str] = None):
        """Read data from file in chunks, supporting both CSV and Excel formats with optional province filtering"""
        if self.is_excel:
            # For Excel, we read the entire file once, then yield chunks
            logger.info(f"Reading Excel file: {self.input_file_path}")
            df = pd.read_excel(self.input_file_path)
            
            # Apply province filter if specified
            if filter_province:
                logger.info(f"Filtering data for province: {filter_province}")
                if 'deed_current_province_name_th' in df.columns:
                    df = df[df['deed_current_province_name_th'] == filter_province]
                    logger.info(f"After filtering, {len(df)} rows remain for province {filter_province}")
                else:
                    logger.warning(f"Column 'deed_current_province_name_th' not found in dataset, cannot filter by province")
            
            total_rows = len(df)
            
            for i in range(0, total_rows, batch_size):
                yield df.iloc[i:min(i + batch_size, total_rows)]
        else:
            # For CSV, if we need filtering we read the whole file first
            if filter_province:
                logger.info(f"Reading CSV file with encoding {self.encoding}: {self.input_file_path}")
                df = pd.read_csv(self.input_file_path, encoding=self.encoding)
                
                logger.info(f"Filtering data for province: {filter_province}")
                if 'deed_current_province_name_th' in df.columns:
                    df = df[df['deed_current_province_name_th'] == filter_province]
                    logger.info(f"After filtering, {len(df)} rows remain for province {filter_province}")
                else:
                    logger.warning(f"Column 'deed_current_province_name_th' not found in dataset, cannot filter by province")
                
                total_rows = len(df)
                for i in range(0, total_rows, batch_size):
                    yield df.iloc[i:min(i + batch_size, total_rows)]
            else:
                # Without filtering, use built-in chunking support
                logger.info(f"Reading CSV file with encoding {self.encoding}: {self.input_file_path}")
                for chunk in pd.read_csv(self.input_file_path, chunksize=batch_size, encoding=self.encoding):
                    yield chunk
    
    def process_csv_to_documents(self, batch_size: int = 1000, max_rows: int = None, filter_province: Optional[str] = None) -> List[SimpleDocument]:
        """
        Process entire data file and convert to documents
        
        Args:
            batch_size: Number of rows to process in each batch
            max_rows: Maximum number of rows to process (None = all rows)
            filter_province: Filter data by province name (e.g., "ชัยนาท")
        """
        
        if self.dataset_config is None:
            raise ValueError("Configuration not set up. Call setup_configuration() first.")
        
        file_type = "Excel" if self.is_excel else "CSV"
        logger.info(f"Starting conversion of {file_type} data using config: {self.dataset_config.name}")
        logger.info(f"Using encoding: {self.encoding}")
        logger.info(f"Priority fields for embedding: {self.dataset_config.embedding_fields}")
        
        if filter_province:
            logger.info(f"Filtering data for province: {filter_province}")
        
        documents = []
        chunk_num = 0
        total_rows = 0
        failed_rows = []
        start_time = datetime.now()
        
        for chunk in self._read_data_in_chunks(batch_size, filter_province=filter_province):
            chunk_num += 1
            chunk_start_time = datetime.now()
            
            logger.info(f"Processing chunk {chunk_num} ({len(chunk)} rows) - Total processed so far: {total_rows:,}")
            
            chunk_failed = 0
            for idx, row in chunk.iterrows():
                total_rows += 1
                
                # Stop if max_rows is reached
                if max_rows is not None and total_rows > max_rows:
                    logger.info(f"Reached maximum number of rows to process: {max_rows}")
                    break
                    
                try:
                    document = self.document_processor.convert_row_to_document(row, row_index=idx)
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {e}")
                    failed_rows.append({'row': idx, 'error': str(e)})
                    chunk_failed += 1
            
            # Chunk completion info
            chunk_time = (datetime.now() - chunk_start_time).total_seconds()
            success_rate = ((len(chunk) - chunk_failed) / len(chunk)) * 100
            elapsed_total = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✓ Chunk {chunk_num} completed in {chunk_time:.1f}s - Success: {success_rate:.1f}% - Total time: {elapsed_total/60:.1f}min")
            
            # Progress summary every 10 chunks
            if chunk_num % 10 == 0:
                avg_time_per_chunk = elapsed_total / chunk_num
                logger.info(f"📊 Progress: {chunk_num} chunks, {len(documents):,} documents, avg {avg_time_per_chunk:.1f}s/chunk")
                
            # Stop if max_rows is reached
            if max_rows is not None and total_rows >= max_rows:
                logger.info(f"Reached maximum number of rows to process: {max_rows}")
                break
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 Processing completed in {total_time/60:.1f} minutes")
        
        # Save error report if there were failures
        if failed_rows:
            self.config_manager.save_error_report(failed_rows)
        
        logger.info(f"Successfully converted {len(documents)} out of {total_rows} rows to documents")
        
        # Generate and save summary statistics
        summary = self.statistics_generator.generate_conversion_summary(documents)
        self.config_manager.save_conversion_summary(summary)
        
        return documents
    
    def save_documents_as_jsonl(self, documents: List[SimpleDocument], filename: str = None) -> str:
        """Save all documents as a single JSONL file"""
        return self.file_output_manager.save_documents_as_jsonl(documents, filename)
    
    def save_documents_as_markdown_files(self, documents: List[SimpleDocument], prefix: str = "deed", batch_size: int = 1000) -> List[str]:
        """Save each document as an individual markdown file"""
        return self.file_output_manager.save_documents_as_markdown_files(documents, prefix, batch_size)
    
    def save_documents_to_database(self, documents: List[SimpleDocument], batch_size: int = 100) -> int:
        """Save documents to PostgreSQL database"""
        try:
            # Connect to the database
            if not self.db_manager.connect():
                logger.error("Failed to connect to the database")
                return 0
                
            # Insert documents with enhanced markdown content
            inserted_count = self.db_manager.insert_documents(
                documents, 
                batch_size=batch_size,
                output_manager=self.file_output_manager
            )
            
            # Close connection
            self.db_manager.close()
            
            return inserted_count
        except Exception as e:
            logger.error(f"Error saving documents to database: {e}")
            return 0
    
    def print_summary_statistics(self, documents: List[SimpleDocument]):
        """Print summary statistics to console"""
        if self.statistics_generator:
            self.statistics_generator.print_summary_statistics(documents)
    
    def analyze_csv_structure(self):
        """Analyze CSV structure and return analysis results"""
        return self.csv_analyzer.analyze_csv_structure(self.input_file_path) 