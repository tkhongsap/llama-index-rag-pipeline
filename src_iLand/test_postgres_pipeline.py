#!/usr/bin/env python3
"""
Test script for the complete PostgreSQL pipeline (data processing + embedding)

This script tests the end-to-end pipeline:
1. Data processing: CSV -> Enhanced PostgreSQL with metadata
2. Embedding generation: PostgreSQL docs -> Vector embeddings in PostgreSQL

Usage:
    python test_postgres_pipeline.py --test-data-processing
    python test_postgres_pipeline.py --test-embedding
    python test_postgres_pipeline.py --test-full-pipeline
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgresPipelineTester:
    """Test the complete PostgreSQL pipeline."""
    
    def __init__(self):
        self.project_root = current_dir.parent  # Go up one level from src-iLand to project root
        self.test_stats = {
            "data_processing": {"success": False, "documents_processed": 0, "error": None},
            "embedding": {"success": False, "embeddings_generated": 0, "error": None},
            "full_pipeline": {"success": False, "total_time": 0, "error": None}
        }
    
    def test_data_processing(self, max_rows: int = 100) -> Dict[str, Any]:
        """Test the data processing pipeline."""
        logger.info("Testing data processing pipeline...")
        
        try:
            # Import data processing module
            from data_processing_postgres.main import main as data_main
            from data_processing_postgres.iland_converter import iLandCSVConverter
            
            # Setup paths
            input_dir = self.project_root / "data" / "input_docs"
            output_dir = self.project_root / "data" / "output_docs"
            
            # Look for test data file
            test_files = [
                # "input_dataset_iLand.xlsx",
                # "input_dataset_iLand.csv",
                "test_data.xlsx",
                # "test_data.csv"
            ]
            
            input_file = None
            for filename in test_files:
                test_path = input_dir / filename
                if test_path.exists():
                    input_file = test_path
                    break
            
            if not input_file:
                raise FileNotFoundError(f"No test data file found in {input_dir}. Tried: {test_files}")
            
            logger.info(f"Using test data file: {input_file}")
            
            # Create converter
            converter = iLandCSVConverter(str(input_file), str(output_dir))
            
            # Setup configuration
            config = converter.setup_configuration(config_name="test_postgres_pipeline", auto_generate=True)
            
            # Process documents (limited for testing)
            logger.info(f"Processing up to {max_rows} rows for testing...")
            documents = converter.process_csv_to_documents(batch_size=50, max_rows=max_rows)
            
            if not documents:
                raise Exception("No documents were generated from the data file")
            
            # Save to database
            logger.info("Saving documents to PostgreSQL database...")
            inserted_count = converter.save_documents_to_database(documents, batch_size=25)
            
            if inserted_count == 0:
                raise Exception("Failed to insert any documents into database")
            
            # Print summary
            converter.print_summary_statistics(documents)
            
            # Update test stats
            self.test_stats["data_processing"] = {
                "success": True,
                "documents_processed": len(documents),
                "documents_inserted": inserted_count,
                "config_name": config.name,
                "error": None
            }
            
            logger.info(f"Data processing test PASSED: {len(documents)} documents processed, {inserted_count} inserted")
            return self.test_stats["data_processing"]
            
        except Exception as e:
            error_msg = f"Data processing test FAILED: {str(e)}"
            logger.error(error_msg)
            self.test_stats["data_processing"]["error"] = str(e)
            return self.test_stats["data_processing"]
    
    def test_embedding_pipeline(self, limit: int = 10) -> Dict[str, Any]:
        """Test the embedding generation pipeline."""
        logger.info("Testing embedding generation pipeline...")
        
        try:
            # Import embedding module with proper path handling
            try:
                from docs_embedding_postgres.enhanced_postgres_embedding import EnhancedPostgresEmbeddingPipeline
            except ImportError:
                # Try relative import from src-iLand
                from .docs_embedding_postgres.enhanced_postgres_embedding import EnhancedPostgresEmbeddingPipeline
            
            # Reset embedding status to pending for testing
            self._reset_embedding_status_for_testing()
            
            # Create pipeline with test configuration
            pipeline = EnhancedPostgresEmbeddingPipeline(
                chunk_size=256,  # Smaller chunks for testing
                chunk_overlap=25,
                batch_size=5,    # Smaller batches for testing
                enable_section_chunking=True,
                enable_multi_model=True  # Test multi-model if available
            )
            
            # Run pipeline with limited documents
            logger.info(f"Running embedding pipeline with limit of {limit} documents...")
            # Try pending first, if no results try completed for testing
            result = pipeline.run_pipeline(limit=limit, status_filter="pending")
            if not result.get("success") and "No documents found" in result.get("message", ""):
                logger.info("No pending documents found, trying with completed status for testing...")
                result = pipeline.run_pipeline(limit=limit, status_filter="completed")
            
            if not result["success"]:
                raise Exception(f"Embedding pipeline failed: {result.get('error', 'Unknown error')}")
            
            # Update test stats
            self.test_stats["embedding"] = {
                "success": True,
                "embeddings_generated": result["stats"]["embeddings_generated"],
                "documents_processed": result["stats"]["documents_processed"],
                "nodes_created": result["stats"]["nodes_created"],
                "db_insertions": result["stats"]["db_insertions"],
                "duration": result["duration"],
                "error": None
            }
            
            logger.info(f"Embedding test PASSED: {result['stats']['embeddings_generated']} embeddings generated")
            return self.test_stats["embedding"]
            
        except Exception as e:
            error_msg = f"Embedding test FAILED: {str(e)}"
            logger.error(error_msg)
            self.test_stats["embedding"]["error"] = str(e)
            return self.test_stats["embedding"]
    
    def test_full_pipeline(self, max_rows: int = 50, embedding_limit: int = 10) -> Dict[str, Any]:
        """Test the complete end-to-end pipeline."""
        logger.info("Testing complete end-to-end PostgreSQL pipeline...")
        
        start_time = time.time()
        
        try:
            # Step 1: Test data processing
            logger.info("Step 1: Testing data processing...")
            data_result = self.test_data_processing(max_rows=max_rows)
            
            if not data_result["success"]:
                raise Exception(f"Data processing failed: {data_result['error']}")
            
            # Step 2: Test embedding generation
            logger.info("Step 2: Testing embedding generation...")
            embedding_result = self.test_embedding_pipeline(limit=embedding_limit)
            
            if not embedding_result["success"]:
                raise Exception(f"Embedding generation failed: {embedding_result['error']}")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Update test stats
            self.test_stats["full_pipeline"] = {
                "success": True,
                "total_time": total_time,
                "data_processing_docs": data_result["documents_processed"],
                "embedding_docs": embedding_result["documents_processed"],
                "total_embeddings": embedding_result["embeddings_generated"],
                "error": None
            }
            
            logger.info(f"Full pipeline test PASSED in {total_time:.2f} seconds")
            return self.test_stats["full_pipeline"]
            
        except Exception as e:
            error_msg = f"Full pipeline test FAILED: {str(e)}"
            logger.error(error_msg)
            self.test_stats["full_pipeline"]["error"] = str(e)
            return self.test_stats["full_pipeline"]
    
    def verify_database_content(self) -> Dict[str, Any]:
        """Verify that data was properly stored in the database."""
        logger.info("Verifying database content...")
        
        conn = None
        cursor = None
        try:
            import psycopg2
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            # Connect to database with autocommit to avoid transaction issues
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME", "iland-vector-dev"),
                user=os.getenv("DB_USER", "vector_user_dev"),
                password=os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
                host=os.getenv("DB_HOST", "10.4.102.11"),
                port=int(os.getenv("DB_PORT", "5432"))
            )
            conn.autocommit = True  # Enable autocommit to avoid transaction errors
            cursor = conn.cursor()
            
            verification_results = {}
            
            # Check source data table
            try:
                cursor.execute("SELECT COUNT(*) FROM iland_md_data")
                source_count = cursor.fetchone()[0]
                verification_results["source_documents"] = source_count
            except Exception as e:
                verification_results["source_documents"] = f"Error: {e}"
            
            # Check embedding tables
            tables = ["iland_chunks", "iland_summaries", "iland_indexnodes", "iland_combined"]
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    verification_results[table] = count
                except Exception as e:
                    verification_results[table] = f"Error: {e}"
            
            # Check embedding status
            try:
                cursor.execute("SELECT embedding_status, COUNT(*) FROM iland_md_data GROUP BY embedding_status")
                status_counts = dict(cursor.fetchall())
                verification_results["embedding_status"] = status_counts
            except Exception as e:
                verification_results["embedding_status"] = f"Error: {e}"
            
            logger.info("Database verification completed:")
            for key, value in verification_results.items():
                logger.info(f"  {key}: {value}")
            
            return {"success": True, "results": verification_results}
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Ensure cleanup
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def print_test_summary(self):
        """Print a comprehensive test summary."""
        logger.info("=" * 60)
        logger.info("POSTGRESQL PIPELINE TEST SUMMARY")
        logger.info("=" * 60)
        
        # Data processing results
        data_result = self.test_stats["data_processing"]
        status = "✅ PASSED" if data_result["success"] else "❌ FAILED"
        logger.info(f"Data Processing: {status}")
        if data_result["success"]:
            logger.info(f"  Documents processed: {data_result['documents_processed']}")
            logger.info(f"  Documents inserted: {data_result.get('documents_inserted', 'N/A')}")
        else:
            logger.info(f"  Error: {data_result['error']}")
        
        # Embedding results
        embedding_result = self.test_stats["embedding"]
        status = "✅ PASSED" if embedding_result["success"] else "❌ FAILED"
        logger.info(f"Embedding Generation: {status}")
        if embedding_result["success"]:
            logger.info(f"  Documents processed: {embedding_result['documents_processed']}")
            logger.info(f"  Embeddings generated: {embedding_result['embeddings_generated']}")
            logger.info(f"  Database insertions: {embedding_result['db_insertions']}")
            logger.info(f"  Duration: {embedding_result['duration']:.2f}s")
        else:
            logger.info(f"  Error: {embedding_result['error']}")
        
        # Full pipeline results
        full_result = self.test_stats["full_pipeline"]
        status = "✅ PASSED" if full_result["success"] else "❌ FAILED"
        logger.info(f"Full Pipeline: {status}")
        if full_result["success"]:
            logger.info(f"  Total time: {full_result['total_time']:.2f}s")
            logger.info(f"  End-to-end success: Documents -> PostgreSQL -> Embeddings")
        else:
            logger.info(f"  Error: {full_result['error']}")
        
        logger.info("=" * 60)
        
        # Overall status
        all_passed = all(result["success"] for result in self.test_stats.values() if "success" in result)
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        logger.info(f"OVERALL STATUS: {overall_status}")
        logger.info("=" * 60)
    
    def _reset_embedding_status_for_testing(self):
        """Reset embedding status to pending for testing purposes."""
        try:
            import psycopg2
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME", "iland-vector-dev"),
                user=os.getenv("DB_USER", "vector_user_dev"),
                password=os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
                host=os.getenv("DB_HOST", "10.4.102.11"),
                port=int(os.getenv("DB_PORT", "5432"))
            )
            
            cursor = conn.cursor()
            cursor.execute("UPDATE iland_md_data SET embedding_status = 'pending' WHERE embedding_status = 'completed'")
            updated_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            
            if updated_count > 0:
                logger.info(f"Reset {updated_count} documents from 'completed' to 'pending' status for testing")
                
        except Exception as e:
            logger.warning(f"Could not reset embedding status: {e}")


def main():
    """Main test function with command-line interface."""
    parser = argparse.ArgumentParser(description='Test PostgreSQL Pipeline')
    
    parser.add_argument('--test-data-processing', action='store_true',
                        help='Test only data processing pipeline')
    parser.add_argument('--test-embedding', action='store_true',
                        help='Test only embedding generation pipeline')
    parser.add_argument('--test-full-pipeline', action='store_true',
                        help='Test complete end-to-end pipeline')
    parser.add_argument('--verify-database', action='store_true',
                        help='Verify database content')
    
    parser.add_argument('--max-rows', type=int, default=50,
                        help='Maximum rows for data processing test (default: 50)')
    parser.add_argument('--embedding-limit', type=int, default=10,
                        help='Document limit for embedding test (default: 10)')
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all tests
    if not any([args.test_data_processing, args.test_embedding, args.test_full_pipeline, args.verify_database]):
        args.test_full_pipeline = True
        args.verify_database = True
    
    tester = PostgresPipelineTester()
    
    logger.info("PostgreSQL Pipeline Tester Started")
    logger.info("=" * 50)
    
    try:
        # Run requested tests
        if args.test_data_processing:
            tester.test_data_processing(max_rows=args.max_rows)
        
        if args.test_embedding:
            tester.test_embedding_pipeline(limit=args.embedding_limit)
        
        if args.test_full_pipeline:
            tester.test_full_pipeline(max_rows=args.max_rows, embedding_limit=args.embedding_limit)
        
        if args.verify_database:
            tester.verify_database_content()
        
        # Print summary
        tester.print_test_summary()
        
        # Return appropriate exit code
        all_passed = all(result["success"] for result in tester.test_stats.values() if "success" in result)
        return 0 if all_passed else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())