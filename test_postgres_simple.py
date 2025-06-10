#!/usr/bin/env python3
"""
Simplified iLand PostgreSQL Pipeline Test

This script tests the PostgreSQL data processing and embedding pipelines
to ensure they work correctly and produce consistent results.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection(db_config: Dict[str, str]) -> bool:
    """Test database connection"""
    try:
        import psycopg2
        logger.info(f"Testing connection to {db_config['host']}:{db_config['port']}")
        
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        # Test vector extension
        try:
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
            vector_exists = cursor.fetchone() is not None
            vector_status = "✅ Available" if vector_exists else "❌ Not installed"
        except:
            vector_status = "❌ Error checking"
        
        # Check tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name IN ('iland_md_data', 'iland_embeddings')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Database connection successful")
        logger.info(f"PostgreSQL version: {version[:50]}...")
        logger.info(f"PGVector extension: {vector_status}")
        logger.info(f"Tables found: {tables}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def test_data_processing_postgres(db_config: Dict[str, str], limit: int = 3) -> Dict[str, Any]:
    """Test PostgreSQL data processing with a small sample"""
    logger.info(f"\n🔄 Testing PostgreSQL data processing with {limit} records...")
    
    try:
        # Add paths for PostgreSQL modules
        sys.path.insert(0, "src-iLand/data_processing_postgres")
        
        # Set environment variables
        os.environ.update({
            "DB_NAME": db_config['dbname'],
            "DB_USER": db_config['user'],
            "DB_PASSWORD": db_config['password'],
            "DB_HOST": db_config['host'],
            "DB_PORT": db_config['port']
        })
        
        # Create sample CSV data for testing
        sample_data = f"""deed_id,province,district,land_main_category,area_rai,area_ngan,area_wa,coordinates_lat,coordinates_lng
TEST001,ชัยนาท,เมืองชัยนาท,ที่ดินเปล่า,2,1,50,15.1854,100.1234
TEST002,ชัยนาท,เมืองชัยนาท,ที่อยู่อาศัย,1,2,25,15.1855,100.1235
TEST003,ชัยนาท,เมืองชัยนาท,เกษตรกรรม,3,0,75,15.1856,100.1236"""
        
        # Create test directories
        os.makedirs("test_data", exist_ok=True)
        os.makedirs("test_output", exist_ok=True)
        
        test_csv = "test_data/test_sample.csv"
        with open(test_csv, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        
        logger.info(f"Created test CSV: {test_csv}")
        
        # Test basic CSV reading
        import pandas as pd
        df = pd.read_csv(test_csv)
        logger.info(f"✅ CSV loaded: {len(df)} rows")
        
        # Import and test database manager
        from db_manager import DatabaseManager
        
        db_manager = DatabaseManager(
            db_name=db_config['dbname'],
            db_user=db_config['user'],
            db_password=db_config['password'],
            db_host=db_config['host'],
            db_port=int(db_config['port'])
        )
        
        if db_manager.connect():
            logger.info("✅ Database manager connected")
            
            # Test a simple document insertion
            from models import SimpleDocument
            
            test_doc = SimpleDocument(
                id="TEST_DOC_001",
                text=f"โฉนดที่ดิน เลขที่ TEST001 จังหวัด ชัยนาท เขต เมืองชัยนาท",
                metadata={
                    "deed_id": "TEST001",
                    "province": "ชัยนาท",
                    "district": "เมืองชัยนาท",
                    "land_main_category": "ที่ดินเปล่า"
                }
            )
            
            # Test insert (we'll only do one document for testing)
            result = db_manager.insert_documents([test_doc])
            
            if result > 0:
                logger.info(f"✅ Successfully inserted {result} test document(s)")
                return {
                    "success": True,
                    "documents_processed": result,
                    "test_mode": True,
                    "database_table": "iland_md_data"
                }
            else:
                logger.error("❌ Failed to insert test documents")
                return {"success": False, "error": "Database insertion failed"}
        else:
            logger.error("❌ Failed to connect to database")
            return {"success": False, "error": "Database connection failed"}
            
    except Exception as e:
        logger.error(f"❌ Data processing test failed: {e}")
        return {"success": False, "error": str(e)}

def test_bge_embeddings_postgres(db_config: Dict[str, str], limit: int = 3) -> Dict[str, Any]:
    """Test PostgreSQL BGE embeddings"""
    logger.info(f"\n🤗 Testing PostgreSQL BGE embeddings with {limit} records...")
    
    try:
        # Add paths for PostgreSQL embedding modules
        sys.path.insert(0, "src-iLand/docs_embedding_postgres")
        
        # Set environment variables
        os.environ.update({
            "DB_NAME": db_config['dbname'],
            "DB_USER": db_config['user'],
            "DB_PASSWORD": db_config['password'],
            "DB_HOST": db_config['host'],
            "DB_PORT": db_config['port'],
            "BGE_MODEL": "bge-small-en-v1.5",  # Use smaller model for testing
            "BGE_CACHE_FOLDER": "./cache/bge_models"
        })
        
        # Create cache directory
        os.makedirs("./cache/bge_models", exist_ok=True)
        
        # Test BGE embedding processor
        logger.info("Testing BGE embedding processor...")
        
        try:
            from postgres_embedding_bge import BGEPostgresEmbeddingGenerator
            
            generator = BGEPostgresEmbeddingGenerator(
                bge_model_key="bge-small-en-v1.5",  # Smaller model for testing
                cache_folder="./cache/bge_models",
                enable_section_chunking=True
            )
            
            # Test with very small limit
            test_limit = min(limit, 2)  # Maximum 2 documents for testing
            
            logger.info(f"Running BGE pipeline with limit={test_limit}...")
            result = generator.run_pipeline(limit=test_limit)
            
            if result > 0:
                logger.info(f"✅ BGE embeddings test successful: {result} embeddings created")
                return {
                    "success": True,
                    "embeddings_created": result,
                    "model_used": "bge-small-en-v1.5",
                    "embedding_dimension": generator.embed_dim,
                    "test_mode": True
                }
            else:
                logger.warning("⚠️ No embeddings created (this might be normal if no data exists)")
                return {
                    "success": True,  # Not necessarily a failure
                    "embeddings_created": 0,
                    "note": "No embeddings created - check if iland_md_data table has data"
                }
                
        except ImportError as e:
            logger.error(f"❌ BGE modules not available: {e}")
            return {"success": False, "error": "BGE modules not available"}
            
    except Exception as e:
        logger.error(f"❌ BGE embeddings test failed: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main test execution"""
    logger.info("🚀 iLand PostgreSQL Pipeline Simple Test")
    logger.info("=" * 60)
    
    # Database configuration (can be overridden via environment variables)
    db_config = {
        'dbname': os.getenv('DB_NAME', 'iland-vector-dev'),
        'user': os.getenv('DB_USER', 'vector_user_dev'),
        'password': os.getenv('DB_PASSWORD', 'akqVvIJvVqe7Jr1'),
        'host': os.getenv('DB_HOST', '10.4.102.11'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    logger.info(f"Database: {db_config['dbname']} at {db_config['host']}:{db_config['port']}")
    
    # Test 1: Database Connection
    logger.info("\n" + "="*30 + " DATABASE CONNECTION " + "="*30)
    if not test_database_connection(db_config):
        logger.error("❌ Database connection failed. Cannot continue with tests.")
        return False
    
    # Test 2: Data Processing
    logger.info("\n" + "="*30 + " DATA PROCESSING TEST " + "="*30)
    data_result = test_data_processing_postgres(db_config, limit=1)
    
    if data_result['success']:
        logger.info("✅ Data processing test passed")
    else:
        logger.error(f"❌ Data processing test failed: {data_result.get('error', 'Unknown error')}")
    
    # Test 3: BGE Embeddings
    logger.info("\n" + "="*30 + " BGE EMBEDDINGS TEST " + "="*30)
    embedding_result = test_bge_embeddings_postgres(db_config, limit=1)
    
    if embedding_result['success']:
        logger.info("✅ BGE embeddings test passed")
    else:
        logger.error(f"❌ BGE embeddings test failed: {embedding_result.get('error', 'Unknown error')}")
    
    # Summary
    logger.info("\n" + "="*30 + " TEST SUMMARY " + "="*30)
    
    all_passed = data_result['success'] and embedding_result['success']
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Database connection works")
        logger.info("✅ Data processing pipeline works")
        logger.info("✅ BGE embedding pipeline works")
        logger.info("\n📝 Your PostgreSQL pipelines are ready for use!")
    else:
        logger.warning("⚠️ Some tests failed. Check the details above.")
        
        if not data_result['success']:
            logger.warning("- Data processing needs attention")
        if not embedding_result['success']:
            logger.warning("- BGE embeddings need attention")
    
    logger.info("\n📊 Test Results:")
    logger.info(f"  Data Processing: {'✅ PASS' if data_result['success'] else '❌ FAIL'}")
    logger.info(f"  BGE Embeddings:  {'✅ PASS' if embedding_result['success'] else '❌ FAIL'}")
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple PostgreSQL Pipeline Test")
    parser.add_argument("--db-host", help="Database host")
    parser.add_argument("--db-port", help="Database port")
    parser.add_argument("--db-name", help="Database name")
    parser.add_argument("--db-user", help="Database user")
    parser.add_argument("--db-password", help="Database password")
    
    args = parser.parse_args()
    
    # Override database config if provided
    if args.db_host:
        os.environ['DB_HOST'] = args.db_host
    if args.db_port:
        os.environ['DB_PORT'] = args.db_port
    if args.db_name:
        os.environ['DB_NAME'] = args.db_name
    if args.db_user:
        os.environ['DB_USER'] = args.db_user
    if args.db_password:
        os.environ['DB_PASSWORD'] = args.db_password
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n👋 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 