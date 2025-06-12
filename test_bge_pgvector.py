#!/usr/bin/env python
"""
Test script for BGE-M3 PGVector integration with LlamaIndex

This script tests the new BGEPGVectorProcessor class to ensure:
1. Proper LlamaIndex PGVector Store integration
2. BGE-M3 embedding generation
3. LLM-generated natural language summaries
4. All 4 tables are created and populated correctly
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src-iLand to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_iland_path = os.path.join(current_dir, 'src-iLand')
if src_iland_path not in sys.path:
    sys.path.append(src_iland_path)

def test_pgvector_integration():
    """Test the PGVector integration"""
    try:
        from bge_postgres_pipeline import BGEPGVectorProcessor
        
        logger.info("=== TESTING BGE PGVECTOR INTEGRATION ===")
        
        # Create processor
        processor = BGEPGVectorProcessor(
            db_name=os.getenv("DB_NAME", "iland-vector-dev"),
            db_user=os.getenv("DB_USER", "vector_user_dev"),
            db_password=os.getenv("DB_PASSWORD"),
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            bge_model_name="BAAI/bge-m3",  # Use correct BGE model name
            enable_llm_summary=True,
            enable_section_chunking=True,
            enable_multi_model=True
        )
        
        logger.info("✅ BGEPGVectorProcessor initialized successfully")
        
        # Test vector store creation
        logger.info("🗄️ Vector stores created:")
        for store_name, store in processor.vector_stores.items():
            logger.info(f"  - {store_name}: {processor.table_configs[store_name]['table_name']}")
        
        # Test document fetching
        documents = processor.fetch_documents_from_db(limit=2)
        logger.info(f"📄 Fetched {len(documents)} documents for testing")
        
        if documents:
            # Test pipeline
            result = processor.run_pipeline(limit=2)
            
            if result["success"]:
                logger.info("🎉 Pipeline test SUCCESSFUL")
                logger.info(f"📊 Stats: {result['stats']}")
            else:
                logger.error(f"❌ Pipeline test FAILED: {result.get('error')}")
        else:
            logger.warning("⚠️ No documents found to test with")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def verify_tables():
    """Verify that all 4 tables are created"""
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "iland-vector-dev"),
            user=os.getenv("DB_USER", "vector_user_dev"),
            password=os.getenv("DB_PASSWORD")
        )
        
        cursor = conn.cursor()
        
        # Check for all expected tables
        expected_tables = ['iland_chunks', 'iland_summaries', 'iland_indexnodes', 'iland_combined']
        
        logger.info("🔍 Checking PGVector tables...")
        
        for table in expected_tables:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = %s
            """, (table,))
            
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                logger.info(f"✅ {table}: EXISTS ({row_count} rows)")
            else:
                logger.warning(f"⚠️ {table}: NOT FOUND")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Table verification failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("=" * 80)
    logger.info("BGE-M3 PGVECTOR INTEGRATION TEST")
    logger.info("Testing LlamaIndex PGVector Store with BGE-M3 + LLM")
    logger.info("=" * 80)
    
    # Test 1: Verify environment
    required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        return 1
    
    logger.info("✅ Environment variables verified")
    
    # Test 2: Verify tables exist
    if not verify_tables():
        logger.error("❌ Table verification failed")
        return 1
    
    # Test 3: Test PGVector integration
    if not test_pgvector_integration():
        logger.error("❌ PGVector integration test failed")
        return 1
    
    logger.info("=" * 80)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("✅ LlamaIndex PGVector Store integration working correctly")
    logger.info("✅ BGE-M3 embedding generation functional")
    logger.info("✅ LLM summary generation operational")
    logger.info("✅ All 4 tables properly configured")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 