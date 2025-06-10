#!/usr/bin/env python3
"""
Test Environment Manager for iLand PostgreSQL Testing

This script helps manage environment configurations and provides utilities
for testing both data_processing_postgres and docs_embedding_postgres modules.
"""

import os
import sys
import subprocess
import psycopg2
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv, set_key
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestEnvironmentManager:
    """Manages test environment configuration and database setup"""
    
    def __init__(self, config_file: str = "test_config.env"):
        self.config_file = config_file
        self.load_environment()
    
    def load_environment(self):
        """Load environment configuration"""
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file {self.config_file} not found. Creating default...")
            self.create_default_config()
        
        load_dotenv(self.config_file, override=True)
        self.environment = os.getenv("ENVIRONMENT", "local")
        logger.info(f"Loaded environment: {self.environment}")
    
    def create_default_config(self):
        """Create default configuration file"""
        default_config = """# iLand PostgreSQL Testing Configuration
ENVIRONMENT=local

# Local testing PostgreSQL
LOCAL_DB_NAME=iland_test_db
LOCAL_DB_USER=iland_test_user
LOCAL_DB_PASSWORD=iland_test_password
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=5433

# Production PostgreSQL (update with your actual values)
PROD_DB_NAME=iland-vector-dev
PROD_DB_USER=vector_user_dev
PROD_DB_PASSWORD=akqVvIJvVqe7Jr1
PROD_DB_HOST=10.4.102.11
PROD_DB_PORT=5432

# BGE Configuration
BGE_MODEL=bge-m3
BGE_CACHE_FOLDER=./cache/bge_models

# Processing Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
BATCH_SIZE=10
"""
        with open(self.config_file, 'w') as f:
            f.write(default_config)
        logger.info(f"Created default config: {self.config_file}")
    
    def switch_environment(self, environment: str):
        """Switch between 'local' and 'production' environments"""
        if environment not in ['local', 'production']:
            raise ValueError("Environment must be 'local' or 'production'")
        
        # Update config file
        set_key(self.config_file, "ENVIRONMENT", environment)
        
        # Update current environment variables
        if environment == "local":
            db_config = {
                "DB_NAME": os.getenv("LOCAL_DB_NAME", "iland_test_db"),
                "DB_USER": os.getenv("LOCAL_DB_USER", "iland_test_user"),
                "DB_PASSWORD": os.getenv("LOCAL_DB_PASSWORD", "iland_test_password"),
                "DB_HOST": os.getenv("LOCAL_DB_HOST", "localhost"),
                "DB_PORT": os.getenv("LOCAL_DB_PORT", "5433")
            }
        else:  # production
            db_config = {
                "DB_NAME": os.getenv("PROD_DB_NAME", "iland-vector-dev"),
                "DB_USER": os.getenv("PROD_DB_USER", "vector_user_dev"),
                "DB_PASSWORD": os.getenv("PROD_DB_PASSWORD", "akqVvIJvVqe7Jr1"),
                "DB_HOST": os.getenv("PROD_DB_HOST", "10.4.102.11"),
                "DB_PORT": os.getenv("PROD_DB_PORT", "5432")
            }
        
        # Update environment variables for current session
        for key, value in db_config.items():
            os.environ[key] = value
            set_key(self.config_file, key, value)
        
        self.environment = environment
        logger.info(f"Switched to {environment} environment")
        logger.info(f"Database: {db_config['DB_NAME']} at {db_config['DB_HOST']}:{db_config['DB_PORT']}")
    
    def start_local_postgres(self):
        """Start local PostgreSQL using Docker Compose"""
        if self.environment != "local":
            logger.warning("Not in local environment. Use switch_environment('local') first.")
            return False
        
        try:
            logger.info("Starting local PostgreSQL with Docker Compose...")
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.test.yml", "up", "-d"],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Docker containers started successfully")
            
            # Wait for PostgreSQL to be ready
            self.wait_for_database()
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker containers: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("Docker Compose not found. Please install Docker and Docker Compose.")
            return False
    
    def stop_local_postgres(self):
        """Stop local PostgreSQL Docker containers"""
        try:
            logger.info("Stopping local PostgreSQL containers...")
            subprocess.run(
                ["docker-compose", "-f", "docker-compose.test.yml", "down"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Docker containers stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop Docker containers: {e}")
            return False
    
    def wait_for_database(self, max_retries: int = 30, retry_interval: int = 2):
        """Wait for database to be ready"""
        logger.info("Waiting for database to be ready...")
        
        for attempt in range(max_retries):
            if self.test_database_connection():
                logger.info(f"Database ready after {attempt + 1} attempts")
                return True
            
            logger.info(f"Database not ready, attempt {attempt + 1}/{max_retries}")
            time.sleep(retry_interval)
        
        logger.error(f"Database not ready after {max_retries} attempts")
        return False
    
    def test_database_connection(self) -> bool:
        """Test database connection"""
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT")
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
            
            cursor.close()
            conn.close()
            
            logger.info(f"Database connection successful")
            logger.info(f"PostgreSQL version: {version[:50]}...")
            logger.info(f"PGVector extension: {vector_status}")
            return True
            
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current database configuration"""
        return {
            "environment": self.environment,
            "db_name": os.getenv("DB_NAME"),
            "db_user": os.getenv("DB_USER"),
            "db_host": os.getenv("DB_HOST"),
            "db_port": os.getenv("DB_PORT"),
            "bge_model": os.getenv("BGE_MODEL", "bge-m3"),
            "cache_folder": os.getenv("BGE_CACHE_FOLDER", "./cache/bge_models")
        }
    
    def create_test_directories(self):
        """Create necessary test directories"""
        directories = [
            "test_output",
            "logs", 
            os.getenv("BGE_CACHE_FOLDER", "./cache/bge_models"),
            "test_data"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def check_test_requirements(self) -> bool:
        """Check if all requirements for testing are met"""
        requirements = []
        
        # Check Python packages
        try:
            import psycopg2
            requirements.append("✅ psycopg2")
        except ImportError:
            requirements.append("❌ psycopg2 (install: pip install psycopg2-binary)")
        
        try:
            import llama_index
            requirements.append("✅ llama_index")
        except ImportError:
            requirements.append("❌ llama_index (install: pip install llama-index)")
        
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            requirements.append("✅ BGE embeddings")
        except ImportError:
            requirements.append("❌ BGE embeddings (install: pip install llama-index-embeddings-huggingface sentence-transformers)")
        
        try:
            from llama_index.vector_stores.postgres import PGVectorStore
            requirements.append("✅ PGVector store")
        except ImportError:
            requirements.append("❌ PGVector store (install: pip install llama-index-vector-stores-postgres)")
        
        # Check Docker (for local testing)
        if self.environment == "local":
            try:
                result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    requirements.append("✅ Docker")
                else:
                    requirements.append("❌ Docker (install Docker Desktop)")
            except FileNotFoundError:
                requirements.append("❌ Docker (install Docker Desktop)")
        
        # Check database connection
        if self.test_database_connection():
            requirements.append("✅ Database connection")
        else:
            requirements.append("❌ Database connection")
        
        logger.info("Requirements check:")
        for req in requirements:
            logger.info(f"  {req}")
        
        return all("✅" in req for req in requirements)


def main():
    """CLI for environment management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="iLand PostgreSQL Test Environment Manager")
    parser.add_argument("action", choices=[
        "switch-local", "switch-production", "start-local", "stop-local", 
        "test-connection", "check-requirements", "show-config"
    ])
    
    args = parser.parse_args()
    
    manager = TestEnvironmentManager()
    
    if args.action == "switch-local":
        manager.switch_environment("local")
    elif args.action == "switch-production":
        manager.switch_environment("production")
    elif args.action == "start-local":
        manager.create_test_directories()
        if manager.start_local_postgres():
            print("✅ Local PostgreSQL started successfully")
            print("🌐 pgAdmin available at: http://localhost:8080")
            print("   Email: admin@iland.test")
            print("   Password: admin_password")
        else:
            print("❌ Failed to start local PostgreSQL")
            sys.exit(1)
    elif args.action == "stop-local":
        manager.stop_local_postgres()
    elif args.action == "test-connection":
        if manager.test_database_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            sys.exit(1)
    elif args.action == "check-requirements":
        if manager.check_test_requirements():
            print("✅ All requirements met")
        else:
            print("❌ Some requirements not met")
            sys.exit(1)
    elif args.action == "show-config":
        config = manager.get_current_config()
        print("Current Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 