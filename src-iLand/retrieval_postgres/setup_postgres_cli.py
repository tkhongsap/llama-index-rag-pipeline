#!/usr/bin/env python3
"""
Setup script for iLand PostgreSQL Retrieval CLI

This script helps set up the environment and dependencies needed for the 
PostgreSQL retrieval system.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is supported."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    dependencies = [
        "psycopg2-binary>=2.9.0",
        "asyncpg>=0.25.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
    ]
    
    print("📦 Installing PostgreSQL retrieval dependencies...")
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {dep.split('>=')[0]} installed")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {dep}")
            return False
    
    return True

def check_environment():
    """Check environment variables."""
    print("🔧 Checking environment variables...")
    
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", 
                    "POSTGRES_USER", "POSTGRES_PASSWORD"]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
        else:
            print(f"   ✅ {var} is set")
    
    if missing_required:
        print(f"   ⚠️  Missing required variables: {', '.join(missing_required)}")
        print("   Set with: export OPENAI_API_KEY='your-key-here'")
    
    missing_optional = []
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
        else:
            print(f"   ✅ {var} is set")
    
    if missing_optional:
        print(f"   ℹ️  Optional variables not set: {', '.join(missing_optional)}")
        print("   These will use default values")
    
    return len(missing_required) == 0

def create_test_config():
    """Create a test configuration file."""
    config_content = """# PostgreSQL Retrieval Test Configuration

# Database connection (adjust as needed)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=iland_rag
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password

# Required for LLM operations
export OPENAI_API_KEY=your_openai_key_here

# Optional: BGE embedding service URL
export BGE_EMBEDDING_URL=http://localhost:8080/embeddings
"""
    
    config_path = Path(__file__).parent / "test_config.env"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print(f"✅ Created test configuration at: {config_path}")
    print("   Edit this file with your actual values, then run:")
    print(f"   source {config_path}")

def test_cli():
    """Test the CLI installation."""
    print("🧪 Testing CLI...")
    
    try:
        # Test basic import
        from .cli import main
        print("   ✅ CLI imports successfully")
        
        # Test help command
        sys.argv = ["setup_postgres_cli.py", "--help"]
        main()
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except SystemExit:
        # Help command exits normally
        print("   ✅ CLI help works")
        return True
    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 iLand PostgreSQL Retrieval System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check environment
    env_ok = check_environment()
    
    # Create test config
    create_test_config()
    
    # Test CLI
    if test_cli():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit test_config.env with your database credentials")
        print("2. Run: source test_config.env")
        print("3. Test connection: python -m retrieval_postgres.cli --test-connection")
        print("4. Try a query: python -m retrieval_postgres.cli --query 'โฉนดที่ดินจังหวัดชัยนาท'")
    else:
        print("\n⚠️  Setup completed with warnings. CLI may need manual testing.")
    
    if not env_ok:
        print("\n⚠️  Remember to set required environment variables before using the CLI.")

if __name__ == "__main__":
    main()