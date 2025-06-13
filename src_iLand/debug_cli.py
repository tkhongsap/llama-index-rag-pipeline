#!/usr/bin/env python
"""
Debug script to test CLI functionality step by step
"""

import sys
from pathlib import Path

# Add the current directory to the path to import retrieval modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test all imports"""
    try:
        print("Testing imports...")
        from retrieval.cli_utils import print_warning, print_error, print_success
        print("✅ cli_utils imported successfully")
        
        from retrieval.cli_operations import iLandCLIOperations
        print("✅ cli_operations imported successfully")
        
        from retrieval.cli_handlers import iLandRetrievalCLI
        print("✅ cli_handlers imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cli_creation():
    """Test CLI creation"""
    try:
        print("\nTesting CLI creation...")
        from retrieval.cli_handlers import iLandRetrievalCLI
        cli = iLandRetrievalCLI()
        print("✅ CLI created successfully")
        return cli
    except Exception as e:
        print(f"❌ CLI creation failed: {e}")
        return None

def test_simple_query():
    """Test a simple query without loading embeddings"""
    try:
        print("\nTesting simple query (should fail gracefully)...")
        from retrieval.cli_handlers import iLandRetrievalCLI
        cli = iLandRetrievalCLI()
        
        # This should fail gracefully
        results = cli.query("test query")
        print(f"✅ Query executed (returned {len(results)} results)")
        return True
    except Exception as e:
        print(f"❌ Simple query failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == '__main__':
    print("🔍 CLI Debugging")
    print("=" * 30)
    
    # Test 1: Imports
    if not test_imports():
        sys.exit(1)
    
    # Test 2: CLI Creation
    cli = test_cli_creation()
    if not cli:
        sys.exit(1)
    
    # Test 3: Simple Query
    if not test_simple_query():
        sys.exit(1)
    
    print("\n✅ All basic tests passed!")
    print("The CLI should work correctly now.") 