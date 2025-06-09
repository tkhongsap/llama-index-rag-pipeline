#!/usr/bin/env python3
"""Quick check for available packages."""

packages_to_check = [
    'llama_index',
    'transformers', 
    'sentence_transformers',
    'torch',
    'numpy',
    'pandas'
]

print("Checking package availability:")
for package in packages_to_check:
    try:
        __import__(package)
        print(f"  ✅ {package} - available")
    except ImportError as e:
        print(f"  ❌ {package} - not available ({e})")

print("\nPython path:")
import sys
for path in sys.path:
    print(f"  {path}")