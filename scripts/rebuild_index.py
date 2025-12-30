#!/usr/bin/env python3
"""
Script to rebuild the search index.
Run this if search is not working or index is out of sync.
"""

import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from search_engine.app import initialize_engine, INDEX_FILE

if __name__ == '__main__':
    print("Rebuilding search index...")
    
    # Delete old index if it exists
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        print(f"Deleted old index file: {INDEX_FILE}")
    
    # Force rebuild
    initialize_engine(force_rebuild=True)
    
    print("\nIndex rebuilt successfully!")
    print("You can now restart the web server.")

