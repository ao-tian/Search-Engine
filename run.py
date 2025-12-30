#!/usr/bin/env python3
"""
Main entry point for the Search Engine application.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from search_engine.app import app, initialize_engine, DOCUMENTS_DIR
import os
import socket

if __name__ == '__main__':
    # Create documents directory if it doesn't exist
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    # Initialize on startup
    initialize_engine()
    
    # Try different ports if 5000 is in use
    port = 5000
    for p in range(5000, 5010):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', p))
        sock.close()
        if result != 0:
            port = p
            break
    
    print("\n" + "="*60)
    print("Search Engine Web Interface")
    print("="*60)
    from search_engine.app import documents
    print(f"Indexed {len(documents) if documents else 0} documents")
    
    # Check for optional dependencies
    from search_engine.app import PDF_SUPPORT, DOCX_SUPPORT
    if not PDF_SUPPORT:
        print("⚠️  PDF support not available. Install: pip install PyPDF2")
    if not DOCX_SUPPORT:
        print("⚠️  DOCX support not available. Install: pip install python-docx")
    
    print(f"Open http://localhost:{port} in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)

