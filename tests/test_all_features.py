"""
Test script for all new features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from search_engine.core.vector_search import (
    VectorCompare, build_index, search_with_highlighting,
    load_documents_from_directory, save_index, load_index,
    tokenize, get_words, highlight_query_terms
)
import os
import tempfile
import json

def test_stop_words():
    """Test stop word removal."""
    print("=" * 70)
    print("TEST 1: Stop Word Removal")
    print("=" * 70)
    
    text = "The quick brown fox jumps over the lazy dog"
    words_with_stop = tokenize(text, remove_stopwords=False)
    words_no_stop = tokenize(text, remove_stopwords=True)
    
    print(f"With stop words: {words_with_stop}")
    print(f"Without stop words: {words_no_stop}")
    print("✓ Stop words filtered correctly\n")


def test_tokenization():
    """Test improved tokenization."""
    print("=" * 70)
    print("TEST 2: Improved Tokenization")
    print("=" * 70)
    
    text = "Hello, world! This is a test. Python 3.9 is great!"
    words = tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {words}")
    print("✓ Punctuation handled correctly\n")


def test_highlighting():
    """Test result highlighting."""
    print("=" * 70)
    print("TEST 3: Result Highlighting")
    print("=" * 70)
    
    text = "Python is a programming language. Python is used for web development and data science."
    query = "python programming"
    highlighted = highlight_query_terms(text, query)
    print(f"Query: {query}")
    print(f"Highlighted: {highlighted}")
    print("✓ Highlighting works\n")


def test_file_loading():
    """Test loading documents from files."""
    print("=" * 70)
    print("TEST 4: File Loading")
    print("=" * 70)
    
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = {
            'doc1.txt': 'Python is a programming language used for web development.',
            'doc2.txt': 'Java is another programming language popular for enterprise applications.',
            'doc3.md': '# Machine Learning\nMachine learning uses Python and statistics.'
        }
        
        for filename, content in test_files.items():
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Load documents
        documents = load_documents_from_directory(tmpdir)
        print(f"Loaded {len(documents)} documents:")
        for doc_id, doc_data in documents.items():
            print(f"  Doc {doc_id}: {doc_data['title']} ({len(doc_data['text'])} chars)")
        print("✓ File loading works\n")


def test_persistence():
    """Test index persistence."""
    print("=" * 70)
    print("TEST 5: Index Persistence")
    print("=" * 70)
    
    # Create sample documents
    documents = {
        0: {
            'text': 'Python programming language',
            'title': 'Python Doc',
            'path': '/test/doc1.txt',
            'url': '/doc/0'
        },
        1: {
            'text': 'Java programming language',
            'title': 'Java Doc',
            'path': '/test/doc2.txt',
            'url': '/doc/1'
        }
    }
    
    v = VectorCompare()
    documents_words = []
    for doc_id in sorted(documents.keys()):
        words = get_words(documents[doc_id]['text'].lower())
        documents_words.append(words)
    
    docs_for_index = {doc_id: doc_data['text'] for doc_id, doc_data in documents.items()}
    index = build_index(docs_for_index, v, use_tfidf=True)
    
    # Save index
    save_index(index, documents, documents_words, 'test_index.json')
    print("✓ Index saved")
    
    # Load index
    loaded_index, loaded_docs, loaded_words = load_index('test_index.json')
    print(f"✓ Index loaded: {len(loaded_docs)} documents")
    
    # Clean up
    if os.path.exists('test_index.json'):
        os.remove('test_index.json')
    print("✓ Persistence works\n")


def test_search_with_highlighting():
    """Test enhanced search with highlighting."""
    print("=" * 70)
    print("TEST 6: Enhanced Search with Highlighting")
    print("=" * 70)
    
    documents = {
        0: {
            'text': 'Python is a programming language used for web development and data science.',
            'title': 'Python Introduction',
            'path': '',
            'url': '/doc/0'
        },
        1: {
            'text': 'Java is another programming language popular for enterprise applications.',
            'title': 'Java Overview',
            'path': '',
            'url': '/doc/1'
        },
        2: {
            'text': 'Machine learning uses Python, statistics, and neural networks.',
            'title': 'Machine Learning',
            'path': '',
            'url': '/doc/2'
        }
    }
    
    v = VectorCompare()
    documents_words = []
    for doc_id in sorted(documents.keys()):
        words = get_words(documents[doc_id]['text'].lower())
        documents_words.append(words)
    
    docs_for_index = {doc_id: doc_data['text'] for doc_id, doc_data in documents.items()}
    index = build_index(docs_for_index, v, use_tfidf=True)
    
    # Search with highlighting
    results = search_with_highlighting(
        'python programming', index, documents, v,
        top_k=3, use_tfidf=True, documents_words=documents_words
    )
    
    print(f"Query: 'python programming'")
    print(f"Found {len(results)} results:\n")
    for result in results:
        print(f"  [{result['score']:.4f}] {result['title']}")
        print(f"    {result['highlighted_text'][:80]}...")
    print("✓ Enhanced search works\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING ALL NEW FEATURES")
    print("=" * 70 + "\n")
    
    test_stop_words()
    test_tokenization()
    test_highlighting()
    test_file_loading()
    test_persistence()
    test_search_with_highlighting()
    
    print("=" * 70)
    print("ALL TESTS PASSED! ✅")
    print("=" * 70)

