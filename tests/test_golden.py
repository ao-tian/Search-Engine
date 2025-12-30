"""
Golden test set for regression testing.
Ensures search results remain consistent across code changes.
"""

import pytest
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from search_engine.core.vector_search import VectorCompare, build_index, get_words
from search_engine.core.advanced_index import build_positional_index, search_with_positional_index


def load_golden_test_set():
    """Load the golden test set."""
    test_file = Path(__file__).parent / "golden_test_set.json"
    with open(test_file, 'r') as f:
        return json.load(f)


class TestGoldenSet:
    """Test against golden test set."""
    
    @pytest.fixture
    def golden_data(self):
        """Load golden test data."""
        return load_golden_test_set()
    
    @pytest.fixture
    def indexed_documents(self, golden_data):
        """Index the golden test documents."""
        documents = {}
        for doc in golden_data['documents']:
            documents[doc['id']] = {
                'text': doc['text'],
                'title': doc['title'],
                'path': '',
                'url': f"/doc/{doc['id']}"
            }
        
        # Build positional index
        index = build_positional_index(documents, use_tfidf=True)
        return documents, index
    
    def test_golden_queries(self, golden_data, indexed_documents):
        """Test all queries in golden set."""
        documents, index = indexed_documents
        
        for test_case in golden_data['expected_results']:
            query = test_case['query']
            expected_doc_ids = set(test_case['expected_doc_ids'])
            min_score = test_case['min_score']
            
            # Perform search
            results = search_with_positional_index(
                query, index, documents, top_k=test_case['top_k']
            )
            
            # Check that we got results
            assert len(results) > 0, f"No results for query: {query}"
            
            # Check that expected documents are in results
            result_doc_ids = {doc_id for _, doc_id, _ in results}
            found_expected = result_doc_ids & expected_doc_ids
            
            assert len(found_expected) > 0, \
                f"Query '{query}' should find at least one expected document. " \
                f"Expected: {expected_doc_ids}, Got: {result_doc_ids}"
            
            # Check minimum score
            top_score = results[0][0] if results else 0
            assert top_score >= min_score, \
                f"Top score {top_score} below minimum {min_score} for query: {query}"
    
    def test_golden_document_count(self, golden_data):
        """Test that all golden documents are loaded."""
        assert len(golden_data['documents']) == 5
        assert len(golden_data['expected_results']) == 6
    
    def test_golden_document_quality(self, golden_data):
        """Test that golden documents have required fields."""
        for doc in golden_data['documents']:
            assert 'id' in doc
            assert 'title' in doc
            assert 'text' in doc
            assert len(doc['text']) > 10  # Non-empty text

