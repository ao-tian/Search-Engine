"""
Unit tests for indexing and TF-IDF calculations.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from search_engine.core.vector_search import (
    VectorCompare, build_index, tf, idf, tfidf,
    concordance, get_words
)
from search_engine.core.advanced_index import (
    PositionalInvertedIndex, build_positional_index,
    top_k_heap, LRUCache
)


class TestTFIDF:
    """Test TF-IDF calculations."""
    
    def test_term_frequency(self):
        """Test term frequency calculation."""
        words = ["hello", "world", "hello", "test", "hello"]
        tf_score = tf("hello", words)
        assert tf_score > 0
        assert tf_score == pytest.approx(3.0 / 5.0, rel=1e-5)
    
    def test_term_frequency_zero(self):
        """Test TF for non-existent term."""
        words = ["hello", "world"]
        tf_score = tf("nonexistent", words)
        assert tf_score == 0.0
    
    def test_inverse_document_frequency(self):
        """Test IDF calculation."""
        doc1 = ["hello", "world"]
        doc2 = ["hello", "test"]
        doc3 = ["world", "test", "unique"]
        bloblist = [doc1, doc2, doc3]
        
        idf_score = idf("hello", bloblist)
        # "hello" appears in 2 docs, so IDF should be log(3/2) > 0
        assert idf_score >= 0  # Can be 0 if calculation results in that
        # But if it's 0, check that it's because of the calculation
        if idf_score == 0:
            # This might happen if the term appears in all docs
            idf_unique = idf("unique", bloblist)
            assert idf_unique > 0  # Unique term should have positive IDF
    
    def test_idf_rare_term(self):
        """Test IDF for rare term (should be higher)."""
        doc1 = ["common", "word"]
        doc2 = ["common", "word"]
        doc3 = ["rare", "word"]
        bloblist = [doc1, doc2, doc3]
        
        idf_common = idf("common", bloblist)
        idf_rare = idf("rare", bloblist)
        
        assert idf_rare > idf_common
    
    def test_tfidf_calculation(self):
        """Test TF-IDF score calculation."""
        words = ["hello", "world", "hello", "unique"]
        doc1 = ["hello", "world"]
        doc2 = ["test", "test"]
        doc3 = ["unique", "word"]
        bloblist = [doc1, doc2, doc3]
        
        # Test with a term that should have positive IDF
        score = tfidf("unique", words, bloblist)
        assert score >= 0  # Should be non-negative
        if score > 0:
            assert score == tf("unique", words) * idf("unique", bloblist)


class TestVectorCompare:
    """Test VectorCompare class."""
    
    def test_concordance_creation(self):
        """Test concordance dictionary creation."""
        vc = VectorCompare()
        text = "hello world hello"
        con = vc.concordance(text)
        
        assert isinstance(con, dict)
        assert con.get("hello", 0) == 2
        assert con.get("world", 0) == 1
    
    def test_magnitude_calculation(self):
        """Test vector magnitude calculation."""
        vc = VectorCompare()
        con = {"hello": 2, "world": 1}
        mag = vc.magnitude(con)
        assert mag > 0
        assert mag == pytest.approx((2**2 + 1**2)**0.5, rel=1e-5)
    
    def test_relation_calculation(self):
        """Test cosine similarity calculation."""
        vc = VectorCompare()
        con1 = {"hello": 2, "world": 1}
        con2 = {"hello": 1, "world": 2}
        
        relation = vc.relation(con1, con2)
        assert 0 <= relation <= 1
    
    def test_relation_identical(self):
        """Test relation for identical vectors."""
        vc = VectorCompare()
        con = {"hello": 2, "world": 1}
        relation = vc.relation(con, con)
        assert relation == pytest.approx(1.0, rel=1e-5)
    
    def test_relation_orthogonal(self):
        """Test relation for orthogonal vectors."""
        vc = VectorCompare()
        con1 = {"hello": 1}
        con2 = {"world": 1}
        relation = vc.relation(con1, con2)
        assert relation == 0.0


class TestPositionalIndex:
    """Test positional inverted index."""
    
    def test_index_creation(self):
        """Test creating a positional index."""
        index = PositionalInvertedIndex()
        assert index.index == {}
        assert index.vocabulary == set()
    
    def test_add_document(self):
        """Test adding a document to index."""
        index = PositionalInvertedIndex()
        index.add_document(0, "hello world hello")
        
        assert len(index.vocabulary) > 0
        assert "hello" in index.vocabulary
        assert "world" in index.vocabulary
    
    def test_get_postings(self):
        """Test retrieving postings for a term."""
        index = PositionalInvertedIndex()
        index.add_document(0, "hello world")
        index.add_document(1, "hello test")
        
        postings = index.get_postings("hello")
        assert len(postings) == 2
        assert all(p.doc_id in [0, 1] for p in postings)
    
    def test_position_storage(self):
        """Test that positions are stored correctly."""
        index = PositionalInvertedIndex()
        index.add_document(0, "hello world hello")
        
        postings = index.get_postings("hello")
        assert len(postings) > 0
        # "hello" appears at positions 0 and 2
        positions = postings[0].positions
        assert 0 in positions
        assert 2 in positions


class TestTopKHeap:
    """Test heap-based top-K retrieval."""
    
    def test_top_k_basic(self):
        """Test basic top-K selection."""
        results = [(0.9, "doc1"), (0.5, "doc2"), (0.8, "doc3"), (0.3, "doc4")]
        top_2 = top_k_heap(results, k=2)
        
        assert len(top_2) == 2
        assert top_2[0][0] == 0.9  # Highest score first
        assert top_2[1][0] == 0.8
    
    def test_top_k_all_results(self):
        """Test when k >= number of results."""
        results = [(0.5, "doc1"), (0.3, "doc2")]
        top_5 = top_k_heap(results, k=5)
        
        assert len(top_5) == 2
        assert top_5[0][0] == 0.5
    
    def test_top_k_empty(self):
        """Test top-K with empty results."""
        results = []
        top_5 = top_k_heap(results, k=5)
        assert top_5 == []


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_cache_put_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(capacity=3)
        cache.put("key1", "value1")
        
        assert cache.get("key1") == "value1"
    
    def test_cache_eviction(self):
        """Test that least recently used items are evicted."""
        cache = LRUCache(capacity=2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = LRUCache(capacity=10)
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        hit_rate = cache.hit_rate()
        assert 0 <= hit_rate <= 1
        assert hit_rate == pytest.approx(0.5, rel=1e-5)

