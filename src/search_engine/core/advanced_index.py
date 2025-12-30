"""
Advanced indexing with positional inverted index, heap-based top-K, and caching.
Implements production-ready data structures for search engine performance.
"""

import math
import heapq
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from search_engine.core.vector_search import tokenize, get_words, tf, idf, tfidf, STOP_WORDS


@dataclass
class Posting:
    """Represents a posting in the inverted index."""
    doc_id: int
    tf: float
    positions: List[int]  # Token positions where term appears
    tfidf_score: float = 0.0


class PositionalInvertedIndex:
    """
    Positional inverted index: term -> [(doc_id, tf, [positions]), ...]
    Enables phrase search and precise highlighting.
    """
    
    def __init__(self):
        self.index: Dict[str, List[Posting]] = defaultdict(list)
        self.doc_lengths: Dict[int, int] = {}  # doc_id -> number of tokens
        self.vocabulary: set = set()
    
    def add_document(self, doc_id: int, text: str, documents_words: List[List[str]] = None):
        """
        Add a document to the index with positional information.
        
        Args:
            doc_id: Document identifier
            text: Document text
            documents_words: All document word lists for IDF calculation
        """
        # Tokenize with positions
        tokens = tokenize(text, remove_stopwords=False)
        self.doc_lengths[doc_id] = len(tokens)
        
        # Build term -> positions mapping
        term_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)
            self.vocabulary.add(token)
        
        # Calculate TF-IDF for each term
        if documents_words:
            doc_words = [t.lower() for t in tokens]
            for term, positions in term_positions.items():
                term_tf = tf(term, doc_words)
                term_idf = idf(term, documents_words) if documents_words else 0.0
                term_tfidf = term_tf * term_idf
                
                posting = Posting(
                    doc_id=doc_id,
                    tf=term_tf,
                    positions=positions,
                    tfidf_score=term_tfidf
                )
                self.index[term].append(posting)
        else:
            # Simple TF only
            doc_words = [t.lower() for t in tokens]
            for term, positions in term_positions.items():
                term_tf = tf(term, doc_words)
                posting = Posting(
                    doc_id=doc_id,
                    tf=term_tf,
                    positions=positions,
                    tfidf_score=term_tf
                )
                self.index[term].append(posting)
    
    def get_postings(self, term: str) -> List[Posting]:
        """Get all postings for a term."""
        return self.index.get(term.lower(), [])
    
    def get_doc_vector(self, doc_id: int) -> Dict[str, float]:
        """Get TF-IDF vector for a document (for cosine similarity)."""
        vector = {}
        for term, postings in self.index.items():
            for posting in postings:
                if posting.doc_id == doc_id:
                    vector[term] = posting.tfidf_score
                    break
        return vector


class LRUCache:
    """
    LRU (Least Recently Used) cache for query results.
    Improves performance for repeated queries.
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating access order."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Add or update value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


def top_k_heap(results: List[Tuple[float, Any]], k: int) -> List[Tuple[float, Any]]:
    """
    Get top K results using a min-heap.
    More efficient than sorting all results.
    
    Args:
        results: List of (score, item) tuples
        k: Number of top results to return
        
    Returns:
        Top K results sorted by score (descending)
    """
    if len(results) <= k:
        return sorted(results, reverse=True, key=lambda x: x[0])
    
    # Use min-heap to keep only top K
    heap = []
    for score, item in results:
        if len(heap) < k:
            heapq.heappush(heap, (score, item))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, item))
    
    # Convert to sorted list (descending)
    return sorted(heap, reverse=True, key=lambda x: x[0])


def build_positional_index(documents: Dict[int, Any], use_tfidf: bool = True) -> PositionalInvertedIndex:
    """
    Build a positional inverted index from documents.
    
    Args:
        documents: Dictionary mapping doc_id to document data
        use_tfidf: Whether to use TF-IDF weighting
        
    Returns:
        PositionalInvertedIndex instance
    """
    index = PositionalInvertedIndex()
    
    # Prepare documents_words for TF-IDF
    documents_words = []
    doc_ids = []
    for doc_id in sorted(documents.keys()):
        doc_data = documents[doc_id]
        text = doc_data['text'] if isinstance(doc_data, dict) else doc_data
        words = get_words(text.lower())
        documents_words.append(words)
        doc_ids.append(doc_id)
    
    # Add each document to index
    for i, doc_id in enumerate(doc_ids):
        doc_data = documents[doc_id]
        text = doc_data['text'] if isinstance(doc_data, dict) else doc_data
        
        if use_tfidf:
            index.add_document(doc_id, text, documents_words)
        else:
            index.add_document(doc_id, text)
    
    return index


def search_phrase(
    phrase_terms: List[str],
    index: PositionalInvertedIndex,
    documents: Dict[int, Any]
) -> List[int]:
    """
    Search for exact phrase using positional index.
    
    Args:
        phrase_terms: List of terms in the phrase (e.g., ["machine", "learning"])
        index: PositionalInvertedIndex instance
        documents: Dictionary of documents
        
    Returns:
        List of document IDs containing the phrase
    """
    if len(phrase_terms) < 2:
        return []
    
    # Get postings for first term
    first_term = phrase_terms[0].lower()
    first_postings = index.get_postings(first_term)
    
    if not first_postings:
        return []
    
    # For each document containing first term, check if subsequent terms appear adjacently
    phrase_docs = []
    for posting in first_postings:
        doc_id = posting.doc_id
        positions = posting.positions
        
        # Check if subsequent terms appear at position+1, position+2, etc.
        for start_pos in positions:
            match = True
            for i, term in enumerate(phrase_terms[1:], 1):
                term_postings = index.get_postings(term.lower())
                doc_term_postings = [p for p in term_postings if p.doc_id == doc_id]
                
                if not doc_term_postings:
                    match = False
                    break
                
                # Check if term appears at start_pos + i
                expected_pos = start_pos + i
                term_positions = doc_term_postings[0].positions
                if expected_pos not in term_positions:
                    match = False
                    break
            
            if match:
                phrase_docs.append(doc_id)
                break  # Found phrase in this doc, no need to check other positions
    
    return phrase_docs


def search_with_positional_index(
    query: str,
    index: PositionalInvertedIndex,
    documents: Dict[int, Any],
    top_k: int = 10,
    cache: Optional[LRUCache] = None
) -> List[Tuple[float, int, Dict[str, List[int]]]]:
    """
    Search using positional inverted index with heap-based top-K retrieval.
    
    Args:
        query: Search query string
        index: PositionalInvertedIndex instance
        documents: Dictionary of documents
        top_k: Number of results to return
        cache: Optional LRU cache for query results
        
    Returns:
        List of (score, doc_id, term_positions) tuples sorted by score
    """
    # Check cache first
    cache_key = f"{query.lower()}:{top_k}"
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    
    # Check for phrase search (quoted strings)
    import re
    phrase_pattern = r'"([^"]+)"'
    phrases = re.findall(phrase_pattern, query)
    
    # Remove phrases from query for regular search
    query_without_phrases = query
    for phrase in phrases:
        query_without_phrases = query_without_phrases.replace(f'"{phrase}"', '')
    
    # Tokenize remaining query
    query_terms = tokenize(query_without_phrases.lower(), remove_stopwords=False)
    
    # Handle phrase searches
    phrase_docs = set()
    if phrases:
        for phrase in phrases:
            phrase_terms_list = tokenize(phrase.lower(), remove_stopwords=False)
            if len(phrase_terms_list) >= 2:
                found_docs = search_phrase(phrase_terms_list, index, documents)
                if phrase_docs:
                    phrase_docs = phrase_docs & set(found_docs)  # Intersection for multiple phrases
                else:
                    phrase_docs = set(found_docs)
    
    if not query_terms and not phrase_docs:
        return []
    
    # Score documents
    doc_scores: Dict[int, float] = defaultdict(float)
    doc_term_positions: Dict[int, Dict[str, List[int]]] = defaultdict(dict)
    
    # Boost scores for phrase matches
    phrase_boost = 2.0
    
    for term in query_terms:
        postings = index.get_postings(term)
        for posting in postings:
            doc_id = posting.doc_id
            # Boost score if document contains a phrase match
            boost = phrase_boost if doc_id in phrase_docs else 1.0
            doc_scores[doc_id] += posting.tfidf_score * boost
            # Store positions for highlighting
            doc_term_positions[doc_id][term] = posting.positions
    
    # Add phrase-only documents (if they don't match regular terms)
    if phrase_docs:
        for doc_id in phrase_docs:
            if doc_id not in doc_scores:
                # Give phrase matches a base score
                doc_scores[doc_id] = 0.5 * phrase_boost
                doc_term_positions[doc_id] = {}
    
    # Convert to list of (score, doc_id, positions) tuples
    results = []
    for doc_id, score in doc_scores.items():
        if score > 0:
            results.append((score, doc_id, doc_term_positions[doc_id]))
    
    # Use heap for top-K
    top_results = top_k_heap(results, top_k)
    
    # Cache results
    if cache:
        cache.put(cache_key, top_results)
    
    return top_results


def highlight_with_positions(
    text: str,
    term_positions: Dict[str, List[int]],
    max_length: int = 300
) -> str:
    """
    Highlight query terms using positional index.
    More accurate than naive string replacement.
    
    Args:
        text: Original text
        term_positions: Dict mapping terms to their token positions
        max_length: Maximum length of preview
        
    Returns:
        HTML string with highlighted terms
    """
    # Tokenize to get tokens with positions
    tokens = tokenize(text, remove_stopwords=False)
    
    # Create position -> term mapping
    pos_to_term = {}
    for term, positions in term_positions.items():
        for pos in positions:
            if pos < len(tokens):
                pos_to_term[pos] = term
    
    # Build highlighted text
    highlighted_tokens = []
    for i, token in enumerate(tokens):
        if i in pos_to_term:
            highlighted_tokens.append(f'<mark>{token}</mark>')
        else:
            highlighted_tokens.append(token)
    
    result = ' '.join(highlighted_tokens)
    
    # Truncate if needed
    if len(result) > max_length:
        result = result[:max_length] + '...'
    
    return result

