"""
Vector Space Indexing Engine with TF-IDF Scoring

A search engine implementation using vector space model for document ranking.
Enhanced with TF-IDF (Term Frequency-Inverse Document Frequency) for better term weighting.
Based on cosine similarity between document and query vectors.
"""

import math
import re
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
    'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
    'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
    'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
    'who', 'oil', 'its', 'now', 'find', 'down', 'day', 'did', 'get',
    'come', 'made', 'may', 'part'
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text into words, handling punctuation and case normalization.
    
    Args:
        text: The text to tokenize
        remove_stopwords: If True, filter out stop words
        
    Returns:
        List of normalized word tokens
    """
    # Convert to lowercase and split on non-word characters
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    
    return words


def concordance(document, remove_stopwords: bool = True):
    """
    Create a concordance (word count dictionary) from a document.
    
    Args:
        document (str): The text document to process
        remove_stopwords (bool): If True, filter out stop words
        
    Returns:
        dict: A dictionary mapping words to their counts
        
    Raises:
        ValueError: If document is not a string
    """
    if type(document) != str:
        raise ValueError('Supplied Argument should be of type string')
    words = tokenize(document, remove_stopwords=remove_stopwords)
    con = {}
    for word in words:
        con[word] = con.get(word, 0) + 1
    return con


def get_words(document, remove_stopwords: bool = True):
    """
    Extract words from a document (normalized to lowercase).
    Uses improved tokenization with punctuation handling.
    
    Args:
        document (str): The text document to process
        remove_stopwords (bool): If True, filter out stop words
        
    Returns:
        list: List of words in the document
    """
    if type(document) != str:
        raise ValueError('Supplied Argument should be of type string')
    return tokenize(document, remove_stopwords=remove_stopwords)


def tf(word, words_list):
    """
    Calculate Term Frequency: the number of times a word appears in a document,
    normalized by the total number of words.
    
    Args:
        word (str): The word to calculate TF for
        words_list (list): List of words in the document
        
    Returns:
        float: Term frequency score
    """
    if not words_list:
        return 0
    return words_list.count(word.lower()) / len(words_list)


def n_containing(word, documents_words):
    """
    Count the number of documents containing a word.
    
    Args:
        word (str): The word to search for
        documents_words (list): List of word lists (one per document)
        
    Returns:
        int: Number of documents containing the word
    """
    return sum(1 for words in documents_words if word.lower() in words)


def idf(word, documents_words):
    """
    Calculate Inverse Document Frequency: measures how common a word is
    among all documents. The more common a word is, the lower its idf.
    
    Args:
        word (str): The word to calculate IDF for
        documents_words (list): List of word lists (one per document)
        
    Returns:
        float: Inverse document frequency score
    """
    if not documents_words:
        return 0
    # Add 1 to prevent division by zero
    return math.log(len(documents_words) / (1 + n_containing(word, documents_words)))


def tfidf(word, words_list, documents_words):
    """
    Calculate TF-IDF score: the product of term frequency and inverse document frequency.
    Higher scores indicate more important/unique words in a document.
    
    Args:
        word (str): The word to calculate TF-IDF for
        words_list (list): List of words in the current document
        documents_words (list): List of word lists (one per document)
        
    Returns:
        float: TF-IDF score
    """
    return tf(word, words_list) * idf(word, documents_words)


class VectorCompare:
    """
    Vector comparison class for calculating document similarity using cosine similarity.
    """
    
    def magnitude(self, concordance):
        """
        Calculate the magnitude (length) of a concordance vector.
        
        Args:
            concordance (dict): Word count dictionary
            
        Returns:
            float: The magnitude of the vector
            
        Raises:
            ValueError: If concordance is not a dictionary
        """
        if type(concordance) != dict:
            raise ValueError('Supplied Argument should be of type dict')
        total = 0
        for word, count in concordance.items():
            total += count ** 2
        return math.sqrt(total)

    def relation(self, concordance1, concordance2):
        """
        Calculate the cosine similarity (relation) between two concordances.
        Returns a value between 0 and 1, where 1 means identical and 0 means no similarity.
        
        Args:
            concordance1 (dict): First word count dictionary (typically the query)
            concordance2 (dict): Second word count dictionary (typically a document)
            
        Returns:
            float: Cosine similarity score between 0 and 1
            
        Raises:
            ValueError: If either argument is not a dictionary
        """
        if type(concordance1) != dict:
            raise ValueError('Supplied Argument 1 should be of type dict')
        if type(concordance2) != dict:
            raise ValueError('Supplied Argument 2 should be of type dict')
        
        relevance = 0
        topvalue = 0
        
        # Calculate dot product
        for word, count in concordance1.items():
            if word in concordance2:
                topvalue += count * concordance2[word]
        
        # Calculate cosine similarity (avoid division by zero)
        mag1 = self.magnitude(concordance1)
        mag2 = self.magnitude(concordance2)
        
        if (mag1 * mag2) != 0:
            return topvalue / (mag1 * mag2)
        else:
            return 0

    def concordance(self, document, remove_stopwords: bool = True):
        """
        Create a concordance from a document (wrapper for the standalone function).
        
        Args:
            document (str): The text document to process
            remove_stopwords (bool): If True, filter out stop words
            
        Returns:
            dict: A dictionary mapping words to their counts
        """
        return concordance(document, remove_stopwords=remove_stopwords)


def build_index(documents, vector_compare, use_tfidf=False):
    """
    Build an index (concordance dictionary) for a collection of documents.
    Optionally uses TF-IDF weighting for better term importance scoring.
    
    Args:
        documents (dict): Dictionary mapping document IDs to document text
        vector_compare (VectorCompare): VectorCompare instance to use
        use_tfidf (bool): If True, use TF-IDF weighted vectors instead of raw counts
        
    Returns:
        dict: Dictionary mapping document IDs to their concordances (or TF-IDF vectors)
    """
    if not use_tfidf:
        # Original simple concordance approach
        index = {}
        for doc_id, doc_text in documents.items():
            index[doc_id] = vector_compare.concordance(doc_text.lower())
        return index
    else:
        # TF-IDF weighted approach
        # First, get all document word lists
        documents_words = []
        doc_ids = []
        for doc_id in sorted(documents.keys()):
            words = get_words(documents[doc_id].lower())
            documents_words.append(words)
            doc_ids.append(doc_id)
        
        # Build TF-IDF weighted index
        index = {}
        for i, doc_id in enumerate(doc_ids):
            words_list = documents_words[i]
            # Get unique words in this document
            unique_words = list(set(words_list))
            # Calculate TF-IDF for each word
            tfidf_vector = {}
            for word in unique_words:
                tfidf_vector[word] = tfidf(word, words_list, documents_words)
            index[doc_id] = tfidf_vector
        
        return index


def search(query, index, documents, vector_compare, top_k=None, use_tfidf=False, documents_words=None):
    """
    Search for documents matching a query.
    Optionally uses TF-IDF weighting for query terms.
    
    Args:
        query (str): Search query string
        index (dict): Dictionary mapping document IDs to concordances (or TF-IDF vectors)
        documents (dict): Dictionary mapping document IDs to document text
        vector_compare (VectorCompare): VectorCompare instance to use
        top_k (int, optional): Number of top results to return. If None, returns all matches.
        use_tfidf (bool): If True, use TF-IDF weighted query vector
        documents_words (list, optional): List of word lists for TF-IDF calculation
        
    Returns:
        list: List of tuples (score, document_preview, doc_id) sorted by relevance (descending)
    """
    if use_tfidf and documents_words is not None:
        # Build TF-IDF weighted query vector
        # For query, we want to include all words (even if they'd normally be stop words)
        # because the user explicitly searched for them
        query_words = tokenize(query.lower(), remove_stopwords=False)
        unique_query_words = list(set(query_words))
        query_vector = {}
        for word in unique_query_words:
            # Calculate TF-IDF, but use documents_words for IDF calculation
            # The query is treated as a single document for TF
            query_tf = tf(word, query_words)
            # IDF is calculated from the document corpus
            query_idf = idf(word, documents_words)
            query_vector[word] = query_tf * query_idf
    else:
        # Simple concordance for query (don't remove stop words for queries)
        query_vector = concordance(query.lower(), remove_stopwords=False)
    
    matches = []
    
    for doc_id in index:
        relation = vector_compare.relation(query_vector, index[doc_id])
        if relation != 0:
            # Handle both dict format (with metadata) and string format (legacy)
            if isinstance(documents[doc_id], dict):
                doc_text = documents[doc_id]['text']
            else:
                doc_text = documents[doc_id]
            
            # Store score and first 100 characters of document
            doc_preview = doc_text[:100] if len(doc_text) > 100 else doc_text
            matches.append((relation, doc_preview, doc_id))
    
    # Sort by relevance (descending)
    matches.sort(reverse=True, key=lambda x: x[0])
    
    # Return top_k results if specified
    if top_k is not None:
        matches = matches[:top_k]
    
    return matches


def highlight_query_terms(text: str, query: str, max_length: int = 200) -> str:
    """
    Highlight query terms in text with HTML tags.
    
    Args:
        text: The text to highlight
        query: The search query
        max_length: Maximum length of preview text
        
    Returns:
        HTML string with highlighted terms
    """
    query_words = set(tokenize(query.lower(), remove_stopwords=False))
    words = text.split()
    
    # Truncate if needed
    if len(' '.join(words)) > max_length:
        words = words[:max_length // 10]  # Rough estimate
        text = ' '.join(words) + '...'
    
    # Highlight matching words
    highlighted = []
    for word in text.split():
        word_lower = re.sub(r'[^\w]', '', word.lower())
        if word_lower in query_words:
            highlighted.append(f'<mark>{word}</mark>')
        else:
            highlighted.append(word)
    
    return ' '.join(highlighted)


def load_documents_from_directory(directory: str, extensions: List[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Load documents from a directory of text files.
    Note: For PDF/DOCX files, use the web interface or parse_document function in app.py
    
    Args:
        directory: Path to directory containing text files
        extensions: List of file extensions to load (default: ['.txt', '.md'])
        
    Returns:
        Dictionary mapping doc_id to document dict with 'text', 'title', 'path' keys
    """
    if extensions is None:
        extensions = ['.txt', '.md']
    
    documents = {}
    doc_id = 0
    path = Path(directory)
    
    if not path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    for file_path in path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    title = file_path.stem
                    documents[doc_id] = {
                        'text': text,
                        'title': title,
                        'path': str(file_path),
                        'url': f"/doc/{doc_id}"  # For web interface
                    }
                    doc_id += 1
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    return documents


def load_documents_from_files(file_paths: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Load documents from a list of file paths.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        Dictionary mapping doc_id to document dict
    """
    documents = {}
    for doc_id, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                title = Path(file_path).stem
                documents[doc_id] = {
                    'text': text,
                    'title': title,
                    'path': file_path,
                    'url': f"/doc/{doc_id}"
                }
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return documents


def save_index(index: Dict, documents: Dict, documents_words: List[List[str]], 
               filepath: str = 'search_index.json'):
    """
    Save the search index and documents to a JSON file.
    
    Args:
        index: The search index
        documents: Dictionary of documents with metadata
        documents_words: List of word lists for each document
        filepath: Path to save the index
    """
    # Convert documents to serializable format
    serializable_docs = {}
    for doc_id, doc_data in documents.items():
        if isinstance(doc_data, dict):
            serializable_docs[str(doc_id)] = doc_data
        else:
            # Legacy format - just text
            serializable_docs[str(doc_id)] = {
                'text': doc_data,
                'title': f'Document {doc_id}',
                'path': '',
                'url': f"/doc/{doc_id}"
            }
    
    data = {
        'index': {str(k): v for k, v in index.items()},
        'documents': serializable_docs,
        'documents_words': documents_words,
        'version': '1.0'
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Index saved to {filepath}")


def load_index(filepath: str = 'search_index.json') -> Tuple[Dict, Dict, List[List[str]]]:
    """
    Load the search index and documents from a JSON file.
    
    Args:
        filepath: Path to load the index from
        
    Returns:
        Tuple of (index, documents, documents_words)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert back to integer keys
    index = {int(k): v for k, v in data['index'].items()}
    documents = {int(k): v for k, v in data['documents'].items()}
    documents_words = data['documents_words']
    
    print(f"Index loaded from {filepath}")
    return index, documents, documents_words


def search_with_highlighting(query: str, index: Dict, documents: Dict, vector_compare: VectorCompare,
                             top_k: int = 10, use_tfidf: bool = True, documents_words: List[List[str]] = None) -> List[Dict]:
    """
    Search with enhanced results including highlighting and metadata.
    
    Args:
        query: Search query string
        index: Search index
        documents: Dictionary of documents (with metadata)
        vector_compare: VectorCompare instance
        top_k: Number of results to return
        use_tfidf: Use TF-IDF weighting
        documents_words: Word lists for TF-IDF
        
    Returns:
        List of result dictionaries with score, text, title, path, highlighted_text, doc_id
    """
    # Get raw search results
    raw_results = search(query, index, documents, vector_compare, top_k=top_k, 
                       use_tfidf=use_tfidf, documents_words=documents_words)
    
    # Enhance with metadata and highlighting
    enhanced_results = []
    for score, preview, doc_id in raw_results:
        doc_data = documents[doc_id]
        
        # Handle both new format (dict) and legacy format (string)
        if isinstance(doc_data, dict):
            text = doc_data['text']
            title = doc_data.get('title', f'Document {doc_id}')
            path = doc_data.get('path', '')
            url = doc_data.get('url', f'/doc/{doc_id}')
        else:
            text = doc_data
            title = f'Document {doc_id}'
            path = ''
            url = f'/doc/{doc_id}'
        
        highlighted = highlight_query_terms(text, query, max_length=300)
        
        enhanced_results.append({
            'score': score,
            'doc_id': doc_id,
            'title': title,
            'text': text[:300] + '...' if len(text) > 300 else text,
            'highlighted_text': highlighted,
            'path': path,
            'url': url
        })
    
    return enhanced_results


def main():
    """
    Main function demonstrating the search engine with sample documents.
    Uses TF-IDF scoring for better relevance ranking.
    """
    # Sample documents (blog post titles and first paragraphs)
    documents = {
        0: '''At Scale You Will Hit Every Performance Issue I used to think I knew a bit about performance scalability and how to keep things trucking when you hit large amounts of data Truth is I know diddly squat on the subject since the most I have ever done is read about how its done To understand how I came about realising this you need some background''',
        1: '''Richard Stallman to visit Australia Im not usually one to promote events and the like unless I feel there is a genuine benefit to be had by attending but this is one stands out Richard M Stallman the guru of Free Software is coming Down Under to hold a talk You can read about him here Open Source Celebrity to visit Australia''',
        2: '''MySQL Backups Done Easily One thing that comes up a lot on sites like Stackoverflow and the like is how to backup MySQL databases The first answer is usually use mysqldump This is all fine and good till you start to want to dump multiple databases You can do this all in one like using the all databases option however this makes restoring a single database an issue since you have to parse out the parts you want which can be a pain''',
        3: '''Why You Shouldnt roll your own CAPTCHA At a TechEd I attended a few years ago I was watching a presentation about Security presented by Rocky Heckman read his blog its quite good In it he was talking about security algorithms The part that really stuck with me went like this''',
        4: '''The Great Benefit of Test Driven Development Nobody Talks About The feeling of productivity because you are writing lots of code Think about that for a moment Ask any developer who wants to develop why they became a developer One of the first things that comes up is I enjoy writing code This is one of the things that I personally enjoy doing Writing code any code especially when its solving my current problem makes me feel productive It makes me feel like Im getting somewhere Its empowering''',
        5: '''Setting up GIT to use a Subversion SVN style workflow Moving from Subversion SVN to GIT can be a little confusing at first I think the biggest thing I noticed was that GIT doesnt have a specific workflow you have to pick your own Personally I wanted to stick to my Subversion like work-flow with a central server which all my machines would pull and push too Since it took a while to set up I thought I would throw up a blog post on how to do it''',
        6: '''Why CAPTCHA Never Use Numbers 0 1 5 7 Interestingly this sort of question pops up a lot in my referring search term stats Why CAPTCHAs never use the numbers 0 1 5 7 Its a relativity simple question with a reasonably simple answer Its because each of the above numbers are easy to confuse with a letter See the below''',
    }
    
    # Initialize vector comparison
    v = VectorCompare()
    
    # Prepare documents_words for TF-IDF (needed for query weighting)
    documents_words = []
    for doc_id in sorted(documents.keys()):
        words = get_words(documents[doc_id].lower())
        documents_words.append(words)
    
    # Build index with TF-IDF weighting
    print("Building index with TF-IDF weighting...")
    index = build_index(documents, v, use_tfidf=True)
    print(f"Indexed {len(index)} documents.\n")
    
    # Interactive search loop
    print("Search Engine Ready! (Using TF-IDF scoring)")
    print("Enter search terms (or 'quit' to exit):\n")
    
    while True:
        try:
            searchterm = input('Enter Search Term: ').strip()
            
            if searchterm.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not searchterm:
                continue
            
            # Perform search with TF-IDF
            matches = search(searchterm, index, documents, v, use_tfidf=True, documents_words=documents_words)
            
            # Display results
            if matches:
                print(f"\nFound {len(matches)} result(s):\n")
                for i, (score, preview, doc_id) in enumerate(matches, 1):
                    print(f"{i}. [Score: {score:.6f}] [Doc ID: {doc_id}]")
                    print(f"   {preview}...\n")
            else:
                print("No matches found.\n")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

