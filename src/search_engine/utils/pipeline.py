"""
ETL Pipeline for document ingestion with data quality checks.
"""

import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from search_engine.core.vector_search import get_words


@dataclass
class PipelineMetrics:
    """Metrics for a pipeline run."""
    run_id: str
    timestamp: str
    docs_ingested: int
    docs_skipped: int
    docs_duplicate: int
    tokens_processed: int
    unique_terms: int
    index_size_bytes: int
    extraction_time: float
    transformation_time: float
    indexing_time: float
    total_time: float
    errors: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    doc_id: int
    title: str
    text: str
    path: str
    simhash: str  # For duplicate detection
    word_count: int
    char_count: int
    created_at: str


class DocumentPipeline:
    """
    ETL pipeline for document processing with data quality checks.
    """
    
    def __init__(self):
        self.metrics_history: List[PipelineMetrics] = []
        self.doc_hashes: Dict[str, int] = {}  # simhash -> doc_id for dedup
    
    def compute_simhash(self, text: str) -> str:
        """
        Compute a simple hash for duplicate detection.
        Uses first 1000 chars for performance.
        """
        sample = text[:1000].lower().strip()
        return hashlib.md5(sample.encode()).hexdigest()
    
    def extract(self, source: Any) -> Dict[str, Any]:
        """
        Extract stage: Load document from source.
        
        Args:
            source: Document data (dict with 'text' or file path)
            
        Returns:
            Extracted document data
        """
        start_time = time.time()
        
        if isinstance(source, dict):
            text = source.get('text', '')
            title = source.get('title', 'Untitled')
            path = source.get('path', '')
        else:
            # Assume it's a file path
            with open(source, 'r', encoding='utf-8') as f:
                text = f.read()
            title = Path(source).stem
            path = str(source)
        
        extraction_time = time.time() - start_time
        
        return {
            'text': text,
            'title': title,
            'path': path,
            'extraction_time': extraction_time
        }
    
    def transform(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform stage: Normalize and validate data.
        
        Args:
            extracted: Data from extract stage
            
        Returns:
            Transformed document with metadata
        """
        start_time = time.time()
        
        text = extracted['text']
        
        # Data quality checks
        errors = []
        
        # Check for empty document
        if not text or not text.strip():
            errors.append("Document is empty")
            return {'errors': errors, 'valid': False}
        
        # Check minimum length
        if len(text.strip()) < 10:
            errors.append("Document too short (minimum 10 characters)")
        
        # Compute hash for duplicate detection
        simhash = self.compute_simhash(text)
        
        # Check for duplicates
        is_duplicate = simhash in self.doc_hashes
        
        # Tokenize and count
        words = get_words(text.lower())
        word_count = len(words)
        char_count = len(text)
        
        transformation_time = time.time() - start_time
        
        return {
            'text': text,
            'title': extracted['title'],
            'path': extracted['path'],
            'simhash': simhash,
            'word_count': word_count,
            'char_count': char_count,
            'is_duplicate': is_duplicate,
            'errors': errors,
            'valid': len(errors) == 0,
            'transformation_time': transformation_time
        }
    
    def load(self, transformed: Dict[str, Any], doc_id: int) -> Dict[str, Any]:
        """
        Load stage: Add document to index and store.
        
        Args:
            transformed: Data from transform stage
            doc_id: Document ID
            
        Returns:
            Document metadata
        """
        if not transformed.get('valid', False):
            return None
        
        # Store hash for duplicate detection
        self.doc_hashes[transformed['simhash']] = doc_id
        
        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=transformed['title'],
            text=transformed['text'],
            path=transformed['path'],
            simhash=transformed['simhash'],
            word_count=transformed['word_count'],
            char_count=transformed['char_count'],
            created_at=datetime.now().isoformat()
        )
        
        return metadata
    
    def process_documents(
        self,
        sources: List[Any],
        start_doc_id: int = 0
    ) -> Tuple[Dict[int, DocumentMetadata], PipelineMetrics]:
        """
        Run full ETL pipeline on multiple documents.
        
        Args:
            sources: List of document sources
            start_doc_id: Starting document ID
            
        Returns:
            Tuple of (documents dict, metrics)
        """
        pipeline_start = time.time()
        run_id = f"run_{int(pipeline_start)}"
        
        documents = {}
        errors = []
        docs_ingested = 0
        docs_skipped = 0
        docs_duplicate = 0
        total_tokens = 0
        all_terms = set()
        
        extraction_times = []
        transformation_times = []
        indexing_times = []
        
        doc_id = start_doc_id
        
        for source in sources:
            try:
                # Extract
                extracted = self.extract(source)
                extraction_times.append(extracted['extraction_time'])
                
                # Transform
                transformed = self.transform(extracted)
                transformation_times.append(transformed.get('transformation_time', 0))
                
                # Check for duplicates
                if transformed.get('is_duplicate', False):
                    docs_duplicate += 1
                    continue
                
                # Check validity
                if not transformed.get('valid', False):
                    docs_skipped += 1
                    errors.extend(transformed.get('errors', []))
                    continue
                
                # Load
                metadata = self.load(transformed, doc_id)
                if metadata:
                    documents[doc_id] = metadata
                    total_tokens += transformed['word_count']
                    all_terms.update(get_words(transformed['text'].lower()))
                    docs_ingested += 1
                    doc_id += 1
                
            except Exception as e:
                docs_skipped += 1
                errors.append(f"Error processing document: {str(e)}")
        
        total_time = time.time() - pipeline_start
        
        metrics = PipelineMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            docs_ingested=docs_ingested,
            docs_skipped=docs_skipped,
            docs_duplicate=docs_duplicate,
            tokens_processed=total_tokens,
            unique_terms=len(all_terms),
            index_size_bytes=0,  # Will be calculated after indexing
            extraction_time=sum(extraction_times),
            transformation_time=sum(transformation_times),
            indexing_time=0,  # Calculated separately
            total_time=total_time,
            errors=errors
        )
        
        self.metrics_history.append(metrics)
        
        return documents, metrics

