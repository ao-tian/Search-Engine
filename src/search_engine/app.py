"""
Flask web interface for the search engine.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from search_engine.core.vector_search import (
    VectorCompare, build_index, search_with_highlighting,
    load_documents_from_directory, save_index, load_index,
    get_words
)
from search_engine.core.advanced_index import (
    PositionalInvertedIndex, LRUCache, build_positional_index,
    search_with_positional_index, highlight_with_positions, top_k_heap,
    search_phrase
)
from search_engine.utils.pipeline import DocumentPipeline
from search_engine.utils.metrics import MetricsCollector
from search_engine.utils.index_versioning import IndexVersionManager
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid
import time

# Document parsing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# Get the project root directory (parent of src/)
# When running from run.py, __file__ is in src/search_engine/
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
except:
    PROJECT_ROOT = Path.cwd()

app = Flask(__name__, template_folder=str(PROJECT_ROOT / 'templates'))

# Global state
vector_compare = VectorCompare()
index = None  # Legacy index (for backward compatibility)
positional_index = None  # Advanced positional index
documents = None
documents_words = None
INDEX_FILE = str(PROJECT_ROOT / 'search_index.json')
DOCUMENTS_DIR = str(PROJECT_ROOT / 'documents')  # Directory for user documents
ALLOWED_EXTENSIONS = {'txt', 'md', 'text', 'pdf', 'docx', 'doc'}

# Advanced features
query_cache = LRUCache(capacity=100)
metrics_collector = MetricsCollector()
pipeline = DocumentPipeline()
version_manager = IndexVersionManager()


def rebuild_index():
    """Rebuild the search index from current documents using ETL pipeline."""
    global index, positional_index, documents, documents_words
    
    if not documents:
        return
    
    indexing_start = time.time()
    
    # Use ETL pipeline for processing
    sources = []
    for doc_id, doc_data in documents.items():
        if isinstance(doc_data, dict):
            sources.append(doc_data)
        else:
            sources.append({'text': doc_data, 'title': f'Document {doc_id}', 'path': ''})
    
    # Process through pipeline
    processed_docs, pipeline_metrics = pipeline.process_documents(sources, start_doc_id=0)
    
    # Convert processed docs back to standard format
    if processed_docs:
        documents = {}
        for doc_id, metadata in processed_docs.items():
            documents[doc_id] = {
                'text': metadata.text,
                'title': metadata.title,
                'path': metadata.path,
                'url': f'/doc/{doc_id}',
                'word_count': metadata.word_count,
                'char_count': metadata.char_count
            }
    
    # Prepare documents_words for TF-IDF
    documents_words = []
    for doc_id in sorted(documents.keys()):
        doc_data = documents[doc_id]
        text = doc_data['text'] if isinstance(doc_data, dict) else doc_data
        words = get_words(text.lower())
        documents_words.append(words)
    
    # Build legacy index (for backward compatibility)
    docs_for_index = {}
    for doc_id, doc_data in documents.items():
        if isinstance(doc_data, dict):
            docs_for_index[doc_id] = doc_data['text']
        else:
            docs_for_index[doc_id] = doc_data
    
    index = build_index(docs_for_index, vector_compare, use_tfidf=True)
    
    # Build advanced positional index
    positional_index = build_positional_index(documents, use_tfidf=True)
    
    indexing_time = time.time() - indexing_start
    
    # Update pipeline metrics
    if pipeline.metrics_history:
        pipeline.metrics_history[-1].indexing_time = indexing_time
        pipeline.metrics_history[-1].index_size_bytes = len(str(index))
    
    # Save index
    save_index(index, documents, documents_words, INDEX_FILE)
    
    # Create version snapshot
    if version_manager:
        try:
            version_id = version_manager.create_version(
                index, documents, documents_words,
                description=f"Rebuild with {len(documents)} documents"
            )
            print(f"Index version created: {version_id}")
        except Exception as e:
            print(f"Warning: Could not create index version: {e}")
    
    print(f"Index rebuilt with {len(documents)} documents (pipeline: {pipeline_metrics.docs_ingested} ingested, {pipeline_metrics.docs_duplicate} duplicates skipped)")


def initialize_engine(documents_dir: str = None, force_rebuild: bool = False):
    """Initialize or load the search engine."""
    global index, positional_index, documents, documents_words
    
    # Ensure directories exist
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    # Try to load existing index
    if not force_rebuild and os.path.exists(INDEX_FILE):
        try:
            loaded_index, loaded_docs, loaded_words = load_index(INDEX_FILE)
            # Verify the loaded data is valid
            if loaded_index and loaded_docs and len(loaded_docs) > 0 and len(loaded_index) == len(loaded_docs):
                index = loaded_index
                documents = loaded_docs
                documents_words = loaded_words
                # Rebuild positional index from loaded documents
                from search_engine.core.advanced_index import build_positional_index
                positional_index = build_positional_index(documents, use_tfidf=True)
                print(f"Loaded index with {len(documents)} documents")
                print(f"Vocabulary size: {len(positional_index.vocabulary)} unique terms")
                return
            else:
                print("Loaded index is empty, invalid, or out of sync. Rebuilding...")
        except Exception as e:
            print(f"Error loading index: {e}. Rebuilding...")
    
    # Check for documents directory first
    if documents_dir is None:
        documents_dir = DOCUMENTS_DIR
    
    # Load documents from directory if it exists and has files
    if os.path.exists(documents_dir):
        print(f"Loading documents from {documents_dir}/")
        loaded_docs = load_documents_from_directory(documents_dir)
        if loaded_docs and len(loaded_docs) > 0:
            # Convert to format expected by pipeline
            documents = {}
            for doc_id, doc_data in loaded_docs.items():
                documents[doc_id] = doc_data
            rebuild_index()
            print(f"Loaded {len(documents)} documents from directory")
            return
        else:
            print(f"Documents directory exists but is empty. Creating seed documents...")
    else:
        print("No documents directory found. Creating seed documents for demonstration.")
    
    # Create seed documents (either directory doesn't exist or is empty)
    print(f"To add your own documents:")
    print(f"  1. Create a '{DOCUMENTS_DIR}/' folder")
    print(f"  2. Add .txt, .md, .pdf, or .docx files to it")
    print(f"  3. Restart the server or use the web interface to add documents")
    
    # Create seed documents with realistic content
    documents = {
        0: {
            'text': '''Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. The field has seen tremendous growth over the past decade, driven by advances in computing power, data availability, and algorithmic improvements.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to teach algorithms to make predictions. Common applications include image recognition, spam detection, and medical diagnosis.

Unsupervised learning finds hidden patterns in data without labeled examples. Clustering and dimensionality reduction are typical unsupervised learning tasks. Reinforcement learning involves training agents to make decisions through trial and error, receiving rewards or penalties for their actions.

Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns. Convolutional neural networks excel at image processing, while recurrent neural networks are well-suited for sequential data like text and time series.

The success of machine learning depends on several factors: quality and quantity of data, appropriate algorithm selection, feature engineering, and proper model evaluation. Overfitting, where models perform well on training data but poorly on new data, is a common challenge that requires careful regularization and validation techniques.''',
            'title': 'Introduction to Machine Learning',
            'path': '',
            'url': '/doc/0'
        },
        1: {
            'text': '''Python Programming Language Overview

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python has become one of the most popular programming languages worldwide.

Python's design philosophy emphasizes code readability through significant use of whitespace. Its syntax allows programmers to express concepts in fewer lines of code than would be possible in languages like C++ or Java.

The language supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python's dynamic typing and automatic memory management make it particularly accessible to beginners.

Python has an extensive standard library and a vast ecosystem of third-party packages. The Python Package Index contains over 400,000 packages covering areas such as web development, data science, machine learning, scientific computing, and automation.

Popular frameworks built on Python include Django and Flask for web development, NumPy and Pandas for data analysis, TensorFlow and PyTorch for machine learning, and Selenium for web automation. Python is also widely used in academic research, data analysis, and as a scripting language for system administration.

The language's versatility and ease of use have made it the preferred choice for many applications, from building web applications to conducting scientific research and developing artificial intelligence systems.''',
            'title': 'Python Programming Language Overview',
            'path': '',
            'url': '/doc/1'
        },
        2: {
            'text': '''Web Development Best Practices

Modern web development requires understanding multiple technologies and following best practices to create secure, performant, and maintainable applications. This guide covers essential principles for building quality web applications.

Frontend development involves HTML for structure, CSS for styling, and JavaScript for interactivity. Modern frameworks like React, Vue, and Angular help manage complex user interfaces. Responsive design ensures applications work well on various screen sizes and devices.

Backend development handles server-side logic, database interactions, and API design. RESTful APIs have become the standard for communication between frontend and backend systems. Security considerations include input validation, authentication, authorization, and protection against common vulnerabilities like SQL injection and cross-site scripting.

Database design is crucial for application performance and scalability. Relational databases like PostgreSQL and MySQL are suitable for structured data, while NoSQL databases like MongoDB excel with unstructured or semi-structured data. Proper indexing and query optimization significantly impact application performance.

Version control with Git is essential for collaborative development. Continuous integration and deployment pipelines automate testing and deployment processes. Code reviews, automated testing, and documentation improve code quality and maintainability.

Performance optimization includes minimizing HTTP requests, compressing assets, implementing caching strategies, and optimizing database queries. Monitoring and logging help identify and resolve issues in production environments.''',
            'title': 'Web Development Best Practices',
            'path': '',
            'url': '/doc/2'
        },
        3: {
            'text': '''Database Management Systems

A database management system (DBMS) is software that provides an interface for managing databases. It handles data storage, retrieval, security, and integrity while allowing multiple users to access data concurrently.

Relational database management systems (RDBMS) organize data into tables with rows and columns. SQL (Structured Query Language) is the standard language for interacting with relational databases. Popular RDBMS include MySQL, PostgreSQL, Oracle, and Microsoft SQL Server.

ACID properties ensure reliable database transactions: Atomicity guarantees all operations complete or none do, Consistency maintains data integrity, Isolation prevents concurrent transactions from interfering, and Durability ensures committed changes persist even after system failures.

Normalization is the process of organizing data to reduce redundancy and improve data integrity. The normal forms (1NF, 2NF, 3NF, BCNF) provide guidelines for database design. Denormalization may be used strategically to improve query performance at the cost of some redundancy.

Indexing improves query performance by creating data structures that allow faster data retrieval. Primary keys uniquely identify rows, foreign keys maintain referential integrity between tables, and composite indexes support queries on multiple columns.

NoSQL databases offer alternatives to relational models. Document databases store data as documents, key-value stores provide simple data models, column-family databases organize data by columns, and graph databases focus on relationships between entities.''',
            'title': 'Database Management Systems',
            'path': '',
            'url': '/doc/3'
        },
        4: {
            'text': '''Software Engineering Principles

Software engineering applies engineering principles to software development. It encompasses the entire software development lifecycle from requirements gathering to maintenance and evolution.

The software development lifecycle typically includes phases such as requirements analysis, system design, implementation, testing, deployment, and maintenance. Agile methodologies like Scrum and Kanban emphasize iterative development, collaboration, and responding to change.

Object-oriented programming principles include encapsulation, inheritance, polymorphism, and abstraction. Design patterns provide reusable solutions to common problems. SOLID principles guide object-oriented design: Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.

Code quality is maintained through practices like code reviews, automated testing, static analysis, and refactoring. Unit tests verify individual components, integration tests check component interactions, and end-to-end tests validate complete workflows.

Version control systems track changes to code over time, enabling collaboration and rollback capabilities. Branching strategies like Git Flow help manage parallel development efforts. Continuous integration automatically builds and tests code changes.

Documentation is essential for maintaining and understanding software systems. API documentation, code comments, architecture diagrams, and user manuals help different stakeholders understand the system. Technical debt, accumulated shortcuts and compromises, must be managed to maintain long-term code quality.''',
            'title': 'Software Engineering Principles',
            'path': '',
            'url': '/doc/4'
        },
        5: {
            'text': '''Cloud Computing Fundamentals

Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, software, and analytics. Instead of owning physical infrastructure, organizations access technology services on demand from cloud providers.

The three main service models are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). IaaS provides virtualized computing resources, PaaS offers development platforms, and SaaS delivers complete software applications.

Deployment models include public clouds (services available to the general public), private clouds (dedicated to a single organization), hybrid clouds (combination of public and private), and multi-cloud (using multiple cloud providers).

Major cloud providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform. Each offers extensive services covering computing, storage, databases, networking, machine learning, and analytics.

Cloud computing benefits include cost reduction through pay-as-you-go pricing, scalability to handle varying workloads, flexibility to access services from anywhere, and reliability through redundant infrastructure. Security and compliance remain important considerations when moving to the cloud.

DevOps practices integrate development and operations, enabling faster software delivery through automation, continuous integration, continuous deployment, and infrastructure as code. Containerization with Docker and orchestration with Kubernetes have become standard for cloud-native applications.''',
            'title': 'Cloud Computing Fundamentals',
            'path': '',
            'url': '/doc/5'
        },
        6: {
            'text': '''Cybersecurity Essentials

Cybersecurity protects systems, networks, and data from digital attacks. As organizations become increasingly digital, understanding security threats and defenses is crucial for protecting sensitive information and maintaining business continuity.

Common threats include malware (viruses, worms, trojans), phishing attacks that trick users into revealing credentials, ransomware that encrypts data until payment is made, and denial-of-service attacks that overwhelm systems with traffic.

Authentication verifies user identity through methods like passwords, multi-factor authentication, biometrics, and security tokens. Authorization determines what authenticated users can access. The principle of least privilege grants users only the minimum permissions necessary.

Encryption protects data at rest and in transit. Symmetric encryption uses the same key for encryption and decryption, while asymmetric encryption uses public-private key pairs. SSL/TLS protocols secure web communications.

Network security includes firewalls that filter traffic, intrusion detection systems that monitor for suspicious activity, and virtual private networks that create secure connections over public networks. Regular security audits and penetration testing identify vulnerabilities.

Security best practices include keeping software updated with security patches, using strong passwords and password managers, implementing regular backups, training users to recognize threats, and having incident response plans. Compliance with regulations like GDPR and HIPAA may also be required depending on the data being handled.''',
            'title': 'Cybersecurity Essentials',
            'path': '',
            'url': '/doc/6'
        },
        }
    
    # Prepare documents_words for TF-IDF
    documents_words = []
    for doc_id in sorted(documents.keys()):
        doc_data = documents[doc_id]
        text = doc_data['text'] if isinstance(doc_data, dict) else doc_data
        words = get_words(text.lower())
        documents_words.append(words)
    
    # Build index
    # Convert to simple dict for indexing (backward compatibility)
    docs_for_index = {}
    for doc_id, doc_data in documents.items():
        if isinstance(doc_data, dict):
            docs_for_index[doc_id] = doc_data['text']
        else:
            docs_for_index[doc_id] = doc_data
    
    index = build_index(docs_for_index, vector_compare, use_tfidf=True)
    
    # Build advanced positional index
    from search_engine.core.advanced_index import build_positional_index
    positional_index = build_positional_index(documents, use_tfidf=True)
    
    # Save index
    save_index(index, documents, documents_words, INDEX_FILE)
    
    print(f"Indexed {len(documents)} seed documents")
    print(f"Vocabulary size: {len(positional_index.vocabulary) if positional_index else 0} unique terms")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_document(file_path, filename):
    """
    Parse various document formats and extract text.
    
    Args:
        file_path: Path to the file
        filename: Original filename
        
    Returns:
        Extracted text content
    """
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if not ext:
        raise ValueError("File has no extension")
    
    try:
        if ext == 'pdf':
            if not PDF_SUPPORT:
                raise ValueError("PDF support not available. Install PyPDF2: pip install PyPDF2")
            
            text = ""
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    num_pages = len(pdf_reader.pages)
                    
                    if num_pages == 0:
                        raise ValueError("PDF file appears to be empty or corrupted")
                    
                    for i, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as page_error:
                            print(f"Warning: Could not extract text from page {i+1}: {page_error}")
                            continue
                    
                    if not text.strip():
                        raise ValueError("Could not extract any text from PDF. The PDF might be image-based or encrypted.")
                    
            except PyPDF2.errors.PdfReadError as e:
                raise ValueError(f"PDF file is corrupted or cannot be read: {str(e)}")
            except Exception as e:
                raise ValueError(f"Error reading PDF: {str(e)}")
            
            return text.strip()
        
        elif ext in ['docx']:
            if not DOCX_SUPPORT:
                raise ValueError("DOCX support not available. Install python-docx: pip install python-docx")
            
            try:
                doc = DocxDocument(file_path)
                paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
                text = "\n".join(paragraphs)
                
                if not text.strip():
                    raise ValueError("Document appears to be empty")
                
                return text.strip()
            except Exception as e:
                raise ValueError(f"Error reading DOCX file: {str(e)}")
        
        elif ext == 'doc':
            raise ValueError("Old .doc format is not supported. Please convert to .docx or .pdf")
        
        elif ext in ['txt', 'md', 'text']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if not text:
                        raise ValueError("Text file appears to be empty")
                    return text
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read().strip()
                except Exception as e:
                    raise ValueError(f"Could not read text file with UTF-8 or Latin-1 encoding: {str(e)}")
        
        else:
            raise ValueError(f"Unsupported file type: .{ext}")
    
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        raise ValueError(f"Error parsing document: {str(e)}")


@app.route('/')
def index_page():
    """Main search page."""
    return render_template('index.html', doc_count=len(documents) if documents else 0, request=request)


@app.route('/manage')
def manage_documents():
    """Document management page."""
    doc_list = []
    if documents:
        for doc_id, doc_data in documents.items():
            if isinstance(doc_data, dict):
                text = doc_data.get('text', '')
                preview = text[:150] + '...' if len(text) > 150 else text
                doc_list.append({
                    'id': doc_id,
                    'title': doc_data.get('title', f'Document {doc_id}'),
                    'path': doc_data.get('path', ''),
                    'text_preview': preview,
                    'word_count': len(text.split()) if text else 0
                })
    return render_template('manage.html', documents=doc_list, doc_count=len(documents) if documents else 0, request=request)


@app.route('/documents')
def list_documents():
    """List all documents page."""
    doc_list = []
    if documents:
        for doc_id, doc_data in documents.items():
            if isinstance(doc_data, dict):
                text = doc_data.get('text', '')
                preview = text[:200] + '...' if len(text) > 200 else text
                doc_list.append({
                    'id': doc_id,
                    'title': doc_data.get('title', f'Document {doc_id}'),
                    'path': doc_data.get('path', ''),
                    'text_preview': preview,
                    'word_count': len(text.split()) if text else 0,
                    'char_count': len(text)
                })
    return render_template('documents.html', documents=doc_list, doc_count=len(documents) if documents else 0, request=request)


@app.route('/api/add-document', methods=['POST'])
def api_add_document():
    """Add a new document via API."""
    global documents
    
    data = request.json
    title = data.get('title', 'Untitled Document')
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Document text cannot be empty'}), 400
    
    # Get next document ID
    if documents:
        next_id = max(documents.keys()) + 1
    else:
        next_id = 0
        documents = {}
    
    # Add document
    documents[next_id] = {
        'text': text,
        'title': title,
        'path': '',
        'url': f'/doc/{next_id}'
    }
    
    # Rebuild index
    rebuild_index()
    
    return jsonify({
        'success': True,
        'doc_id': next_id,
        'message': f'Document "{title}" added successfully'
    })


@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    """Upload a file and add it as a document."""
    global documents
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'unknown'
        return jsonify({'error': f'Invalid file type: .{ext}. Allowed: .txt, .md, .text, .pdf, .docx'}), 400
    
    try:
        # Save uploaded file temporarily
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        
        # Save the file first
        file.save(filepath)
        
        # Check file extension and verify support
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext == 'pdf' and not PDF_SUPPORT:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': 'PDF support not available. Please install PyPDF2: pip install PyPDF2'
            }), 400
        
        if ext == 'docx' and not DOCX_SUPPORT:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': 'DOCX support not available. Please install python-docx: pip install python-docx'
            }), 400
        
        # Parse document based on file type
        try:
            text = parse_document(filepath, filename)
        except ValueError as parse_error:
            # Clean up the saved file if parsing fails
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': str(parse_error)}), 400
        except Exception as parse_error:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': f'Failed to parse document: {str(parse_error)}'}), 400
        
        if not text or not text.strip():
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Document appears to be empty or could not extract text'}), 400
        
        title = Path(file.filename).stem
        
        # Get next document ID
        if documents:
            next_id = max(documents.keys()) + 1
        else:
            next_id = 0
            documents = {}
        
        # For text files, ensure UTF-8 encoding is saved
        if ext in ['txt', 'md', 'text']:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Add document
        documents[next_id] = {
            'text': text,
            'title': title,
            'path': filepath,
            'url': f'/doc/{next_id}'
        }
        
        # Rebuild index
        rebuild_index()
        
        return jsonify({
            'success': True,
            'doc_id': next_id,
            'message': f'File "{filename}" uploaded and indexed successfully ({len(text)} characters extracted)'
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Upload error: {error_msg}")
        print(traceback_str)
        # Clean up file if it was saved
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({'error': f'Error processing file: {error_msg}'}), 500


@app.route('/api/delete-document/<int:doc_id>', methods=['DELETE'])
def api_delete_document(doc_id):
    """Delete a document."""
    
    if documents is None or doc_id not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    # Delete file if it exists
    doc_data = documents[doc_id]
    if isinstance(doc_data, dict) and doc_data.get('path') and os.path.exists(doc_data['path']):
        try:
            os.remove(doc_data['path'])
        except:
            pass
    
    # Remove document
    del documents[doc_id]
    
    # Rebuild index
    rebuild_index()
    
    return jsonify({'success': True, 'message': 'Document deleted successfully'})


@app.route('/api/search', methods=['GET', 'POST'])
def api_search():
    """API endpoint for search with advanced indexing and caching."""
    global documents_words
    
    if (positional_index is None and index is None) or documents is None:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    query = request.args.get('q', '') or (request.json.get('q', '') if request.is_json else '')
    
    if not query:
        return jsonify({'results': [], 'query': query, 'count': 0})
    
    query_start = time.time()
    cache_hit = False
    
    try:
        # Use advanced positional index if available
        if positional_index:
            # Check cache first
            cached_results = query_cache.get(query.lower())
            if cached_results:
                cache_hit = True
                results_data = []
                for score, doc_id, term_positions in cached_results:
                    doc_data = documents[doc_id]
                    if isinstance(doc_data, dict):
                        text = doc_data['text']
                        title = doc_data.get('title', f'Document {doc_id}')
                        highlighted = highlight_with_positions(text, term_positions, max_length=300)
                    else:
                        text = doc_data
                        title = f'Document {doc_id}'
                        highlighted = highlight_with_positions(text, term_positions, max_length=300)
                    
                    results_data.append({
                        'score': score,
                        'doc_id': doc_id,
                        'title': title,
                        'text': text[:300] + '...' if len(text) > 300 else text,
                        'highlighted_text': highlighted,
                        'path': doc_data.get('path', '') if isinstance(doc_data, dict) else '',
                        'url': f'/doc/{doc_id}'
                    })
                
                latency_ms = (time.time() - query_start) * 1000
                metrics_collector.record_query(query, len(results_data), latency_ms, cache_hit=True)
                
                return jsonify({
                    'results': results_data,
                    'query': query,
                    'count': len(results_data),
                    'cache_hit': True
                })
            
            # Perform search with positional index
            search_results = search_with_positional_index(
                query, positional_index, documents, top_k=20, cache=query_cache
            )
            
            # Format results
            results_data = []
            for score, doc_id, term_positions in search_results:
                doc_data = documents[doc_id]
                if isinstance(doc_data, dict):
                    text = doc_data['text']
                    title = doc_data.get('title', f'Document {doc_id}')
                    highlighted = highlight_with_positions(text, term_positions, max_length=300)
                    path = doc_data.get('path', '')
                else:
                    text = doc_data
                    title = f'Document {doc_id}'
                    highlighted = highlight_with_positions(text, term_positions, max_length=300)
                    path = ''
                
                results_data.append({
                    'score': score,
                    'doc_id': doc_id,
                    'title': title,
                    'text': text[:300] + '...' if len(text) > 300 else text,
                    'highlighted_text': highlighted,
                    'path': path,
                    'url': f'/doc/{doc_id}'
                })
            
            latency_ms = (time.time() - query_start) * 1000
            metrics_collector.record_query(query, len(results_data), latency_ms, cache_hit=False)
            
            return jsonify({
                'results': results_data,
                'query': query,
                'count': len(results_data),
                'cache_hit': False
            })
        else:
            # Fallback to legacy search
            if documents_words is None:
                from search_engine.core.vector_search import get_words
                documents_words = []
                for doc_id in sorted(documents.keys()):
                    doc_data = documents[doc_id]
                    text = doc_data['text'] if isinstance(doc_data, dict) else doc_data
                    words = get_words(text.lower())
                    documents_words.append(words)
            
            results = search_with_highlighting(
                query, index, documents, vector_compare,
                top_k=20, use_tfidf=True, documents_words=documents_words
            )
            
            latency_ms = (time.time() - query_start) * 1000
            metrics_collector.record_query(query, len(results), latency_ms, cache_hit=False)
            
            return jsonify({
                'results': results,
                'query': query,
                'count': len(results)
            })
    except Exception as e:
        import traceback
        latency_ms = (time.time() - query_start) * 1000
        metrics_collector.record_query(query, 0, latency_ms, cache_hit=False)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get search engine statistics."""
    vocab_size = len(positional_index.vocabulary) if positional_index else 0
    index_size = len(str(index)) if index else 0
    
    system_metrics = metrics_collector.get_system_metrics(
        total_documents=len(documents) if documents else 0,
        index_size_bytes=index_size,
        vocabulary_size=vocab_size,
        cache_hit_rate=query_cache.hit_rate()
    )
    
    return jsonify({
        'document_count': len(documents) if documents else 0,
        'indexed': index is not None,
        'index_file': INDEX_FILE,
        'index_size': len(index) if index else 0,
        'vocabulary_size': vocab_size,
        'cache_hit_rate': query_cache.hit_rate(),
        'cache_hits': query_cache.hits,
        'cache_misses': query_cache.misses,
        'total_queries': system_metrics.total_queries,
        'avg_latency_ms': system_metrics.avg_query_latency_ms,
        'p50_latency_ms': system_metrics.p50_latency_ms,
        'p95_latency_ms': system_metrics.p95_latency_ms
    })


@app.route('/metrics')
def metrics_dashboard():
    """Metrics and monitoring dashboard."""
    vocab_size = len(positional_index.vocabulary) if positional_index else 0
    index_size = len(str(index)) if index else 0
    
    system_metrics = metrics_collector.get_system_metrics(
        total_documents=len(documents) if documents else 0,
        index_size_bytes=index_size,
        vocabulary_size=vocab_size,
        cache_hit_rate=query_cache.hit_rate()
    )
    
    query_stats = metrics_collector.get_query_stats()
    recent_queries = metrics_collector.get_recent_queries(20)
    
    # Get latest pipeline metrics
    latest_pipeline = pipeline.metrics_history[-1] if pipeline.metrics_history else None
    
    return render_template('metrics.html',
                         system_metrics=system_metrics,
                         query_stats=query_stats,
                         recent_queries=recent_queries,
                         pipeline_metrics=latest_pipeline,
                         request=request)


@app.route('/api/export-metrics', methods=['GET'])
def api_export_metrics():
    """Export metrics as JSON or CSV."""
    format_type = request.args.get('format', 'json').lower()
    
    vocab_size = len(positional_index.vocabulary) if positional_index else 0
    index_size = len(str(index)) if index else 0
    
    system_metrics = metrics_collector.get_system_metrics(
        total_documents=len(documents) if documents else 0,
        index_size_bytes=index_size,
        vocabulary_size=vocab_size,
        cache_hit_rate=query_cache.hit_rate()
    )
    
    query_stats = metrics_collector.get_query_stats()
    recent_queries = metrics_collector.get_recent_queries(100)
    latest_pipeline = pipeline.metrics_history[-1] if pipeline.metrics_history else None
    
    if format_type == 'csv':
        import csv
        from io import StringIO
        from flask import Response
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write system metrics
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Documents', system_metrics.total_documents])
        writer.writerow(['Total Queries', system_metrics.total_queries])
        writer.writerow(['Cache Hit Rate', f"{system_metrics.cache_hit_rate:.2%}"])
        writer.writerow(['Avg Latency (ms)', f"{system_metrics.avg_query_latency_ms:.2f}"])
        writer.writerow(['P50 Latency (ms)', f"{system_metrics.p50_latency_ms:.2f}"])
        writer.writerow(['P95 Latency (ms)', f"{system_metrics.p95_latency_ms:.2f}"])
        writer.writerow(['Vocabulary Size', system_metrics.vocabulary_size])
        
        # Write query stats
        writer.writerow([])
        writer.writerow(['Query Statistics'])
        writer.writerow(['Total', query_stats.get('total', 0)])
        writer.writerow(['Cache Hits', query_stats.get('cache_hits', 0)])
        writer.writerow(['Cache Misses', query_stats.get('cache_misses', 0)])
        
        # Write recent queries
        if recent_queries:
            writer.writerow([])
            writer.writerow(['Recent Queries'])
            writer.writerow(['Query', 'Results', 'Latency (ms)', 'Cache Hit', 'Timestamp'])
            for q in recent_queries:
                writer.writerow([q.query, q.result_count, f"{q.latency_ms:.2f}", 
                               'Yes' if q.cache_hit else 'No', q.timestamp])
        
        # Write pipeline metrics
        if latest_pipeline:
            writer.writerow([])
            writer.writerow(['Pipeline Metrics'])
            writer.writerow(['Run ID', latest_pipeline.run_id])
            writer.writerow(['Timestamp', latest_pipeline.timestamp])
            writer.writerow(['Docs Ingested', latest_pipeline.docs_ingested])
            writer.writerow(['Docs Skipped', latest_pipeline.docs_skipped])
            writer.writerow(['Duplicates', latest_pipeline.docs_duplicate])
            writer.writerow(['Tokens Processed', latest_pipeline.tokens_processed])
            writer.writerow(['Unique Terms', latest_pipeline.unique_terms])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=metrics_export.csv'}
        )
    else:
        # JSON format
        data = {
            'system_metrics': {
                'total_documents': system_metrics.total_documents,
                'total_queries': system_metrics.total_queries,
                'cache_hit_rate': system_metrics.cache_hit_rate,
                'avg_latency_ms': system_metrics.avg_query_latency_ms,
                'p50_latency_ms': system_metrics.p50_latency_ms,
                'p95_latency_ms': system_metrics.p95_latency_ms,
                'vocabulary_size': system_metrics.vocabulary_size
            },
            'query_stats': query_stats,
            'recent_queries': [
                {
                    'query': q.query,
                    'result_count': q.result_count,
                    'latency_ms': q.latency_ms,
                    'cache_hit': q.cache_hit,
                    'timestamp': q.timestamp
                }
                for q in recent_queries
            ],
            'pipeline_metrics': latest_pipeline.to_dict() if latest_pipeline else None,
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(data)


@app.route('/api/rebuild-index', methods=['POST'])
def api_rebuild_index():
    """Force rebuild the search index."""
    
    try:
        initialize_engine(force_rebuild=True)
        return jsonify({
            'success': True,
            'message': f'Index rebuilt successfully with {len(documents) if documents else 0} documents',
            'version_id': version_manager.get_current_version() if version_manager else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/index-versions', methods=['GET'])
def api_list_versions():
    """List all index versions."""
    if not version_manager:
        return jsonify({'error': 'Version manager not initialized'}), 500
    
    versions = version_manager.list_versions()
    current = version_manager.get_current_version()
    
    return jsonify({
        'versions': versions,
        'current_version': current
    })


@app.route('/api/index-versions/<version_id>', methods=['POST'])
def api_load_version(version_id):
    """Load a specific index version."""
    global index, documents, documents_words, positional_index
    
    if not version_manager:
        return jsonify({'error': 'Version manager not initialized'}), 500
    
    try:
        index, documents, documents_words = version_manager.load_version(version_id)
        
        # Rebuild positional index from loaded documents
        from search_engine.core.advanced_index import build_positional_index
        positional_index = build_positional_index(documents, use_tfidf=True)
        
        # Save to main index file
        save_index(index, documents, documents_words, INDEX_FILE)
        
        return jsonify({
            'success': True,
            'message': f'Loaded version {version_id}',
            'version_id': version_id,
            'document_count': len(documents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/index-versions/<version_id>', methods=['DELETE'])
def api_delete_version(version_id):
    """Delete an index version."""
    if not version_manager:
        return jsonify({'error': 'Version manager not initialized'}), 500
    
    try:
        success = version_manager.delete_version(version_id)
        if success:
            return jsonify({
                'success': True,
                'message': f'Version {version_id} deleted'
            })
        else:
            return jsonify({
                'error': 'Cannot delete current version or version not found'
            }), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def render_rich_content(text: str, path: str = '') -> str:
    """
    Render text as rich content (Markdown or HTML).
    Supports images, links, headers, code blocks, tables, and more.
    
    Args:
        text: Document text content
        path: File path to determine format
        
    Returns:
        Rendered HTML content
    """
    if not text:
        return ''
    
    # Check if it's Markdown by file extension
    is_markdown_by_ext = path.endswith('.md') or path.endswith('.markdown')
    
    # Also check if text contains Markdown syntax (for pasted content)
    # Check for common Markdown patterns
    markdown_indicators = ['# ', '## ', '### ', '```', '![', '](', '* ', '- ', '|', '> ', '**', '__']
    has_markdown_syntax = any(indicator in text for indicator in markdown_indicators)
    # Also check for image syntax specifically (more lenient)
    has_image_syntax = '![' in text and '](' in text
    is_markdown = is_markdown_by_ext or has_markdown_syntax or has_image_syntax
    
    # Check if it looks like HTML
    is_html = text.strip().startswith('<') and ('<html' in text.lower() or '<body' in text.lower() or '<p>' in text.lower() or '<div>' in text.lower())
    
    try:
        if is_markdown:
            import markdown
            # Convert Markdown to HTML with full feature support
            html = markdown.markdown(text, extensions=['extra', 'codehilite', 'tables', 'fenced_code', 'attr_list'])
            # Sanitize to prevent XSS but allow rich content
            import bleach
            allowed_tags = bleach.sanitizer.ALLOWED_TAGS | {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                                                           'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
                                                           'img', 'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
                                                           'strong', 'em', 'b', 'i', 'u', 'br', 'hr', 'div', 'span',
                                                           'del', 'ins', 'sub', 'sup'}
            allowed_attributes = {
                'img': ['src', 'alt', 'title', 'width', 'height', 'class', 'style'],
                'a': ['href', 'title', 'target', 'rel'],
                'code': ['class'],
                'div': ['class', 'style'],
                'span': ['class', 'style'],
                'table': ['class', 'style'],
                'td': ['align', 'colspan', 'rowspan'],
                'th': ['align', 'colspan', 'rowspan']
            }
            # Allow https and http protocols for images and links
            allowed_protocols = ['http', 'https', 'data']
            return bleach.clean(html, tags=allowed_tags, attributes=allowed_attributes, protocols=allowed_protocols)
        elif is_html:
            # Sanitize HTML
            import bleach
            allowed_tags = bleach.sanitizer.ALLOWED_TAGS | {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                                           'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
                                                           'img', 'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
                                                           'strong', 'em', 'b', 'i', 'u', 'br', 'hr', 'div', 'span',
                                                           'del', 'ins', 'sub', 'sup'}
            allowed_attributes = {
                'img': ['src', 'alt', 'title', 'width', 'height', 'class', 'style'],
                'a': ['href', 'title', 'target', 'rel'],
                'div': ['class', 'style'],
                'span': ['class', 'style'],
                'table': ['class', 'style'],
                'td': ['align', 'colspan', 'rowspan'],
                'th': ['align', 'colspan', 'rowspan']
            }
            # Allow https and http protocols for images and links
            allowed_protocols = ['http', 'https', 'data']
            return bleach.clean(text, tags=allowed_tags, attributes=allowed_attributes, protocols=allowed_protocols)
        else:
            # Plain text - convert to HTML paragraphs
            paragraphs = text.split('\n\n')
            html_paragraphs = []
            for para in paragraphs:
                if para.strip():
                    # Convert line breaks to <br>
                    para_html = para.replace('\n', '<br>')
                    html_paragraphs.append(f'<p>{para_html}</p>')
            return '\n'.join(html_paragraphs) if html_paragraphs else f'<p>{text}</p>'
    except ImportError:
        # Fallback if markdown/bleach not installed
        paragraphs = text.split('\n\n')
        html_paragraphs = []
        for para in paragraphs:
            if para.strip():
                para_html = para.replace('\n', '<br>')
                html_paragraphs.append(f'<p>{para_html}</p>')
        return '\n'.join(html_paragraphs) if html_paragraphs else f'<p>{text}</p>'


@app.route('/doc/<int:doc_id>')
def view_document(doc_id):
    """View a specific document with rich content rendering."""
    if documents is None or doc_id not in documents:
        return "Document not found", 404
    
    doc_data = documents[doc_id]
    if isinstance(doc_data, dict):
        text = doc_data.get('text', '')
        path = doc_data.get('path', '')
        rendered_content = render_rich_content(text, path)
        return render_template('document.html', 
                             doc_id=doc_id,
                             title=doc_data.get('title', f'Document {doc_id}'),
                             text=text,  # Keep original for editing
                             rendered_content=rendered_content,  # Rendered HTML
                             path=path,
                             request=request)
    else:
        rendered_content = render_rich_content(doc_data, '')
        return render_template('document.html',
                             doc_id=doc_id,
                             title=f'Document {doc_id}',
                             text=doc_data,
                             rendered_content=rendered_content,
                             path='',
                             request=request)


if __name__ == '__main__':
    # Create documents directory if it doesn't exist
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    # Initialize on startup
    # Looks for documents/ folder, or uses sample documents
    initialize_engine()
    
    # Try different ports if 5000 is in use
    port = 5000
    import socket
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
    print(f"Indexed {len(documents) if documents else 0} documents")
    
    # Check for optional dependencies
    if not PDF_SUPPORT:
        print("⚠️  PDF support not available. Install: pip install PyPDF2")
    if not DOCX_SUPPORT:
        print("⚠️  DOCX support not available. Install: pip install python-docx")
    
    print(f"Open http://localhost:{port} in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)

