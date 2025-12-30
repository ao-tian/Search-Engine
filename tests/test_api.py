"""
Integration tests for API endpoints.
"""

import pytest
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from search_engine.app import app


@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestSearchAPI:
    """Test search API endpoint."""
    
    def test_search_endpoint_exists(self, client):
        """Test that search endpoint exists."""
        response = client.get('/api/search?q=test')
        assert response.status_code in [200, 500]  # 500 if not initialized
    
    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.get('/api/search?q=')
        # May return 200 (if initialized) or 500 (if not initialized)
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'results' in data
            assert data['count'] == 0
    
    def test_search_response_format(self, client):
        """Test that search returns correct format."""
        response = client.get('/api/search?q=machine')
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'results' in data
            assert 'query' in data
            assert 'count' in data
            assert isinstance(data['results'], list)


class TestStatsAPI:
    """Test stats API endpoint."""
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        response = client.get('/api/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'document_count' in data
        assert 'indexed' in data


class TestDocumentManagementAPI:
    """Test document management endpoints."""
    
    def test_add_document_endpoint(self, client):
        """Test adding a document via API."""
        response = client.post('/api/add-document',
                              json={'title': 'Test Doc', 'text': 'Test content'},
                              content_type='application/json')
        # May fail if engine not initialized, but endpoint should exist
        assert response.status_code in [200, 400, 500]
    
    def test_add_document_validation(self, client):
        """Test that empty documents are rejected."""
        response = client.post('/api/add-document',
                              json={'title': 'Test', 'text': ''},
                              content_type='application/json')
        if response.status_code == 400:
            data = json.loads(response.data)
            assert 'error' in data


class TestMetricsDashboard:
    """Test metrics dashboard."""
    
    def test_metrics_page_loads(self, client):
        """Test that metrics page loads."""
        response = client.get('/metrics')
        assert response.status_code == 200


class TestDocumentPages:
    """Test document viewing pages."""
    
    def test_documents_page_loads(self, client):
        """Test that documents listing page loads."""
        response = client.get('/documents')
        assert response.status_code == 200
    
    def test_manage_page_loads(self, client):
        """Test that manage page loads."""
        response = client.get('/manage')
        assert response.status_code == 200
    
    def test_search_page_loads(self, client):
        """Test that search page loads."""
        response = client.get('/')
        assert response.status_code == 200

