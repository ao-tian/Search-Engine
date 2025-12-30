"""
Index versioning and rollback functionality.
Allows saving and restoring index snapshots.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from search_engine.core.vector_search import save_index, load_index


class IndexVersionManager:
    """Manages index versions and rollback."""
    
    def __init__(self, versions_dir: str = None):
        if versions_dir is None:
            # Get project root (3 levels up from this file: utils -> search_engine -> src -> root)
            project_root = Path(__file__).parent.parent.parent.parent
            versions_dir = str(project_root / 'index_versions')
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        self.version_metadata_file = self.versions_dir / 'versions.json'
        self.versions_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load version metadata."""
        if self.version_metadata_file.exists():
            try:
                with open(self.version_metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {'versions': [], 'current_version': None}
        return {'versions': [], 'current_version': None}
    
    def _save_metadata(self):
        """Save version metadata."""
        with open(self.version_metadata_file, 'w') as f:
            json.dump(self.versions_metadata, f, indent=2)
    
    def create_version(
        self,
        index: Dict,
        documents: Dict,
        documents_words: List[List[str]],
        description: str = ''
    ) -> str:
        """
        Create a new index version snapshot.
        
        Args:
            index: The search index
            documents: Dictionary of documents
            documents_words: List of word lists
            description: Optional description for this version
            
        Returns:
            Version ID string
        """
        # Generate version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"v{len(self.versions_metadata.get('versions', [])) + 1}_{timestamp}"
        
        # Create version directory
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save index files
        index_file = version_dir / 'search_index.json'
        save_index(index, documents, documents_words, str(index_file))
        
        # Save metadata
        version_info = {
            'version_id': version_id,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'document_count': len(documents),
            'index_file': str(index_file)
        }
        
        if 'versions' not in self.versions_metadata:
            self.versions_metadata['versions'] = []
        
        self.versions_metadata['versions'].append(version_info)
        self.versions_metadata['current_version'] = version_id
        self._save_metadata()
        
        return version_id
    
    def list_versions(self) -> List[Dict]:
        """List all available versions."""
        return self.versions_metadata.get('versions', [])
    
    def get_current_version(self) -> Optional[str]:
        """Get current version ID."""
        return self.versions_metadata.get('current_version')
    
    def load_version(
        self,
        version_id: str
    ) -> Tuple[Dict, Dict, List[List[str]]]:
        """
        Load a specific version of the index.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            Tuple of (index, documents, documents_words)
        """
        # Find version
        version_info = None
        for v in self.versions_metadata.get('versions', []):
            if v['version_id'] == version_id:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version_id} not found")
        
        index_file = version_info['index_file']
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        # Load index
        index, documents, documents_words = load_index(index_file)
        
        # Update current version
        self.versions_metadata['current_version'] = version_id
        self._save_metadata()
        
        return index, documents, documents_words
    
    def delete_version(self, version_id: str) -> bool:
        """
        Delete a version (cannot delete current version).
        
        Args:
            version_id: Version ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if version_id == self.versions_metadata.get('current_version'):
            return False  # Cannot delete current version
        
        # Find and remove version
        versions = self.versions_metadata.get('versions', [])
        for i, v in enumerate(versions):
            if v['version_id'] == version_id:
                # Delete version directory
                version_dir = self.versions_dir / version_id
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                
                # Remove from metadata
                versions.pop(i)
                self._save_metadata()
                return True
        
        return False
    
    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """Get information about a specific version."""
        for v in self.versions_metadata.get('versions', []):
            if v['version_id'] == version_id:
                return v
        return None

