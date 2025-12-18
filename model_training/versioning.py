import hashlib
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
import yaml

class ModelVersioner:
    """
    Handles versioning, hashing, and packaging of ML models
    """
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.version_file = Path(self.config['model']['storage_path']) / 'versions.json'
        self.version_file.parent.mkdir(exist_ok=True)
        
        if not self.version_file.exists():
            with open(self.version_file, 'w') as f:
                json.dump({'versions': []}, f)
    
    def calculate_file_hash(self, file_path, algorithm='sha256'):
        """Calculate cryptographic hash of a file"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    def create_model_package(self, model_path, metadata):
        """Create a reproducible package of the model"""
        package_path = model_path.replace('.pt', '.zip')
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model file
            zipf.write(model_path, os.path.basename(model_path))
            
            # Add metadata
            metadata_str = json.dumps(metadata, indent=2)
            zipf.writestr('metadata.json', metadata_str)
            
            # Add requirements if available
            req_path = 'requirements.txt'
            if os.path.exists(req_path):
                zipf.write(req_path, 'requirements.txt')
            
            # Add training script snapshot
            training_script = 'model_training/train.py'
            if os.path.exists(training_script):
                zipf.write(training_script, 'training_script.py')
        
        return package_path
    
    def version_model(self, model_path, metadata, model_hash):
        """Create a new version entry for the model"""
        with open(self.version_file, 'r') as f:
            versions_data = json.load(f)
        
        # Get next version number
        current_versions = versions_data['versions']
        if current_versions:
            last_version = current_versions[-1]['version']
            # Parse version string like "v1.2.3"
            major, minor, patch = map(int, last_version[1:].split('.'))
            if metadata.get('major_change', False):
                major += 1
                minor = 0
                patch = 0
            elif metadata.get('minor_change', False):
                minor += 1
                patch = 0
            else:
                patch += 1
            new_version = f"v{major}.{minor}.{patch}"
        else:
            new_version = "v1.0.0"
        
        # Create version entry
        version_entry = {
            'version': new_version,
            'model_hash': model_hash,
            'file_hash': self.calculate_file_hash(model_path),
            'model_path': model_path,
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'parent_version': current_versions[-1]['version'] if current_versions else None
        }
        
        # Create reproducible package
        package_path = self.create_model_package(model_path, metadata)
        version_entry['package_path'] = package_path
        version_entry['package_hash'] = self.calculate_file_hash(package_path)
        
        # Add to versions list
        versions_data['versions'].append(version_entry)
        
        # Save updated versions
        with open(self.version_file, 'w') as f:
            json.dump(versions_data, f, indent=2)
        
        return version_entry
    
    def verify_model(self, model_path, expected_hash):
        """Verify model integrity by comparing hashes"""
        actual_hash = self.calculate_file_hash(model_path)
        return actual_hash == expected_hash, actual_hash
    
    def get_version_history(self, limit=10):
        """Get version history"""
        with open(self.version_file, 'r') as f:
            versions_data = json.load(f)
        
        return versions_data['versions'][-limit:]
    
    def rollback_to_version(self, target_version):
        """Rollback to a previous version"""
        with open(self.version_file, 'r') as f:
            versions_data = json.load(f)
        
        versions = versions_data['versions']
        target_idx = None
        
        for i, version in enumerate(versions):
            if version['version'] == target_version:
                target_idx = i
                break
        
        if target_idx is None:
            raise ValueError(f"Version {target_version} not found")
        
        # Return target version info
        return versions[target_idx]