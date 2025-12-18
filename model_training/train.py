import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import hashlib
import json
import yaml
from datetime import datetime
import os
from .versioning import ModelVersioner
import mlflow
import mlflow.pytorch

class ReproducibleModelTrainer:
    """
    Trainer for creating reproducible ML models with deterministic behavior
    """
    
    def __init__(self, config_path='config.yaml', seed=42):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set all random seeds for reproducibility
        self.set_seeds(seed)
        self.versioner = ModelVersioner()
        self.model_storage = self.config['model']['storage_path']
        os.makedirs(self.model_storage, exist_ok=True)
        
    def set_seeds(self, seed):
        """Set all random seeds for complete reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def create_model_hash(self, model_state, metadata):
        """Create cryptographic hash of model and metadata"""
        model_bytes = json.dumps(model_state, sort_keys=True).encode()
        meta_bytes = json.dumps(metadata, sort_keys=True).encode()
        combined = model_bytes + meta_bytes
        return hashlib.sha256(combined).hexdigest()
    
    def train_model(self, dataset_name, model_type='linear', **kwargs):
        """
        Train a model with full reproducibility
        
        Args:
            dataset_name: Name of dataset (iris, mnist, etc.)
            model_type: Type of model to train
            **kwargs: Additional training parameters
        """
        
        # Load and preprocess data
        X, y = self.load_dataset(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create model
        model = self.create_model_architecture(
            input_size=X_train.shape[1],
            output_size=len(np.unique(y)),
            model_type=model_type
        )
        
        # Training configuration
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'dataset': dataset_name,
                'model_type': model_type,
                'random_seed': 42,
                'input_size': X_train.shape[1],
                'output_size': len(np.unique(y))
            })
            
            # Training loop
            epochs = kwargs.get('epochs', self.config['training']['default_epochs'])
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        test_outputs = model(X_test_tensor)
                        test_loss = criterion(test_outputs, y_test_tensor)
                        accuracy = (test_outputs.argmax(1) == y_test_tensor).float().mean()
                    
                    print(f'Epoch {epoch}: Loss={loss.item():.4f}, Test Accuracy={accuracy:.4f}')
                    mlflow.log_metrics({
                        'train_loss': loss.item(),
                        'test_loss': test_loss.item(),
                        'accuracy': accuracy.item()
                    }, step=epoch)
        
        # Prepare model metadata
        metadata = {
            'dataset': dataset_name,
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'input_shape': X_train.shape[1],
            'output_classes': len(np.unique(y)),
            'hyperparameters': {
                'epochs': epochs,
                'learning_rate': 0.001,
                'optimizer': 'Adam'
            },
            'performance': {
                'final_accuracy': accuracy.item(),
                'final_loss': test_loss.item()
            },
            'reproducibility_info': {
                'random_seed': 42,
                'pytorch_version': torch.__version__,
                'numpy_version': np.__version__
            }
        }
        
        # Save model and metadata
        model_filename = f"model_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        model_path = os.path.join(self.model_storage, model_filename)
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata,
            'scaler_state': scaler
        }, model_path)
        
        # Create and log model hash
        model_hash = self.create_model_hash(
            model.state_dict(), 
            metadata
        )
        
        # Version the model
        version_info = self.versioner.version_model(
            model_path=model_path,
            metadata=metadata,
            model_hash=model_hash
        )
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(model_path)
        mlflow.log_dict(metadata, "metadata.json")
        
        print(f"Model saved to {model_path}")
        print(f"Model hash: {model_hash}")
        print(f"Version: {version_info['version']}")
        
        return {
            'model_path': model_path,
            'model_hash': model_hash,
            'metadata': metadata,
            'version_info': version_info
        }
    
    def load_dataset(self, name):
        """Load example datasets"""
        if name == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris()
            return data.data, data.target
        elif name == 'mnist':
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1)
            return mnist.data[:1000], mnist.target[:1000].astype(int)
        else:
            raise ValueError(f"Dataset {name} not supported")
    
    def create_model_architecture(self, input_size, output_size, model_type='linear'):
        """Create model architecture based on type"""
        if model_type == 'linear':
            return nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )
        elif model_type == 'deep':
            return nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )
        else:
            raise ValueError(f"Model type {model_type} not supported")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a reproducible ML model')
    parser.add_argument('--dataset', type=str, default='iris', help='Dataset to use')
    parser.add_argument('--model-type', type=str, default='linear', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    args = parser.parse_args()
    
    trainer = ReproducibleModelTrainer()
    result = trainer.train_model(
        dataset_name=args.dataset,
        model_type=args.model_type,
        epochs=args.epochs
    )
    
    print(f"\nTraining complete!")
    print(f"Model saved at: {result['model_path']}")
    print(f"Model hash: {result['model_hash']}")