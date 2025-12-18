import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import hashlib
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

class SklearnReproducibleTrainer:
    """Scikit-learn trainer with full reproducibility and autoML features"""
    
    def __init__(self, config):
        self.config = config
        self.random_state = config['training']['random_seed']
        np.random.seed(self.random_state)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    def train(self, dataset_name, model_type='random_forest', auto_tune=True):
        """Train scikit-learn model with optional hyperparameter tuning"""
        
        # Load dataset
        X, y, feature_names = self._load_dataset(dataset_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Create preprocessing pipeline
        preprocessor = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Select model
        model = self._create_model(model_type)
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Hyperparameter tuning
        if auto_tune:
            best_params = self._hyperparameter_tuning(pipeline, X_train, y_train, model_type)
            pipeline.set_params(**best_params)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance (for tree-based models)
        feature_importance = self._get_feature_importance(pipeline, feature_names)
        
        # Create model hash
        model_hash = self._create_model_hash(pipeline, dataset_name, model_type, accuracy)
        
        # Save model
        self._save_model(pipeline, dataset_name, model_type, model_hash, accuracy, feature_importance)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(pipeline, "model")
            mlflow.log_params(pipeline.get_params())
            mlflow.log_metrics({
                'accuracy': accuracy,
                'cross_val_score': np.mean(cross_val_score(pipeline, X, y, cv=5))
            })
            mlflow.log_param("model_hash", model_hash)
        
        return {
            'model_hash': model_hash,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'model_type': model_type,
            'pipeline': pipeline
        }
    
    def _create_model(self, model_type):
        """Create scikit-learn model"""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        
        elif model_type == 'svm':
            return SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
        
        elif model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=self.random_state
            )
        
        elif model_type == 'stacking':
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(random_state=self.random_state)),
                ('svm', SVC(probability=True, random_state=self.random_state))
            ]
            
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _hyperparameter_tuning(self, pipeline, X_train, y_train, model_type):
        """Perform hyperparameter tuning using GridSearchCV"""
        
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            
            'gradient_boosting': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0]
            },
            
            'svm': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__kernel': ['linear', 'rbf', 'poly'],
                'classifier__gamma': ['scale', 'auto', 0.1, 1]
            },
            
            'mlp': {
                'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__learning_rate_init': [0.001, 0.01, 0.1],
                'classifier__alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        if model_type not in param_grids:
            return {}
        
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_type],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def _get_feature_importance(self, pipeline, feature_names):
        """Extract feature importance if available"""
        classifier = pipeline.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Create feature importance dictionary
            importance_dict = {
                feature: importance 
                for feature, importance in zip(feature_names, importances)
            }
            
            # Sort by importance
            sorted_importance = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return sorted_importance
        
        elif hasattr(classifier, 'coef_'):
            # For linear models
            if len(classifier.coef_.shape) == 1:
                importances = np.abs(classifier.coef_)
            else:
                importances = np.mean(np.abs(classifier.coef_), axis=0)
            
            importance_dict = {
                feature: importance 
                for feature, importance in zip(feature_names, importances)
            }
            
            sorted_importance = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return sorted_importance
        
        else:
            return {}
    
    def _create_model_hash(self, pipeline, dataset_name, model_type, accuracy):
        """Create cryptographic hash of the model"""
        # Serialize model parameters
        model_params = pipeline.get_params()
        params_bytes = json.dumps(model_params, sort_keys=True).encode()
        
        # Create metadata
        metadata = {
            'framework': 'sklearn',
            'model_type': model_type,
            'dataset': dataset_name,
            'accuracy': accuracy,
            'sklearn_version': joblib.__version__,
            'random_state': self.random_state,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
        
        # Create hash
        model_hash = hashlib.sha256(params_bytes + metadata_bytes).hexdigest()
        return model_hash
    
    def _save_model(self, pipeline, dataset_name, model_type, model_hash, accuracy, feature_importance):
        """Save model to disk"""
        save_dir = Path(self.config['model']['storage_path']) / 'sklearn_models' / model_hash[:16]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / 'model.joblib'
        joblib.dump(pipeline, model_path)
        
        # Save metadata
        metadata = {
            'model_hash': model_hash,
            'model_type': model_type,
            'dataset': dataset_name,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'parameters': pipeline.get_params(),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance plot if available
        if feature_importance:
            self._plot_feature_importance(feature_importance, save_dir)
        
        return model_path
    
    def _plot_feature_importance(self, feature_importance, save_dir):
        """Create feature importance visualization"""
        import matplotlib.pyplot as plt
        
        features = list(feature_importance.keys())[:20]  # Top 20 features
        importances = list(feature_importance.values())[:20]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        
        plot_path = save_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
    
    def _load_dataset(self, dataset_name):
        """Load different datasets"""
        if dataset_name == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris()
            return data.data, data.target, data.feature_names
        
        elif dataset_name == 'digits':
            from sklearn.datasets import load_digits
            data = load_digits()
            return data.data, data.target, [f'pixel_{i}' for i in range(data.data.shape[1])]
        
        elif dataset_name == 'wine':
            from sklearn.datasets import load_wine
            data = load_wine()
            return data.data, data.target, data.feature_names
        
        elif dataset_name == 'breast_cancer':
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            return data.data, data.target, data.feature_names
        
        elif dataset_name == 'california_housing':
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            return data.data, data.target, data.feature_names
        
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

# Advanced ensemble methods
class AdvancedEnsemble:
    """Advanced ensemble methods for scikit-learn"""
    
    @staticmethod
    def create_voting_ensemble():
        """Create voting classifier ensemble"""
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        
        estimators = [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('nb', GaussianNB())
        ]
        
        return VotingClassifier(estimators=estimators, voting='soft')
    
    @staticmethod
    def create_blending_ensemble(X_train, y_train, X_val, y_val):
        """Create blending ensemble with meta-learner"""
        from sklearn.linear_model import LogisticRegression
        
        # Base models
        base_models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(random_state=42),
            SVC(probability=True, random_state=42)
        ]
        
        # Train base models
        base_predictions = []
        for model in base_models:
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            base_predictions.append(preds)
        
        # Stack predictions
        stacked_predictions = np.hstack(base_predictions)
        
        # Train meta-learner
        meta_learner = LogisticRegression()
        meta_learner.fit(stacked_predictions, y_val)
        
        return base_models, meta_learner
    
    @staticmethod
    def create_bagging_ensemble(base_estimator, n_estimators=10):
        """Create bagging ensemble"""
        from sklearn.ensemble import BaggingClassifier
        
        return BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )