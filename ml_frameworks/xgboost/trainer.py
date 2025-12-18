import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import joblib
import json
import hashlib
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')

class XGBoostReproducibleTrainer:
    """XGBoost trainer with GPU support and advanced features"""
    
    def __init__(self, config):
        self.config = config
        self.random_state = config['training']['random_seed']
        np.random.seed(self.random_state)
        
        # Set XGBoost parameters
        self.use_gpu = config['ml_frameworks']['xgboost']['gpu_support'] and self._check_gpu()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    def _check_gpu(self):
        """Check if GPU is available for XGBoost"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def train(self, dataset_name, task='classification', use_gpu=None):
        """Train XGBoost model with optional GPU acceleration"""
        
        # Override GPU setting if specified
        if use_gpu is not None:
            self.use_gpu = use_gpu
        
        # Load dataset
        X, y, feature_names = self._load_dataset(dataset_name, task)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Create DMatrix for XGBoost (optimized data structure)
        if task == 'classification':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set parameters based on task and GPU availability
        params = self._get_parameters(task)
        
        # Train model
        print(f"Training XGBoost model on {'GPU' if self.use_gpu else 'CPU'}...")
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config['training']['xgboost']['n_estimators'],
            evals=[(dtrain, 'train'), (dtest, 'test')],
            evals_result=evals_result,
            early_stopping_rounds=self.config['training']['early_stopping_patience'],
            verbose_eval=self.config['training']['verbose']
        )
        
        # Evaluate
        y_pred = model.predict(dtest)
        
        if task == 'classification':
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            auc = roc_auc_score(y_test, y_pred)
            metrics = {'accuracy': accuracy, 'auc': auc}
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            metrics = {'mse': mse, 'rmse': rmse}
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        feature_importance = self._process_feature_importance(importance, feature_names)
        
        # Create model hash
        model_hash = self._create_model_hash(model, dataset_name, task, metrics)
        
        # Save model
        self._save_model(model, dataset_name, task, model_hash, metrics, feature_importance, evals_result)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.xgboost.log_model(model, "model")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_hash", model_hash)
            mlflow.log_param("gpu_used", self.use_gpu)
        
        return {
            'model_hash': model_hash,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model': model,
            'evals_result': evals_result
        }
    
    def _get_parameters(self, task):
        """Get XGBoost parameters based on task and hardware"""
        base_params = {
            'seed': self.random_state,
            'verbosity': 0,
            'n_jobs': -1
        }
        
        if self.use_gpu:
            base_params['tree_method'] = 'gpu_hist'
            base_params['predictor'] = 'gpu_predictor'
            base_params['gpu_id'] = 0
        else:
            base_params['tree_method'] = 'hist'
        
        if task == 'classification':
            task_params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'error'],
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
        else:  # regression
            task_params = {
                'objective': 'reg:squarederror',
                'eval_metric': ['rmse', 'mae'],
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
        
        return {**base_params, **task_params}
    
    def _process_feature_importance(self, importance, feature_names):
        """Process XGBoost feature importance"""
        # Convert to dictionary with proper feature names
        importance_dict = {}
        
        for idx, (fid, score) in enumerate(importance.items()):
            # fid is like 'f0', 'f1', etc.
            feature_idx = int(fid[1:])
            if feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
            else:
                feature_name = f'feature_{feature_idx}'
            
            importance_dict[feature_name] = score
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance
    
    def _create_model_hash(self, model, dataset_name, task, metrics):
        """Create cryptographic hash of the model"""
        # Get model parameters and booster
        params = model.get_params()
        booster_bytes = model.save_raw('json').encode()
        
        # Create metadata
        metadata = {
            'framework': 'xgboost',
            'task': task,
            'dataset': dataset_name,
            'metrics': metrics,
            'xgboost_version': xgb.__version__,
            'random_state': self.random_state,
            'gpu_used': self.use_gpu,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
        
        # Create hash
        model_hash = hashlib.sha256(booster_bytes + metadata_bytes).hexdigest()
        return model_hash
    
    def _save_model(self, model, dataset_name, task, model_hash, metrics, feature_importance, evals_result):
        """Save XGBoost model"""
        save_dir = Path(self.config['model']['storage_path']) / 'xgboost_models' / model_hash[:16]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model in multiple formats
        model_path = save_dir / 'model.json'
        model.save_model(str(model_path))
        
        # Also save as joblib for scikit-learn compatibility
        joblib_path = save_dir / 'model.joblib'
        joblib.dump(model, joblib_path)
        
        # Save metadata
        metadata = {
            'model_hash': model_hash,
            'task': task,
            'dataset': dataset_name,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'parameters': model.get_params(),
            'evals_result': evals_result,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance plot
        self._plot_feature_importance(feature_importance, save_dir)
        
        # Save training history plot
        self._plot_training_history(evals_result, save_dir)
        
        return model_path
    
    def _plot_feature_importance(self, feature_importance, save_dir):
        """Create feature importance visualization"""
        import matplotlib.pyplot as plt
        
        features = list(feature_importance.keys())[:20]
        importances = list(feature_importance.values())[:20]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance (Gain)')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        
        plot_path = save_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
    
    def _plot_training_history(self, evals_result, save_dir):
        """Plot XGBoost training history"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation metrics
        for i, metric in enumerate(evals_result['train'].keys()):
            plt.subplot(1, len(evals_result['train']), i + 1)
            
            train_metric = evals_result['train'][metric]
            test_metric = evals_result['test'][metric]
            
            plt.plot(range(1, len(train_metric) + 1), train_metric, label='Train')
            plt.plot(range(1, len(test_metric) + 1), test_metric, label='Test')
            
            plt.xlabel('Boosting Rounds')
            plt.ylabel(metric)
            plt.title(f'{metric.upper()} History')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
    
    def _load_dataset(self, dataset_name, task):
        """Load dataset for XGBoost"""
        if dataset_name == 'breast_cancer':
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            return data.data, data.target, data.feature_names
        
        elif dataset_name == 'diabetes':
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            return data.data, data.target, data.feature_names
        
        elif dataset_name == 'boston':
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            return data.data, data.target, data.feature_names
        
        elif dataset_name == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris()
            return data.data, data.target, data.feature_names
        
        elif dataset_name == 'custom_csv':
            # Load from custom CSV file
            data_path = self.config['data']['custom_path']
            df = pd.read_csv(data_path)
            
            # Assume last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1].tolist()
            
            return X, y, feature_names
        
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

# Advanced XGBoost features
class AdvancedXGBoost:
    """Advanced XGBoost features and utilities"""
    
    @staticmethod
    def cross_validation_train(params, dtrain, num_boost_round=100, nfold=5):
        """Perform cross-validation training"""
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=nfold,
            stratified=True,
            early_stopping_rounds=10,
            verbose_eval=True,
            show_stdv=True
        )
        
        return cv_results
    
    @staticmethod
    def hyperparameter_tuning(X_train, y_train, param_grid, task='classification'):
        """Bayesian optimization for XGBoost hyperparameters"""
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer
        
        # Define search space
        if task == 'classification':
            search_spaces = {
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 10),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
                'reg_alpha': Real(1e-3, 10, prior='log-uniform'),
                'reg_lambda': Real(1e-3, 10, prior='log-uniform'),
                'min_child_weight': Integer(1, 10)
            }
        else:
            search_spaces = {
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 10),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
                'reg_alpha': Real(1e-3, 10, prior='log-uniform'),
                'reg_lambda': Real(1e-3, 10, prior='log-uniform'),
                'min_child_weight': Integer(1, 10)
            }
        
        # Create model
        if task == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=42
            )
        
        # Bayesian optimization
        opt = BayesSearchCV(
            model,
            search_spaces,
            n_iter=50,
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        opt.fit(X_train, y_train)
        
        return opt.best_params_, opt.best_score_
    
    @staticmethod
    def create_explainability_report(model, X, feature_names):
        """Create comprehensive explainability report using SHAP"""
        try:
            import shap
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Summary plot
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            
            # Dependence plots for top features
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
            
            # Force plot for single prediction
            shap.force_plot(explainer.expected_value, shap_values[0, :], X[0, :], feature_names=feature_names, show=False)
            
            return explainer, shap_values
            
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return None, None
    
    @staticmethod
    def deploy_onnx(model, input_sample, output_path):
        """Convert XGBoost model to ONNX format for deployment"""
        try:
            from onnxmltools import convert_xgboost
            from onnxconverter_common.data_types import FloatTensorType
            
            # Convert to ONNX
            initial_type = [('float_input', FloatTensorType([None, input_sample.shape[1]]))]
            onnx_model = convert_xgboost(model, initial_types=initial_type)
            
            # Save ONNX model
            import onnx
            onnx.save(onnx_model, output_path)
            
            return True
        except ImportError:
            print("ONNX tools not installed")
            return False