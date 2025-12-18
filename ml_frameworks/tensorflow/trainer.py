import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
from pathlib import Path
import json
import hashlib
from datetime import datetime
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class PyTorchReproducibleTrainer:
    """Advanced PyTorch trainer with full reproducibility and distributed training"""
    
    def __init__(self, config: Dict[str, Any], use_ddp: bool = False):
        self.config = config
        self.use_ddp = use_ddp
        self.device = self._setup_device()
        self.world_size = 1
        self.rank = 0
        
        if use_ddp:
            self._setup_distributed()
        
        # Set all random seeds
        self._set_seeds(config['training']['random_seed'])
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        
    def _setup_device(self):
        """Setup GPU/CPU device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = self.config['ml_frameworks']['pytorch']['cuda_benchmark']
            torch.backends.cudnn.deterministic = True
        else:
            device = torch.device("cpu")
        return device
    
    def _setup_distributed(self):
        """Initialize distributed training"""
        dist.init_process_group(backend='nccl')
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        torch.cuda.set_device(self.rank)
    
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_model(self, model_name: str, num_classes: int):
        """Create model architecture"""
        if model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=self.config['training']['use_pretrained'])
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif model_name == 'vit':
            from transformers import ViTForImageClassification
            model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        
        elif model_name == 'efficientnet':
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        
        elif model_name == 'transformer':
            model = nn.Transformer(
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Move to device and wrap with DDP if needed
        model = model.to(self.device)
        if self.use_ddp:
            model = DDP(model, device_ids=[self.rank])
        
        return model
    
    def train(self, dataset_name: str, model_name: str, **kwargs):
        """Advanced training with mixed precision, gradient accumulation, etc."""
        from torch.cuda.amp import autocast, GradScaler
        
        # Setup dataset
        train_loader, val_loader = self._prepare_data(dataset_name)
        
        # Create model
        model = self.create_model(model_name, self._get_num_classes(dataset_name))
        
        # Setup optimizer, scheduler, and scaler
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        scaler = GradScaler() if self.config['ml_frameworks']['pytorch']['mixed_precision'] else None
        
        # Training metrics
        metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Start MLflow run
        with mlflow.start_run():
            mlflow.log_params({
                'model': model_name,
                'dataset': dataset_name,
                'batch_size': self.config['training']['batch_size'],
                'optimizer': optimizer.__class__.__name__,
                'learning_rate': self.config['training']['learning_rate'],
                'mixed_precision': scaler is not None,
                'distributed': self.use_ddp
            })
            
            # Training loop
            for epoch in range(self.config['training']['epochs']):
                start_time = time.time()
                
                # Train step
                train_loss = self._train_epoch(
                    model, train_loader, optimizer, 
                    scheduler, scaler, epoch
                )
                
                # Validation step
                val_acc = self._validate(model, val_loader)
                
                # Calculate epoch time
                epoch_time = time.time() - start_time
                
                # Log metrics
                if self.rank == 0:  # Only log from master process
                    metrics['train_loss'].append(train_loss)
                    metrics['val_accuracy'].append(val_acc)
                    metrics['epoch_time'].append(epoch_time)
                    metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
                    
                    mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_accuracy': val_acc,
                        'epoch_time': epoch_time,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    }, step=epoch)
                    
                    print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}: '
                          f'Loss={train_loss:.4f}, Accuracy={val_acc:.4f}, '
                          f'Time={epoch_time:.2f}s')
                
                # Save checkpoint
                if (epoch + 1) % self.config['training']['checkpoint_frequency'] == 0:
                    self._save_checkpoint(model, optimizer, epoch, metrics)
            
            # Final model save
            model_hash = self._save_final_model(model, dataset_name, model_name, metrics)
            
            return {
                'model_hash': model_hash,
                'final_accuracy': metrics['val_accuracy'][-1],
                'training_time': sum(metrics['epoch_time']),
                'best_epoch': np.argmax(metrics['val_accuracy'])
            }
    
    def _train_epoch(self, model, train_loader, optimizer, scheduler, scaler, epoch):
        """Training loop for one epoch"""
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with autocast():
                    output = model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['training']['gradient_clip']
                )
        
        scheduler.step()
        return total_loss / len(train_loader)
    
    def _validate(self, model, val_loader):
        """Validation step"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        
        # Sync accuracy across all processes in DDP
        if self.use_ddp:
            accuracy_tensor = torch.tensor([accuracy], device=self.device)
            dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
            accuracy = accuracy_tensor.item() / self.world_size
        
        return accuracy
    
    def _save_final_model(self, model, dataset, model_name, metrics):
        """Save final model with cryptographic hash"""
        from collections import OrderedDict
        
        # Prepare model state (handle DDP)
        if self.use_ddp and self.rank == 0:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        # Create metadata
        metadata = {
            'framework': 'pytorch',
            'model_architecture': model_name,
            'dataset': dataset,
            'training_config': self.config['training'],
            'final_metrics': {
                'accuracy': metrics['val_accuracy'][-1],
                'loss': metrics['train_loss'][-1]
            },
            'hardware_info': {
                'device': str(self.device),
                'cuda_version': torch.version.cuda,
                'distributed': self.use_ddp,
                'world_size': self.world_size
            },
            'reproducibility_info': {
                'random_seed': self.config['training']['random_seed'],
                'deterministic_algorithms': True,
                'cudnn_deterministic': True
            }
        }
        
        # Create hash
        model_bytes = str(OrderedDict(sorted(model_state.items()))).encode()
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
        model_hash = hashlib.sha256(model_bytes + metadata_bytes).hexdigest()
        
        # Save model
        save_path = Path(self.config['model']['storage_path']) / 'pytorch_models'
        save_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{model_name}_{dataset}_{model_hash[:16]}.pt"
        filepath = save_path / filename
        
        torch.save({
            'model_state_dict': model_state,
            'metadata': metadata,
            'model_hash': model_hash,
            'config': self.config
        }, filepath)
        
        # Log to MLflow
        if self.rank == 0:
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_artifact(str(filepath))
            mlflow.log_dict(metadata, "metadata.json")
            mlflow.log_param("model_hash", model_hash)
        
        return model_hash

# Distributed training launcher
def launch_distributed_training():
    """Launch distributed training across multiple GPUs/nodes"""
    import torch.multiprocessing as mp
    
    def train_worker(rank, world_size, config):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        trainer = PyTorchReproducibleTrainer(config, use_ddp=True)
        result = trainer.train('imagenet', 'resnet50')
        
        if rank == 0:
            print(f"Training complete. Model hash: {result['model_hash']}")
    
    # Launch multiple processes
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size)