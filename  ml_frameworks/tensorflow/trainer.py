import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import tensorflow_datasets as tfds
import numpy as np
import json
import hashlib
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.tensorflow
import warnings
warnings.filterwarnings('ignore')

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class TensorFlowReproducibleTrainer:
    """Advanced TensorFlow trainer with full reproducibility"""
    
    def __init__(self, config):
        self.config = config
        
        # Set all random seeds
        self._set_seeds(config['training']['random_seed'])
        
        # Enable deterministic operations
        tf.config.experimental.enable_op_determinism()
        
        # Setup strategy (MirroredStrategy for multi-GPU)
        self.strategy = self._setup_strategy()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        
    def _set_seeds(self, seed):
        """Set all random seeds"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    
    def _setup_strategy(self):
        """Setup distribution strategy"""
        if len(tf.config.list_physical_devices('GPU')) > 1:
            return tf.distribute.MirroredStrategy()
        else:
            return tf.distribute.OneDeviceStrategy("/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0")
    
    def create_model(self, model_name, input_shape, num_classes):
        """Create TensorFlow model"""
        with self.strategy.scope():
            if model_name == 'efficientnet':
                base_model = keras.applications.EfficientNetB0(
                    include_top=False,
                    weights='imagenet' if self.config['training']['use_pretrained'] else None,
                    input_shape=input_shape
                )
                base_model.trainable = True
                
                model = keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.Dropout(0.2),
                    layers.Dense(num_classes, activation='softmax')
                ])
            
            elif model_name == 'resnet50':
                base_model = keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet' if self.config['training']['use_pretrained'] else None,
                    input_shape=input_shape
                )
                
                model = keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(1024, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(num_classes, activation='softmax')
                ])
            
            elif model_name == 'transformer':
                # Vision Transformer implementation
                class Patches(layers.Layer):
                    def __init__(self, patch_size):
                        super().__init__()
                        self.patch_size = patch_size
                    
                    def call(self, images):
                        batch_size = tf.shape(images)[0]
                        patches = tf.image.extract_patches(
                            images=images,
                            sizes=[1, self.patch_size, self.patch_size, 1],
                            strides=[1, self.patch_size, self.patch_size, 1],
                            rates=[1, 1, 1, 1],
                            padding="VALID"
                        )
                        patch_dims = patches.shape[-1]
                        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
                        return patches
                
                # Build ViT model
                inputs = layers.Input(shape=input_shape)
                patches = Patches(patch_size=16)(inputs)
                
                # Position embedding
                num_patches = patches.shape[1]
                projection_dim = 64
                
                positions = tf.range(start=0, limit=num_patches, delta=1)
                position_embedding = layers.Embedding(
                    input_dim=num_patches, output_dim=projection_dim
                )(positions)
                
                encoded = layers.Dense(projection_dim)(patches) + position_embedding
                
                # Transformer blocks
                for _ in range(8):
                    # Layer normalization
                    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
                    
                    # Multi-head attention
                    attention_output = layers.MultiHeadAttention(
                        num_heads=4, key_dim=projection_dim
                    )(x1, x1)
                    
                    # Skip connection
                    x2 = layers.Add()([attention_output, encoded])
                    
                    # Feed forward network
                    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                    x3 = layers.Dense(projection_dim * 2, activation="relu")(x3)
                    x3 = layers.Dense(projection_dim)(x3)
                    
                    encoded = layers.Add()([x3, x2])
                
                # Classification head
                representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
                representation = layers.Flatten()(representation)
                representation = layers.Dropout(0.5)(representation)
                outputs = layers.Dense(num_classes, activation="softmax")(representation)
                
                model = keras.Model(inputs=inputs, outputs=outputs)
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Compile model with mixed precision
            optimizer = keras.optimizers.Adam(
                learning_rate=self.config['training']['learning_rate'],
                beta_1=0.9, beta_2=0.999
            )
            
            loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            metrics = [
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
            ]
            
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            return model
    
    def train(self, dataset_name, model_name):
        """Train model with advanced features"""
        # Prepare dataset
        train_dataset, val_dataset = self._prepare_dataset(dataset_name)
        
        # Create model
        input_shape = (224, 224, 3)
        num_classes = self._get_num_classes(dataset_name)
        
        model = self.create_model(model_name, input_shape, num_classes)
        
        # Callbacks
        callbacks = self._create_callbacks(dataset_name, model_name)
        
        # Start MLflow run
        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            mlflow.log_params({
                'model': model_name,
                'dataset': dataset_name,
                'strategy': str(self.strategy.__class__.__name__),
                'mixed_precision': True,
                'batch_size': self.config['training']['batch_size']
            })
            
            # Train model
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config['training']['epochs'],
                callbacks=callbacks,
                verbose=1 if self.config['training']['verbose'] else 2
            )
            
            # Save final model
            model_hash = self._save_model(model, dataset_name, model_name, history)
            
            return {
                'model_hash': model_hash,
                'final_accuracy': history.history['val_accuracy'][-1],
                'history': history.history
            }
    
    def _prepare_dataset(self, dataset_name):
        """Prepare TensorFlow dataset"""
        if dataset_name == 'cifar10':
            (train_data, train_labels), (val_data, val_labels) = keras.datasets.cifar10.load_data()
            
            # Preprocessing
            def preprocess(image, label):
                image = tf.image.resize(image, [224, 224])
                image = tf.cast(image, tf.float32) / 255.0
                label = tf.one_hot(label, depth=10)
                return image, label
            
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            train_dataset = train_dataset.shuffle(10000).batch(self.config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
            val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            val_dataset = val_dataset.batch(self.config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
            
            return train_dataset, val_dataset
        
        elif dataset_name == 'imagenet':
            # Use tfds for ImageNet (requires manual download)
            dataset, info = tfds.load(
                'imagenet2012',
                split=['train', 'validation'],
                with_info=True,
                shuffle_files=True,
                as_supervised=True
            )
            
            train_dataset, val_dataset = dataset
            
            def preprocess(image, label):
                image = tf.image.resize(image, [224, 224])
                image = tf.cast(image, tf.float32) / 255.0
                image = keras.applications.imagenet_utils.preprocess_input(image)
                label = tf.one_hot(label, depth=1000)
                return image, label
            
            train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            train_dataset = train_dataset.shuffle(10000).batch(self.config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            val_dataset = val_dataset.batch(self.config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
            
            return train_dataset, val_dataset
    
    def _create_callbacks(self, dataset_name, model_name):
        """Create training callbacks"""
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # ReduceLROnPlateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=f"checkpoints/{model_name}_{dataset_name}/best_model.keras",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=f"logs/{model_name}_{dataset_name}",
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            
            # Custom callback for MLflow
            self._create_mlflow_callback()
        ]
        
        return callbacks
    
    def _create_mlflow_callback(self):
        """Create custom MLflow callback"""
        class MLflowCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    mlflow.log_metrics({
                        'train_loss': logs.get('loss', 0),
                        'train_accuracy': logs.get('accuracy', 0),
                        'val_loss': logs.get('val_loss', 0),
                        'val_accuracy': logs.get('val_accuracy', 0),
                        'learning_rate': float(self.model.optimizer.learning_rate)
                    }, step=epoch)
        
        return MLflowCallback()
    
    def _save_model(self, model, dataset_name, model_name, history):
        """Save model with cryptographic hash"""
        # Get model weights
        weights = model.get_weights()
        
        # Create metadata
        metadata = {
            'framework': 'tensorflow',
            'model_architecture': model_name,
            'dataset': dataset_name,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'num_parameters': model.count_params(),
            'training_history': history,
            'keras_version': keras.__version__,
            'tensorflow_version': tf.__version__,
            'saved_at': datetime.now().isoformat()
        }
        
        # Create hash from weights and metadata
        weights_bytes = b''.join([w.tobytes() for w in weights])
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
        model_hash = hashlib.sha256(weights_bytes + metadata_bytes).hexdigest()
        
        # Save model in multiple formats
        save_dir = Path(self.config['model']['storage_path']) / 'tensorflow_models' / model_hash[:16]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Keras format
        model.save(save_dir / 'model.keras')
        
        # Save as SavedModel format
        tf.saved_model.save(model, str(save_dir / 'saved_model'))
        
        # Save as TensorFlow Lite (for mobile)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(save_dir / 'model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Save metadata
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log to MLflow
        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(str(save_dir))
        mlflow.log_param("model_hash", model_hash)
        
        return model_hash

# Quantization and Optimization
class ModelOptimizer:
    """Optimize models for deployment"""
    
    @staticmethod
    def quantize_model(model, quantization_type='int8'):
        """Quantize model for inference optimization"""
        if quantization_type == 'int8':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            quantized_model = converter.convert()
            return quantized_model
        
        elif quantization_type == 'float16':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            quantized_model = converter.convert()
            return quantized_model
        
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    @staticmethod
    def prune_model(model, pruning_rate=0.5):
        """Prune model weights"""
        import tensorflow_model_optimization as tfmot
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=pruning_rate,
                begin_step=0,
                frequency=100
            )
        }
        
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        
        # Recompile the model
        model_for_pruning.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        return model_for_pruning