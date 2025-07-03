"""
Unsupervised contrastive learning for MNIST representation learning.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_utils import load_mnist_data, preprocess_for_contrastive, save_representations
from shared.visualization import plot_training_history, plot_embeddings_2d
from models import create_contrastive_model
from contrastive_loss import create_contrastive_loss
from data_augmentation import create_mild_augmentation, create_simclr_augmentation


class UnsupervisedContrastiveTrainer:
    """Trainer for unsupervised contrastive learning."""
    
    def __init__(self, 
                 model: keras.Model,
                 loss_fn: keras.losses.Loss,
                 optimizer: keras.optimizers.Optimizer,
                 augmentation,
                 log_dir: str = "logs/unsupervised"):
        """
        Initialize the unsupervised contrastive trainer.
        
        Args:
            model: Contrastive model to train
            loss_fn: Contrastive loss function (should be unsupervised)
            optimizer: Optimizer for training
            augmentation: Data augmentation pipeline
            log_dir: Directory for logging
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.augmentation = augmentation
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.val_loss = keras.metrics.Mean(name='val_loss')
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    @tf.function
    def train_step(self, batch):
        """Single training step for unsupervised contrastive learning."""
        images = batch  # No labels needed for unsupervised learning
        
        # Generate two augmented views
        aug_images1 = tf.map_fn(
            lambda img: self.augmentation.augment_pair(img)[0],
            images,
            fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
        )
        
        aug_images2 = tf.map_fn(
            lambda img: self.augmentation.augment_pair(img)[1],
            images,
            fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
        )
        
        # Add channel dimension
        aug_images1 = tf.expand_dims(aug_images1, -1)
        aug_images2 = tf.expand_dims(aug_images2, -1)
        
        with tf.GradientTape() as tape:
            # Get projections for both views
            _, projections1 = self.model(aug_images1, training=True)
            _, projections2 = self.model(aug_images2, training=True)
            
            # Compute contrastive loss
            loss = self.loss_fn(projections1, projections2)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        
        return loss
    
    @tf.function
    def val_step(self, batch):
        """Single validation step."""
        images = batch
        
        # Generate two augmented views
        aug_images1 = tf.map_fn(
            lambda img: self.augmentation.augment_pair(img)[0],
            images,
            fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
        )
        
        aug_images2 = tf.map_fn(
            lambda img: self.augmentation.augment_pair(img)[1],
            images,
            fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
        )
        
        # Add channel dimension
        aug_images1 = tf.expand_dims(aug_images1, -1)
        aug_images2 = tf.expand_dims(aug_images2, -1)
        
        # Get projections for both views
        _, projections1 = self.model(aug_images1, training=False)
        _, projections2 = self.model(aug_images2, training=False)
        
        # Compute contrastive loss
        loss = self.loss_fn(projections1, projections2)
        
        # Update metrics
        self.val_loss.update_state(loss)
        
        return loss
    
    def train(self, 
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              epochs: int,
              steps_per_epoch: int = None,
              validation_steps: int = None,
              save_freq: int = 10,
              verbose: int = 1):
        """
        Train the unsupervised contrastive model.
        
        Args:
            train_dataset: Training dataset (images only, no labels)
            val_dataset: Validation dataset
            epochs: Number of epochs to train
            steps_per_epoch: Steps per epoch (if None, use full dataset)
            validation_steps: Validation steps (if None, use full dataset)
            save_freq: Frequency to save model checkpoints
            verbose: Verbosity level
        """
        print(f"Starting unsupervised contrastive training for {epochs} epochs...")
        print(f"Model: {self.model.encoder_type}")
        print(f"Loss: {self.loss_fn.name}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Reset metrics
            self.train_loss.reset_state()
            self.val_loss.reset_state()
            
            # Training
            if verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs}")
                
            step_count = 0
            for batch in train_dataset:
                if steps_per_epoch and step_count >= steps_per_epoch:
                    break
                    
                loss = self.train_step(batch)
                step_count += 1
                
                if verbose >= 2 and step_count % 100 == 0:
                    print(f"  Step {step_count}, Loss: {loss:.4f}")
            
            # Validation
            val_step_count = 0
            for batch in val_dataset:
                if validation_steps and val_step_count >= validation_steps:
                    break
                    
                self.val_step(batch)
                val_step_count += 1
            
            # Record metrics
            train_loss_value = self.train_loss.result()
            val_loss_value = self.val_loss.result()
            lr_value = self.optimizer.learning_rate
            
            self.history['train_loss'].append(float(train_loss_value))
            self.history['val_loss'].append(float(val_loss_value))
            self.history['learning_rate'].append(float(lr_value))
            
            # Print epoch results
            if verbose >= 1:
                print(f"  Train Loss: {float(train_loss_value):.4f}")
                print(f"  Val Loss: {float(val_loss_value):.4f}")
                print(f"  Learning Rate: {float(lr_value):.6f}")
                print()
            
            # Save best model
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                self.save_model(os.path.join(self.log_dir, "best_model.weights.h5"))
                if verbose >= 1:
                    print(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch + 1}.weights.h5")
                self.save_model(checkpoint_path)
                if verbose >= 1:
                    print(f"  Checkpoint saved: {checkpoint_path}")
        
        print("Unsupervised training completed!")
        return self.history
    
    def save_model(self, filepath: str):
        """Save the model."""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load the model."""
        self.model.load_weights(filepath)
    
    def extract_representations(self, 
                              x_data: np.ndarray,
                              batch_size: int = 256) -> np.ndarray:
        """
        Extract learned representations from the encoder.
        
        Args:
            x_data: Input images
            batch_size: Batch size for processing
            
        Returns:
            Learned representations
        """
        # Ensure proper shape
        if len(x_data.shape) == 3:
            x_data = x_data.reshape(-1, 28, 28, 1)
        
        # Extract representations in batches
        representations = []
        for i in range(0, len(x_data), batch_size):
            batch = x_data[i:i + batch_size]
            batch_repr = self.model.get_embeddings(batch, training=False)
            representations.append(batch_repr.numpy())
        
        return np.concatenate(representations, axis=0)


def train_unsupervised_contrastive(encoder_type: str = "simple_cnn",
                                 loss_type: str = "ntxent",
                                 epochs: int = 100,
                                 batch_size: int = 256,
                                 learning_rate: float = 1e-3,
                                 temperature: float = 0.1,
                                 embedding_dim: int = 128,
                                 projection_dim: int = 64,
                                 augmentation_type: str = "mild",
                                 log_dir: Optional[str] = None,
                                 save_representations_path: Optional[str] = None) -> Tuple[keras.Model, Dict]:
    """
    Train an unsupervised contrastive model on MNIST.
    
    Args:
        encoder_type: Type of encoder architecture
        loss_type: Type of contrastive loss
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: Temperature for contrastive loss
        embedding_dim: Embedding dimension
        projection_dim: Projection dimension
        augmentation_type: Type of augmentation ("mild" or "strong")
        log_dir: Directory for logging
        save_representations_path: Path to save learned representations
        
    Returns:
        Trained model and training history
    """
    # Set up logging directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/unsupervised_{encoder_type}_{loss_type}_{timestamp}"
    
    print("Unsupervised Contrastive Learning Configuration:")
    print(f"  Encoder: {encoder_type}")
    print(f"  Loss: {loss_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Temperature: {temperature}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Projection dim: {projection_dim}")
    print(f"  Augmentation: {augmentation_type}")
    print(f"  Log dir: {log_dir}")
    print()
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data(normalize=True)
    
    # Preprocess for contrastive learning
    x_train, x_test = preprocess_for_contrastive(x_train, x_test, add_channel_dim=False)
    
    # Create datasets (no labels needed for unsupervised learning)
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_contrastive_model(
        encoder_type=encoder_type,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim
    )
    
    # Create loss function (only unsupervised losses)
    if loss_type.lower() not in ['ntxent', 'infonce']:
        raise ValueError(f"Loss type {loss_type} not supported for unsupervised learning. Use 'ntxent' or 'infonce'.")
    
    loss_fn = create_contrastive_loss(loss_type, temperature=temperature)
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create augmentation
    if augmentation_type == 'mild':
        augmentation = create_mild_augmentation()
    else:
        augmentation = create_simclr_augmentation()
    
    # Create trainer
    trainer = UnsupervisedContrastiveTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        augmentation=augmentation,
        log_dir=log_dir
    )
    
    # Train model
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(log_dir, "training_history.png")
    )
    
    # Save final model
    final_model_path = os.path.join(log_dir, "final_model.weights.h5")
    trainer.save_model(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Extract and save representations
    if save_representations_path:
        print("Extracting learned representations...")
        
        # Add channel dimension for representation extraction
        x_train_repr = x_train.reshape(-1, 28, 28, 1)
        x_test_repr = x_test.reshape(-1, 28, 28, 1)
        
        train_representations = trainer.extract_representations(x_train_repr)
        test_representations = trainer.extract_representations(x_test_repr)
        
        # Save representations
        save_representations(
            np.concatenate([train_representations, test_representations]),
            np.concatenate([y_train, y_test]),
            save_representations_path
        )
        
        # Visualize embeddings
        try:
            # Sample subset for visualization
            n_viz = 5000
            indices = np.random.choice(len(train_representations), n_viz, replace=False)
            
            plot_embeddings_2d(
                train_representations[indices],
                y_train[indices],
                method='tsne',
                save_path=os.path.join(log_dir, "embeddings_tsne.png")
            )
        except Exception as e:
            print(f"Could not create embedding visualization: {e}")
    
    print("Unsupervised contrastive learning completed successfully!")
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train unsupervised contrastive model on MNIST')
    parser.add_argument('--encoder', type=str, default='simple_cnn', 
                       choices=['simple_cnn', 'resnet', 'vit'],
                       help='Encoder architecture')
    parser.add_argument('--loss', type=str, default='ntxent',
                       choices=['ntxent', 'infonce'],
                       help='Contrastive loss function')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for contrastive loss')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=64,
                       help='Projection dimension')
    parser.add_argument('--augmentation', type=str, default='mild',
                       choices=['mild', 'strong'],
                       help='Augmentation strength')
    parser.add_argument('--save_representations', type=str, default=None,
                       help='Path to save learned representations')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_unsupervised_contrastive(
        encoder_type=args.encoder,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        augmentation_type=args.augmentation,
        save_representations_path=args.save_representations
    )
