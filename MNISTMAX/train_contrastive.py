"""
Training script for contrastive learning on MNIST.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Tuple

# Import our custom modules
from data_augmentation import ContrastiveAugmentation, create_simclr_augmentation, create_mild_augmentation
from contrastive_loss import NTXentLoss, SupConLoss, create_contrastive_loss
from models import create_contrastive_model, create_downstream_model


class ContrastiveTrainer:
    """Trainer class for contrastive learning."""
    
    def __init__(self, 
                 model: keras.Model,
                 loss_fn: keras.losses.Loss,
                 optimizer: keras.optimizers.Optimizer,
                 augmentation: ContrastiveAugmentation,
                 log_dir: str = "logs"):
        """
        Initialize the contrastive trainer.
        
        Args:
            model: Contrastive model to train
            loss_fn: Contrastive loss function
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
        """Single training step."""
        images, labels = batch
        
        # Generate augmented versions
        aug_images = tf.map_fn(
            lambda img: self.augmentation.augment_pair(img)[0],  # Use first augmentation
            images,
            fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
        )
        
        # Add channel dimension
        aug_images = tf.expand_dims(aug_images, -1)
        
        with tf.GradientTape() as tape:
            # Get projections for augmented images
            _, projections = self.model(aug_images, training=True)
            
            # Compute loss based on loss type
            if isinstance(self.loss_fn, SupConLoss):
                # Supervised contrastive loss
                loss = self.loss_fn(projections, labels)
            else:
                # Self-supervised loss - generate two augmented views
                aug_images2 = tf.map_fn(
                    lambda img: self.augmentation.augment_pair(img)[1],  # Use second augmentation
                    images,
                    fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
                )
                aug_images2 = tf.expand_dims(aug_images2, -1)
                _, projections2 = self.model(aug_images2, training=True)
                loss = self.loss_fn(projections, projections2)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        
        return loss
    
    @tf.function
    def val_step(self, batch):
        """Single validation step."""
        images, labels = batch
        
        # Generate augmented versions
        aug_images = tf.map_fn(
            lambda img: self.augmentation.augment_pair(img)[0],  # Use first augmentation
            images,
            fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
        )
        
        # Add channel dimension
        aug_images = tf.expand_dims(aug_images, -1)
        
        # Get projections for augmented images
        _, projections = self.model(aug_images, training=False)
        
        # Compute loss based on loss type
        if isinstance(self.loss_fn, SupConLoss):
            # Supervised contrastive loss
            loss = self.loss_fn(projections, labels)
        else:
            # Self-supervised loss - generate two augmented views
            aug_images2 = tf.map_fn(
                lambda img: self.augmentation.augment_pair(img)[1],  # Use second augmentation
                images,
                fn_output_signature=tf.TensorSpec(shape=(28, 28), dtype=tf.float32)
            )
            aug_images2 = tf.expand_dims(aug_images2, -1)
            _, projections2 = self.model(aug_images2, training=False)
            loss = self.loss_fn(projections, projections2)
        
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
        Train the contrastive model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs to train
            steps_per_epoch: Steps per epoch (if None, use full dataset)
            validation_steps: Validation steps (if None, use full dataset)
            save_freq: Frequency to save model checkpoints
            verbose: Verbosity level
        """
        print(f"Starting contrastive training for {epochs} epochs...")
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
        
        print("Training completed!")
        return self.history
    
    def save_model(self, filepath: str):
        """Save the model."""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load the model."""
        self.model.load_weights(filepath)
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        ax1.plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        ax1.set_title('Contrastive Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(self.history['learning_rate'], label='Learning Rate', color='green', alpha=0.8)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_representations(model: keras.Model,
                           x_train: np.ndarray,
                           y_train: np.ndarray,
                           x_test: np.ndarray,
                           y_test: np.ndarray,
                           num_classes: int = 10) -> Dict[str, float]:
    """
    Evaluate learned representations using linear classification.
    
    Args:
        model: Trained contrastive model
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating learned representations...")
    
    # Extract features
    train_features = model.get_embeddings(x_train, training=False)
    test_features = model.get_embeddings(x_test, training=False)
    
    # Create and train linear classifier
    classifier = keras.Sequential([
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    classifier.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train classifier
    history = classifier.fit(
        train_features, y_train,
        validation_data=(test_features, y_test),
        epochs=50,
        batch_size=256,
        verbose=0
    )
    
    # Evaluate
    test_loss, test_accuracy = classifier.evaluate(test_features, y_test, verbose=0)
    
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'best_val_accuracy': max(history.history['val_accuracy'])
    }
    
    print(f"Linear evaluation results:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Best Val Accuracy: {results['best_val_accuracy']:.4f}")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train contrastive learning model on MNIST')
    parser.add_argument('--encoder', type=str, default='simple_cnn', 
                       choices=['simple_cnn', 'resnet', 'vit'],
                       help='Encoder architecture')
    parser.add_argument('--loss', type=str, default='ntxent',
                       choices=['ntxent', 'supcon', 'infonce', 'triplet'],
                       help='Contrastive loss function')
    parser.add_argument('--epochs', type=int, default=100,
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
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Log directory')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate representations after training')
    
    args = parser.parse_args()
    
    # Set up logging directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"logs/contrastive_{args.encoder}_{args.loss}_{timestamp}"
    
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create datasets with labels for supervised contrastive learning
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_contrastive_model(
        encoder_type=args.encoder,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim
    )
    
    # Create loss function
    loss_fn = create_contrastive_loss(args.loss, temperature=args.temperature)
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # Create augmentation
    if args.augmentation == 'mild':
        augmentation = create_mild_augmentation()
    else:
        augmentation = create_simclr_augmentation()
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        augmentation=augmentation,
        log_dir=args.log_dir
    )
    
    # Train model
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        verbose=1
    )
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(args.log_dir, "training_history.png")
    )
    
    # Save final model
    final_model_path = os.path.join(args.log_dir, "final_model.weights.h5")
    trainer.save_model(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Evaluate representations
    if args.evaluate:
        # Reshape data for evaluation
        x_train_eval = x_train.reshape(-1, 28, 28, 1)
        x_test_eval = x_test.reshape(-1, 28, 28, 1)
        
        results = evaluate_representations(
            model, x_train_eval, y_train, x_test_eval, y_test
        )
        
        # Save results
        import json
        results_path = os.path.join(args.log_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved: {results_path}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
