"""
Training script for denoising autoencoders with live visualization.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_utils import load_mnist_data, convert_to_bitmap
from shared.visualization import LiveTrainingVisualizer, plot_training_history
from noise_generation import NoiseGenerator, create_default_noise_generator, create_aggressive_noise_generator, create_mild_noise_generator
from bitmap_utils import convert_to_bitmap, compute_bitmap_metrics, visualize_bitmap_comparison
from denoising_models import create_denoising_model, get_denoising_loss, get_denoising_metrics


class DenoisingTrainer:
    """Trainer for denoising autoencoders with live visualization."""
    
    def __init__(self, 
                 model: keras.Model,
                 noise_generator: NoiseGenerator,
                 loss_fn: callable,
                 optimizer: keras.optimizers.Optimizer,
                 metrics: list,
                 log_dir: str = "logs/denoising",
                 live_visualization: bool = True):
        """
        Initialize the denoising trainer.
        
        Args:
            model: Denoising model to train
            noise_generator: Noise generator for creating training data
            loss_fn: Loss function
            optimizer: Optimizer for training
            metrics: List of metrics to track
            log_dir: Directory for logging
            live_visualization: Whether to show live visualization
        """
        self.model = model
        self.noise_generator = noise_generator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.log_dir = log_dir
        self.live_visualization = live_visualization
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_pixel_accuracy': [],
            'val_pixel_accuracy': [],
            'train_dice_coefficient': [],
            'val_dice_coefficient': [],
            'learning_rate': []
        }
        
        # Live visualizer
        if self.live_visualization:
            self.visualizer = LiveTrainingVisualizer(update_freq=50)
        else:
            self.visualizer = None
    
    def create_training_dataset(self, 
                              x_clean: np.ndarray,
                              batch_size: int = 32,
                              clean_ratio: float = 0.5) -> tf.data.Dataset:
        """
        Create training dataset with mixed clean/noisy samples.
        
        Args:
            x_clean: Clean binary images
            batch_size: Batch size
            clean_ratio: Ratio of clean samples in each batch
            
        Returns:
            TensorFlow dataset yielding (input, target) pairs
        """
        def data_generator():
            """Generator for training data."""
            n_samples = len(x_clean)
            indices = np.arange(n_samples)
            
            while True:
                # Shuffle indices
                np.random.shuffle(indices)
                
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    if len(batch_indices) < batch_size:
                        continue
                    
                    batch_clean = x_clean[batch_indices]
                    
                    # Determine which samples to keep clean vs add noise
                    n_clean = int(batch_size * clean_ratio)
                    clean_mask = np.zeros(batch_size, dtype=bool)
                    clean_mask[:n_clean] = True
                    np.random.shuffle(clean_mask)
                    
                    # Create input batch (mix of clean and noisy)
                    batch_input = batch_clean.copy()
                    if np.sum(~clean_mask) > 0:  # If there are samples to add noise to
                        batch_input[~clean_mask] = self.noise_generator.add_noise(batch_clean[~clean_mask])
                    
                    # Target is always clean
                    batch_target = batch_clean
                    
                    # Add channel dimension if needed
                    if len(batch_input.shape) == 3:
                        batch_input = np.expand_dims(batch_input, -1)
                        batch_target = np.expand_dims(batch_target, -1)
                    
                    yield batch_input, batch_target
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, 28, 28, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 28, 28, 1), dtype=tf.float32)
            )
        )
        
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    def train_step(self, batch_input: tf.Tensor, batch_target: tf.Tensor) -> Dict[str, float]:
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(batch_input, training=True)
            loss = self.loss_fn(batch_target, predictions)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute metrics
        results = {'loss': float(loss)}
        for metric in self.metrics:
            metric_value = metric(batch_target, predictions)
            results[metric.__name__] = float(metric_value)
        
        return results
    
    def val_step(self, batch_input: tf.Tensor, batch_target: tf.Tensor) -> Dict[str, float]:
        """Single validation step."""
        predictions = self.model(batch_input, training=False)
        loss = self.loss_fn(batch_target, predictions)
        
        # Compute metrics
        results = {'loss': float(loss)}
        for metric in self.metrics:
            metric_value = metric(batch_target, predictions)
            results[metric.__name__] = float(metric_value)
        
        return results
    
    def train(self, 
              x_train: np.ndarray,
              x_val: np.ndarray,
              epochs: int,
              batch_size: int = 32,
              clean_ratio: float = 0.5,
              steps_per_epoch: int = None,
              validation_steps: int = None,
              save_freq: int = 10,
              verbose: int = 1):
        """
        Train the denoising model.
        
        Args:
            x_train: Training images (clean)
            x_val: Validation images (clean)
            epochs: Number of epochs to train
            batch_size: Batch size
            clean_ratio: Ratio of clean samples in training batches
            steps_per_epoch: Steps per epoch (if None, use full dataset)
            validation_steps: Validation steps (if None, use full dataset)
            save_freq: Frequency to save model checkpoints
            verbose: Verbosity level
        """
        print(f"Starting denoising autoencoder training for {epochs} epochs...")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Clean ratio: {clean_ratio}")
        print("-" * 50)
        
        # Create datasets
        train_dataset = self.create_training_dataset(x_train, batch_size, clean_ratio)
        val_dataset = self.create_training_dataset(x_val, batch_size, clean_ratio)
        
        # Calculate steps
        if steps_per_epoch is None:
            steps_per_epoch = len(x_train) // batch_size
        if validation_steps is None:
            validation_steps = len(x_val) // batch_size
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            if verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = {'loss': [], 'pixel_accuracy': [], 'dice_coefficient': []}
            step_count = 0
            
            for batch_input, batch_target in train_dataset:
                if step_count >= steps_per_epoch:
                    break
                
                # Training step
                step_results = self.train_step(batch_input, batch_target)
                
                # Accumulate metrics
                for key, value in step_results.items():
                    if key in train_metrics:
                        train_metrics[key].append(value)
                
                # Update live visualization
                if self.visualizer and step_count % self.visualizer.update_freq == 0:
                    # Get current predictions for visualization
                    predictions = self.model(batch_input[:3], training=False)
                    
                    # Convert to numpy and remove channel dimension for visualization
                    clean_vis = batch_target[:3].numpy().squeeze()
                    noisy_vis = batch_input[:3].numpy().squeeze()
                    denoised_vis = predictions[:3].numpy().squeeze()
                    
                    self.visualizer.update_loss(step_results['loss'])
                    self.visualizer.update_samples(clean_vis, noisy_vis, denoised_vis)
                    self.visualizer.refresh_display()
                
                step_count += 1
                
                if verbose >= 2 and step_count % 100 == 0:
                    print(f"  Step {step_count}/{steps_per_epoch}, Loss: {step_results['loss']:.4f}")
            
            # Validation
            val_metrics = {'loss': [], 'pixel_accuracy': [], 'dice_coefficient': []}
            val_step_count = 0
            
            for batch_input, batch_target in val_dataset:
                if val_step_count >= validation_steps:
                    break
                
                # Validation step
                step_results = self.val_step(batch_input, batch_target)
                
                # Accumulate metrics
                for key, value in step_results.items():
                    if key in val_metrics:
                        val_metrics[key].append(value)
                
                val_step_count += 1
            
            # Calculate epoch metrics
            epoch_train_loss = np.mean(train_metrics['loss'])
            epoch_val_loss = np.mean(val_metrics['loss'])
            epoch_train_acc = np.mean(train_metrics['pixel_accuracy'])
            epoch_val_acc = np.mean(val_metrics['pixel_accuracy'])
            epoch_train_dice = np.mean(train_metrics['dice_coefficient'])
            epoch_val_dice = np.mean(val_metrics['dice_coefficient'])
            
            # Record history
            self.history['train_loss'].append(epoch_train_loss)
            self.history['val_loss'].append(epoch_val_loss)
            self.history['train_pixel_accuracy'].append(epoch_train_acc)
            self.history['val_pixel_accuracy'].append(epoch_val_acc)
            self.history['train_dice_coefficient'].append(epoch_train_dice)
            self.history['val_dice_coefficient'].append(epoch_val_dice)
            self.history['learning_rate'].append(float(self.optimizer.learning_rate))
            
            # Print epoch results
            if verbose >= 1:
                print(f"  Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
                print(f"  Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")
                print(f"  Train Dice: {epoch_train_dice:.4f}, Val Dice: {epoch_val_dice:.4f}")
                print()
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                self.save_model(os.path.join(self.log_dir, "best_model.weights.h5"))
                if verbose >= 1:
                    print(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch + 1}.weights.h5")
                self.save_model(checkpoint_path)
                if verbose >= 1:
                    print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Close live visualization
        if self.visualizer:
            self.visualizer.close()
        
        print("Training completed!")
        return self.history
    
    def save_model(self, filepath: str):
        """Save the model."""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load the model."""
        self.model.load_weights(filepath)
    
    def evaluate_model(self, 
                      x_test: np.ndarray,
                      noise_types: list = None,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            x_test: Test images (clean)
            noise_types: List of noise types to test
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        if noise_types is None:
            noise_types = ['salt_pepper', 'pixel_flip', 'gaussian', 'blocks']
        
        results = {}
        
        for noise_type in noise_types:
            print(f"Evaluating on {noise_type} noise...")
            
            # Add noise to test images
            noisy_test = self.noise_generator.add_noise(x_test, noise_type)
            
            # Add channel dimension
            if len(x_test.shape) == 3:
                clean_input = np.expand_dims(x_test, -1)
                noisy_input = np.expand_dims(noisy_test, -1)
            else:
                clean_input = x_test
                noisy_input = noisy_test
            
            # Get predictions
            denoised = self.model.predict(noisy_input, verbose=0)
            
            # Remove channel dimension for evaluation
            clean_eval = clean_input.squeeze()
            denoised_eval = denoised.squeeze()
            
            # Compute metrics
            metrics = compute_bitmap_metrics(clean_eval, denoised_eval)
            results[noise_type] = metrics
            
            print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
            print(f"  Dice Coefficient: {metrics['dice_coefficient']['mean']:.4f}")
            print(f"  IoU Score: {metrics['jaccard_index']['mean']:.4f}")
            
            # Save visualization
            if save_results:
                viz_path = os.path.join(self.log_dir, f"evaluation_{noise_type}.png")
                visualize_bitmap_comparison(
                    clean_eval[:5], 
                    noisy_test[:5], 
                    denoised_eval[:5],
                    save_path=viz_path
                )
        
        # Save results
        if save_results:
            results_path = os.path.join(self.log_dir, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Evaluation results saved to {results_path}")
        
        return results


def train_denoising_autoencoder(model_type: str = "unet",
                               noise_type: str = "default",
                               epochs: int = 50,
                               batch_size: int = 32,
                               learning_rate: float = 1e-3,
                               clean_ratio: float = 0.5,
                               loss_type: str = "binary_crossentropy",
                               bitmap_threshold: float = 0.5,
                               log_dir: Optional[str] = None,
                               live_visualization: bool = True,
                               evaluate_after_training: bool = True) -> Tuple[keras.Model, Dict]:
    """
    Train a denoising autoencoder on MNIST.
    
    Args:
        model_type: Type of denoising model
        noise_type: Type of noise generator ("mild", "default", "aggressive")
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        clean_ratio: Ratio of clean samples in training batches
        loss_type: Type of loss function
        bitmap_threshold: Threshold for converting to binary
        log_dir: Directory for logging
        live_visualization: Whether to show live visualization
        evaluate_after_training: Whether to evaluate after training
        
    Returns:
        Trained model and training history
    """
    # Set up logging directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/denoising_{model_type}_{noise_type}_{timestamp}"
    
    print("Denoising Autoencoder Training Configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Noise type: {noise_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Clean ratio: {clean_ratio}")
    print(f"  Loss type: {loss_type}")
    print(f"  Bitmap threshold: {bitmap_threshold}")
    print(f"  Log dir: {log_dir}")
    print(f"  Live visualization: {live_visualization}")
    print()
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data(normalize=True)
    
    # Convert to binary images
    x_train_binary = convert_to_bitmap(x_train, threshold=bitmap_threshold)
    x_test_binary = convert_to_bitmap(x_test, threshold=bitmap_threshold)
    
    print(f"Training samples: {len(x_train_binary)}")
    print(f"Test samples: {len(x_test_binary)}")
    print()
    
    # Create noise generator
    if noise_type == "mild":
        noise_generator = create_mild_noise_generator()
    elif noise_type == "aggressive":
        noise_generator = create_aggressive_noise_generator()
    else:
        noise_generator = create_default_noise_generator()
    
    # Create model
    print("Creating model...")
    model = create_denoising_model(model_type=model_type)
    
    # Create loss function and metrics
    loss_fn = get_denoising_loss(loss_type)
    metrics = get_denoising_metrics()
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create trainer
    trainer = DenoisingTrainer(
        model=model,
        noise_generator=noise_generator,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        log_dir=log_dir,
        live_visualization=live_visualization
    )
    
    # Train model
    history = trainer.train(
        x_train=x_train_binary,
        x_val=x_test_binary,
        epochs=epochs,
        batch_size=batch_size,
        clean_ratio=clean_ratio,
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
    
    # Evaluate model
    if evaluate_after_training:
        evaluation_results = trainer.evaluate_model(x_test_binary)
        print("Evaluation completed!")
    
    print("Training completed successfully!")
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train denoising autoencoder on MNIST')
    parser.add_argument('--model', type=str, default='unet', 
                       choices=['basic', 'unet', 'residual', 'attention', 'vae'],
                       help='Model architecture')
    parser.add_argument('--noise', type=str, default='default',
                       choices=['mild', 'default', 'aggressive'],
                       help='Noise generator type')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--clean_ratio', type=float, default=0.5,
                       help='Ratio of clean samples in training batches')
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                       choices=['binary_crossentropy', 'dice', 'combined', 'focal'],
                       help='Loss function type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Bitmap threshold')
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable live visualization')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip evaluation after training')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_denoising_autoencoder(
        model_type=args.model,
        noise_type=args.noise,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clean_ratio=args.clean_ratio,
        loss_type=args.loss,
        bitmap_threshold=args.threshold,
        live_visualization=not args.no_viz,
        evaluate_after_training=not args.no_eval
    )
