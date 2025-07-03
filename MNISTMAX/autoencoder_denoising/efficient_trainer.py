"""
Efficient training script with TensorFlow data pipeline for denoising autoencoders.
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
from PIL import Image
import imageio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_utils import load_mnist_data
from shared.visualization import plot_training_history
from bitmap_utils import convert_to_bitmap, compute_bitmap_metrics, visualize_bitmap_comparison
from denoising_models import create_denoising_model, get_denoising_loss, get_denoising_metrics


@tf.function
def add_salt_pepper_noise_tf(image, noise_prob=0.1, salt_prob=0.5):
    """TensorFlow function to add salt and pepper noise."""
    # Generate random values for noise mask and salt/pepper decision
    noise_mask = tf.random.uniform(tf.shape(image)) < noise_prob
    salt_mask = tf.random.uniform(tf.shape(image)) < salt_prob
    
    # Apply noise
    noisy = tf.where(noise_mask & salt_mask, 1.0, image)  # Salt
    noisy = tf.where(noise_mask & ~salt_mask, 0.0, noisy)  # Pepper
    
    return noisy


@tf.function
def add_pixel_flip_noise_tf(image, flip_prob=0.05):
    """TensorFlow function to add pixel flip noise."""
    flip_mask = tf.random.uniform(tf.shape(image)) < flip_prob
    return tf.where(flip_mask, 1.0 - image, image)


@tf.function
def add_gaussian_noise_tf(image, noise_std=0.1, threshold=0.5):
    """TensorFlow function to add Gaussian noise and re-binarize."""
    noise = tf.random.normal(tf.shape(image), stddev=noise_std)
    noisy = image + noise
    return tf.cast(noisy > threshold, tf.float32)


def create_efficient_dataset(x_data, batch_size=32, clean_ratio=0.5, noise_type='mixed'):
    """
    Create an efficient TensorFlow dataset with on-the-fly noise generation.
    
    Args:
        x_data: Clean binary images
        batch_size: Batch size
        clean_ratio: Ratio of clean samples in each batch
        noise_type: Type of noise to apply
        
    Returns:
        TensorFlow dataset yielding (noisy, clean) pairs
    """
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(x_data)
    
    def apply_noise_and_create_pair(clean_image):
        """Apply noise to create training pair."""
        # Decide whether to keep clean or add noise
        should_add_noise = tf.random.uniform([]) > clean_ratio
        
        if noise_type == 'salt_pepper':
            noisy_image = tf.cond(
                should_add_noise,
                lambda: add_salt_pepper_noise_tf(clean_image),
                lambda: clean_image
            )
        elif noise_type == 'pixel_flip':
            noisy_image = tf.cond(
                should_add_noise,
                lambda: add_pixel_flip_noise_tf(clean_image),
                lambda: clean_image
            )
        elif noise_type == 'gaussian':
            noisy_image = tf.cond(
                should_add_noise,
                lambda: add_gaussian_noise_tf(clean_image),
                lambda: clean_image
            )
        else:  # mixed noise
            # Randomly choose noise type
            noise_choice = tf.random.uniform([])
            noisy_image = tf.cond(
                should_add_noise,
                lambda: tf.cond(
                    noise_choice < 0.33,
                    lambda: add_salt_pepper_noise_tf(clean_image),
                    lambda: tf.cond(
                        noise_choice < 0.66,
                        lambda: add_pixel_flip_noise_tf(clean_image),
                        lambda: add_gaussian_noise_tf(clean_image)
                    )
                ),
                lambda: clean_image
            )
        
        # Add channel dimension
        noisy_image = tf.expand_dims(noisy_image, -1)
        clean_image = tf.expand_dims(clean_image, -1)
        
        return noisy_image, clean_image
    
    # Apply noise generation and create pairs
    dataset = dataset.map(apply_noise_and_create_pair, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


class EfficientDenoisingTrainer:
    """Efficient trainer for denoising autoencoders."""
    
    def __init__(self, 
                 model: keras.Model,
                 loss_fn: callable,
                 optimizer: keras.optimizers.Optimizer,
                 metrics: list,
                 log_dir: str = "logs/denoising",
                 save_animation: bool = False,
                 live_visualization: bool = False):
        """
        Initialize the efficient denoising trainer.
        
        Args:
            model: Denoising model to train
            loss_fn: Loss function
            optimizer: Optimizer for training
            metrics: List of metrics to track
            log_dir: Directory for logging
            save_animation: Whether to save epoch-by-epoch animation frames
            live_visualization: Whether to show live visualization during training
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.log_dir = log_dir
        self.save_animation = save_animation
        self.live_visualization = live_visualization
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Animation and live visualization setup
        if self.save_animation or self.live_visualization:
            if self.save_animation:
                self.animation_dir = os.path.join(log_dir, "animation_frames")
                os.makedirs(self.animation_dir, exist_ok=True)
                self.animation_frames = []
            
            # Prepare fixed test samples for animation/visualization
            self._setup_animation_samples()
            
            # Live visualization setup
            if self.live_visualization:
                self._setup_live_visualization()
        
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
    
    def _setup_animation_samples(self):
        """Setup fixed samples for animation."""
        # Load a few test samples for consistent animation
        (_, _), (x_test, _) = load_mnist_data(normalize=True)
        x_test_binary = convert_to_bitmap(x_test, threshold=0.5)
        
        # Select 3 diverse samples for animation
        self.animation_clean = x_test_binary[42:45]  # Fixed indices for consistency
        
        # Create noisy versions
        self.animation_noisy = add_salt_pepper_noise_tf(self.animation_clean, noise_prob=0.15).numpy()
    
    def _setup_live_visualization(self):
        """Setup live visualization window."""
        plt.ion()  # Turn on interactive mode
        self.live_fig, self.live_axes = plt.subplots(3, 3, figsize=(10, 10))
        self.live_fig.suptitle('Live Denoising Training Progress', fontsize=16, fontweight='bold')
        
        # Set up subplot titles
        for i in range(3):
            self.live_axes[i, 0].set_title('Clean' if i == 0 else '', fontsize=12)
            self.live_axes[i, 1].set_title('Noisy' if i == 0 else '', fontsize=12)
            self.live_axes[i, 2].set_title('Denoised' if i == 0 else '', fontsize=12)
            
            for j in range(3):
                self.live_axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def _update_live_visualization(self, epoch: int, train_loss: float, val_loss: float, 
                                 train_acc: float, val_acc: float):
        """Update live visualization window."""
        if not self.live_visualization:
            return
            
        # Get current model predictions
        noisy_input = np.expand_dims(self.animation_noisy, -1)
        denoised = self.model.predict(noisy_input, verbose=0)
        denoised = denoised.squeeze()
        
        # Clear and update plots
        for i in range(3):
            # Clear previous images
            self.live_axes[i, 0].clear()
            self.live_axes[i, 1].clear()
            self.live_axes[i, 2].clear()
            
            # Plot new images
            self.live_axes[i, 0].imshow(self.animation_clean[i], cmap='gray', vmin=0, vmax=1)
            self.live_axes[i, 1].imshow(self.animation_noisy[i], cmap='gray', vmin=0, vmax=1)
            self.live_axes[i, 2].imshow(denoised[i], cmap='gray', vmin=0, vmax=1)
            
            # Set titles and remove axes
            if i == 0:
                self.live_axes[i, 0].set_title('Clean', fontsize=12)
                self.live_axes[i, 1].set_title('Noisy', fontsize=12)
                self.live_axes[i, 2].set_title('Denoised', fontsize=12)
            
            for j in range(3):
                self.live_axes[i, j].axis('off')
        
        # Update main title with current metrics
        self.live_fig.suptitle(
            f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
            f'Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def _save_animation_frame(self, epoch: int):
        """Save animation frame for current epoch."""
        if not self.save_animation:
            return
            
        # Get current model predictions
        noisy_input = np.expand_dims(self.animation_noisy, -1)
        denoised = self.model.predict(noisy_input, verbose=0)
        denoised = denoised.squeeze()
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        fig.suptitle(f'Epoch {epoch + 1} - Denoising Progress', fontsize=16, fontweight='bold')
        
        for i in range(3):
            # Original clean
            axes[i, 0].imshow(self.animation_clean[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Clean' if i == 0 else '', fontsize=12)
            axes[i, 0].axis('off')
            
            # Noisy
            axes[i, 1].imshow(self.animation_noisy[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Noisy' if i == 0 else '', fontsize=12)
            axes[i, 1].axis('off')
            
            # Denoised
            axes[i, 2].imshow(denoised[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Denoised' if i == 0 else '', fontsize=12)
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(self.animation_dir, f"epoch_{epoch + 1:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.animation_frames.append(frame_path)
    
    def _create_animation_gif(self):
        """Create animated GIF from saved frames."""
        if not self.save_animation or not self.animation_frames:
            return
            
        print("Creating training animation...")
        
        # Load images
        images = []
        for frame_path in self.animation_frames:
            img = Image.open(frame_path)
            images.append(img)
        
        # Create GIF
        gif_path = os.path.join(self.log_dir, "training_animation.gif")
        
        # Save with different durations for first/last frames
        durations = []
        for i in range(len(images)):
            if i == 0 or i == len(images) - 1:
                durations.append(1500)  # 1.5 seconds for first and last frame
            else:
                durations.append(500)   # 0.5 seconds for middle frames
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0
        )
        
        print(f"Training animation saved: {gif_path}")
    
    @tf.function
    def train_step(self, batch_input, batch_target):
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(batch_input, training=True)
            loss = self.loss_fn(batch_target, predictions)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute metrics
        results = {'loss': loss}
        for metric in self.metrics:
            metric_value = metric(batch_target, predictions)
            results[metric.__name__] = metric_value
        
        return results
    
    @tf.function
    def val_step(self, batch_input, batch_target):
        """Single validation step."""
        predictions = self.model(batch_input, training=False)
        loss = self.loss_fn(batch_target, predictions)
        
        # Compute metrics
        results = {'loss': loss}
        for metric in self.metrics:
            metric_value = metric(batch_target, predictions)
            results[metric.__name__] = metric_value
        
        return results
    
    def train(self, 
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              epochs: int,
              steps_per_epoch: int = None,
              validation_steps: int = None,
              save_freq: int = 10,
              verbose: int = 1):
        """
        Train the denoising model efficiently.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs to train
            steps_per_epoch: Steps per epoch (if None, use full dataset)
            validation_steps: Validation steps (if None, use full dataset)
            save_freq: Frequency to save model checkpoints
            verbose: Verbosity level
        """
        print(f"Starting efficient denoising training for {epochs} epochs...")
        print(f"Model type: {type(self.model).__name__}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            if verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = {
                'loss': tf.keras.metrics.Mean(),
                'pixel_accuracy': tf.keras.metrics.Mean(),
                'dice_coefficient': tf.keras.metrics.Mean()
            }
            
            step_count = 0
            for batch_input, batch_target in train_dataset:
                if steps_per_epoch and step_count >= steps_per_epoch:
                    break
                
                # Training step
                step_results = self.train_step(batch_input, batch_target)
                
                # Update metrics
                for key, value in step_results.items():
                    if key in train_metrics:
                        train_metrics[key].update_state(value)
                
                step_count += 1
                
                if verbose >= 2 and step_count % 100 == 0:
                    print(f"  Step {step_count}, Loss: {float(step_results['loss']):.4f}")
            
            # Validation
            val_metrics = {
                'loss': tf.keras.metrics.Mean(),
                'pixel_accuracy': tf.keras.metrics.Mean(),
                'dice_coefficient': tf.keras.metrics.Mean()
            }
            
            val_step_count = 0
            for batch_input, batch_target in val_dataset:
                if validation_steps and val_step_count >= validation_steps:
                    break
                
                # Validation step
                step_results = self.val_step(batch_input, batch_target)
                
                # Update metrics
                for key, value in step_results.items():
                    if key in val_metrics:
                        val_metrics[key].update_state(value)
                
                val_step_count += 1
            
            # Record epoch metrics
            epoch_train_loss = float(train_metrics['loss'].result())
            epoch_val_loss = float(val_metrics['loss'].result())
            epoch_train_acc = float(train_metrics['pixel_accuracy'].result())
            epoch_val_acc = float(val_metrics['pixel_accuracy'].result())
            epoch_train_dice = float(train_metrics['dice_coefficient'].result())
            epoch_val_dice = float(val_metrics['dice_coefficient'].result())
            
            # Update history
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
            
            # Update live visualization
            self._update_live_visualization(epoch, epoch_train_loss, epoch_val_loss, 
                                          epoch_train_acc, epoch_val_acc)
            
            # Save animation frame
            self._save_animation_frame(epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch + 1}.weights.h5")
                self.save_model(checkpoint_path)
                if verbose >= 1:
                    print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Create animation GIF
        self._create_animation_gif()
        
        print("Efficient training completed!")
        return self.history
    
    def save_model(self, filepath: str):
        """Save the model."""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load the model."""
        self.model.load_weights(filepath)


def train_efficient_denoising_autoencoder(model_type: str = "basic",
                                        noise_type: str = "mixed",
                                        epochs: int = 10,
                                        batch_size: int = 64,
                                        learning_rate: float = 1e-3,
                                        clean_ratio: float = 0.3,
                                        loss_type: str = "binary_crossentropy",
                                        bitmap_threshold: float = 0.5,
                                        log_dir: Optional[str] = None,
                                        steps_per_epoch: Optional[int] = None,
                                        save_animation: bool = False,
                                        live_visualization: bool = False) -> Tuple[keras.Model, Dict]:
    """
    Train a denoising autoencoder efficiently on MNIST.
    
    Args:
        model_type: Type of denoising model
        noise_type: Type of noise ("salt_pepper", "pixel_flip", "gaussian", "mixed")
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        clean_ratio: Ratio of clean samples in training batches
        loss_type: Type of loss function
        bitmap_threshold: Threshold for converting to binary
        log_dir: Directory for logging
        
    Returns:
        Trained model and training history
    """
    # Set up logging directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/efficient_{model_type}_{noise_type}_{timestamp}"
    
    print("Efficient Denoising Autoencoder Training Configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Noise type: {noise_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Clean ratio: {clean_ratio}")
    print(f"  Loss type: {loss_type}")
    print(f"  Bitmap threshold: {bitmap_threshold}")
    print(f"  Log dir: {log_dir}")
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
    
    # Create efficient datasets
    print("Creating efficient data pipelines...")
    train_dataset = create_efficient_dataset(
        x_train_binary, 
        batch_size=batch_size, 
        clean_ratio=clean_ratio, 
        noise_type=noise_type
    )
    
    val_dataset = create_efficient_dataset(
        x_test_binary, 
        batch_size=batch_size, 
        clean_ratio=clean_ratio, 
        noise_type=noise_type
    )
    
    # Create model
    print("Creating model...")
    model = create_denoising_model(model_type=model_type)
    
    # Create loss function and metrics
    loss_fn = get_denoising_loss(loss_type)
    metrics = get_denoising_metrics()
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create trainer
    trainer = EfficientDenoisingTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        log_dir=log_dir,
        save_animation=save_animation,
        live_visualization=live_visualization
    )
    
    # Train model
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
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
    
    # Quick evaluation
    print("\nQuick evaluation...")
    
    # Get a few test samples for evaluation
    test_samples = x_test_binary[:100]
    
    # Add some noise manually for evaluation
    if noise_type == 'salt_pepper':
        noisy_samples = add_salt_pepper_noise_tf(test_samples).numpy()
    elif noise_type == 'pixel_flip':
        noisy_samples = add_pixel_flip_noise_tf(test_samples).numpy()
    else:
        noisy_samples = add_salt_pepper_noise_tf(test_samples).numpy()  # Default
    
    # Get predictions
    noisy_input = np.expand_dims(noisy_samples, -1)
    denoised = model.predict(noisy_input, verbose=0)
    denoised = denoised.squeeze()
    
    # Compute metrics
    metrics_result = compute_bitmap_metrics(test_samples, denoised)
    print(f"Pixel Accuracy: {metrics_result['pixel_accuracy']:.4f}")
    print(f"Dice Coefficient: {metrics_result['dice_coefficient']['mean']:.4f}")
    print(f"IoU Score: {metrics_result['jaccard_index']['mean']:.4f}")
    
    # Save visualization
    viz_path = os.path.join(log_dir, "evaluation_samples.png")
    visualize_bitmap_comparison(
        test_samples[:5], 
        noisy_samples[:5], 
        denoised[:5],
        save_path=viz_path
    )
    
    print("Efficient training completed successfully!")
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train denoising autoencoder efficiently on MNIST')
    parser.add_argument('--model', type=str, default='basic', 
                       choices=['basic', 'unet', 'residual', 'attention', 'vae'],
                       help='Model architecture')
    parser.add_argument('--noise', type=str, default='mixed',
                       choices=['salt_pepper', 'pixel_flip', 'gaussian', 'mixed'],
                       help='Noise type')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--clean_ratio', type=float, default=0.3,
                       help='Ratio of clean samples in training batches')
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                       choices=['binary_crossentropy', 'dice', 'combined', 'focal'],
                       help='Loss function type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Bitmap threshold')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                       help='Steps per epoch (None for full dataset)')
    parser.add_argument('--save_animation', action='store_true',
                       help='Save training animation showing denoising progress')
    parser.add_argument('--live_viz', action='store_true',
                       help='Show live visualization during training')
    
    args = parser.parse_args()
    
    # Calculate steps per epoch for ~1000 samples if not specified
    if args.steps_per_epoch is None and args.epochs > 10:
        args.steps_per_epoch = max(16, 1000 // args.batch_size)  # ~1000 samples per epoch
        print(f"Using {args.steps_per_epoch} steps per epoch (~{args.steps_per_epoch * args.batch_size} samples)")
    
    # Train model
    model, history = train_efficient_denoising_autoencoder(
        model_type=args.model,
        noise_type=args.noise,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clean_ratio=args.clean_ratio,
        loss_type=args.loss,
        bitmap_threshold=args.threshold,
        steps_per_epoch=args.steps_per_epoch,
        save_animation=args.save_animation,
        live_visualization=args.live_viz
    )
