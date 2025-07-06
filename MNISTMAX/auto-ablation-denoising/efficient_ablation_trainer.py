"""
Efficient training pipeline for ablation denoising with live visualization.
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
from autoencoder_denoising.bitmap_utils import convert_to_bitmap, visualize_bitmap_comparison

from ablation_noise import AdditiveNoiseGenerator, create_additive_noise_dataset
from ablation_loss import get_ablation_loss, get_ablation_metrics, compute_comprehensive_ablation_metrics
from ablation_models import create_ablation_model


class EfficientAblationTrainer:
    """Efficient trainer for ablation denoising models."""
    
    def __init__(self, 
                 model: keras.Model,
                 loss_fn: callable,
                 optimizer: keras.optimizers.Optimizer,
                 metrics: list,
                 log_dir: str = "logs/ablation",
                 save_animation: bool = False,
                 live_visualization: bool = False):
        """
        Initialize the efficient ablation trainer.
        
        Args:
            model: Ablation denoising model to train
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
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_f1_score': [],
            'val_f1_score': [],
            'train_specificity': [],
            'val_specificity': [],
            'learning_rate': []
        }
    
    def _setup_animation_samples(self):
        """Setup fixed samples for animation."""
        # Load a few test samples for consistent animation
        (_, _), (x_test, _) = load_mnist_data(normalize=True)
        x_test_binary = convert_to_bitmap(x_test, threshold=0.5)
        
        # Select 3 diverse samples for animation
        self.animation_clean = x_test_binary[42:45]  # Fixed indices for consistency
        
        # Create noisy versions using additive noise
        noise_gen = AdditiveNoiseGenerator()
        self.animation_noisy = noise_gen.add_mixed_noise(
            self.animation_clean, intensity=0.6
        )
    
    def _setup_live_visualization(self):
        """Setup live visualization window."""
        plt.ion()  # Turn on interactive mode
        self.live_fig, self.live_axes = plt.subplots(3, 4, figsize=(12, 9))
        self.live_fig.suptitle('Live Ablation Denoising Training Progress', fontsize=16, fontweight='bold')
        
        # Set up subplot titles
        titles = ['Clean', 'Noisy', 'Denoised', 'Ablation Map']
        for j, title in enumerate(titles):
            self.live_axes[0, j].set_title(title, fontsize=12)
        
        for i in range(3):
            for j in range(4):
                self.live_axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def _update_live_visualization(self, epoch: int, train_loss: float, val_loss: float, 
                                 train_precision: float, val_precision: float):
        """Update live visualization window."""
        if not self.live_visualization:
            return
            
        # Get current model predictions
        noisy_input = np.expand_dims(self.animation_noisy, -1)
        denoised = self.model.predict(noisy_input, verbose=0)
        denoised = denoised.squeeze()
        
        # Compute ablation maps (what was removed)
        ablation_maps = self.animation_noisy - denoised
        
        # Clear and update plots
        for i in range(3):
            # Clear previous images
            for j in range(4):
                self.live_axes[i, j].clear()
            
            # Plot new images
            self.live_axes[i, 0].imshow(self.animation_clean[i], cmap='gray', vmin=0, vmax=1)
            self.live_axes[i, 1].imshow(self.animation_noisy[i], cmap='gray', vmin=0, vmax=1)
            self.live_axes[i, 2].imshow(denoised[i], cmap='gray', vmin=0, vmax=1)
            self.live_axes[i, 3].imshow(ablation_maps[i], cmap='Reds', vmin=0, vmax=1)
            
            # Set titles for first row
            if i == 0:
                titles = ['Clean', 'Noisy', 'Denoised', 'Ablated']
                for j, title in enumerate(titles):
                    self.live_axes[i, j].set_title(title, fontsize=12)
            
            # Remove axes
            for j in range(4):
                self.live_axes[i, j].axis('off')
        
        # Update main title with current metrics
        self.live_fig.suptitle(
            f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
            f'Train Precision: {train_precision:.3f} | Val Precision: {val_precision:.3f}',
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
        
        # Compute ablation maps
        ablation_maps = self.animation_noisy - denoised
        
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle(f'Epoch {epoch + 1} - Ablation Denoising Progress', fontsize=16, fontweight='bold')
        
        titles = ['Clean', 'Noisy', 'Denoised', 'Ablated']
        
        for i in range(3):
            # Clean
            axes[i, 0].imshow(self.animation_clean[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(titles[0] if i == 0 else '', fontsize=12)
            axes[i, 0].axis('off')
            
            # Noisy
            axes[i, 1].imshow(self.animation_noisy[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(titles[1] if i == 0 else '', fontsize=12)
            axes[i, 1].axis('off')
            
            # Denoised
            axes[i, 2].imshow(denoised[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(titles[2] if i == 0 else '', fontsize=12)
            axes[i, 2].axis('off')
            
            # Ablation map
            axes[i, 3].imshow(ablation_maps[i], cmap='Reds', vmin=0, vmax=1)
            axes[i, 3].set_title(titles[3] if i == 0 else '', fontsize=12)
            axes[i, 3].axis('off')
        
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
            
        print("Creating ablation training animation...")
        
        # Load images
        images = []
        for frame_path in self.animation_frames:
            img = Image.open(frame_path)
            images.append(img)
        
        # Create GIF
        gif_path = os.path.join(self.log_dir, "ablation_training_animation.gif")
        
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
        
        print(f"Ablation training animation saved: {gif_path}")
    
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
        Train the ablation denoising model efficiently.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs to train
            steps_per_epoch: Steps per epoch (if None, use full dataset)
            validation_steps: Validation steps (if None, use full dataset)
            save_freq: Frequency to save model checkpoints
            verbose: Verbosity level
        """
        print(f"Starting efficient ablation denoising training for {epochs} epochs...")
        print(f"Model type: {type(self.model).__name__}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            if verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = {
                'loss': tf.keras.metrics.Mean(),
                'precision_score': tf.keras.metrics.Mean(),
                'recall_score': tf.keras.metrics.Mean(),
                'f1_score': tf.keras.metrics.Mean(),
                'specificity_score': tf.keras.metrics.Mean()
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
                'precision_score': tf.keras.metrics.Mean(),
                'recall_score': tf.keras.metrics.Mean(),
                'f1_score': tf.keras.metrics.Mean(),
                'specificity_score': tf.keras.metrics.Mean()
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
            epoch_train_precision = float(train_metrics['precision_score'].result())
            epoch_val_precision = float(val_metrics['precision_score'].result())
            epoch_train_recall = float(train_metrics['recall_score'].result())
            epoch_val_recall = float(val_metrics['recall_score'].result())
            epoch_train_f1 = float(train_metrics['f1_score'].result())
            epoch_val_f1 = float(val_metrics['f1_score'].result())
            epoch_train_specificity = float(train_metrics['specificity_score'].result())
            epoch_val_specificity = float(val_metrics['specificity_score'].result())
            
            # Update history
            self.history['train_loss'].append(epoch_train_loss)
            self.history['val_loss'].append(epoch_val_loss)
            self.history['train_precision'].append(epoch_train_precision)
            self.history['val_precision'].append(epoch_val_precision)
            self.history['train_recall'].append(epoch_train_recall)
            self.history['val_recall'].append(epoch_val_recall)
            self.history['train_f1_score'].append(epoch_train_f1)
            self.history['val_f1_score'].append(epoch_val_f1)
            self.history['train_specificity'].append(epoch_train_specificity)
            self.history['val_specificity'].append(epoch_val_specificity)
            self.history['learning_rate'].append(float(self.optimizer.learning_rate))
            
            # Print epoch results
            if verbose >= 1:
                print(f"  Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
                print(f"  Train Precision: {epoch_train_precision:.4f}, Val Precision: {epoch_val_precision:.4f}")
                print(f"  Train Recall: {epoch_train_recall:.4f}, Val Recall: {epoch_val_recall:.4f}")
                print(f"  Train F1: {epoch_train_f1:.4f}, Val F1: {epoch_val_f1:.4f}")
                print(f"  Train Specificity: {epoch_train_specificity:.4f}, Val Specificity: {epoch_val_specificity:.4f}")
                print()
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                self.save_model(os.path.join(self.log_dir, "best_model.weights.h5"))
                if verbose >= 1:
                    print(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
            
            # Update live visualization
            self._update_live_visualization(epoch, epoch_train_loss, epoch_val_loss, 
                                          epoch_train_precision, epoch_val_precision)
            
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
        
        print("Efficient ablation training completed!")
        return self.history
    
    def save_model(self, filepath: str):
        """Save the model."""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load the model."""
        self.model.load_weights(filepath)


def train_efficient_ablation_denoiser(model_type: str = "basic",
                                    noise_type: str = "mixed",
                                    epochs: int = 20,
                                    batch_size: int = 64,
                                    learning_rate: float = 1e-3,
                                    noise_intensity: float = 0.5,
                                    loss_type: str = "precision_bce",
                                    bitmap_threshold: float = 0.5,
                                    log_dir: Optional[str] = None,
                                    steps_per_epoch: Optional[int] = None,
                                    save_animation: bool = False,
                                    live_visualization: bool = False) -> Tuple[keras.Model, Dict]:
    """
    Train an ablation denoising model efficiently on MNIST.
    
    Args:
        model_type: Type of ablation model
        noise_type: Type of additive noise
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        noise_intensity: Noise intensity level
        loss_type: Type of loss function
        bitmap_threshold: Threshold for converting to binary
        log_dir: Directory for logging
        steps_per_epoch: Steps per epoch (None for full dataset)
        save_animation: Save training animation
        live_visualization: Show live visualization
        
    Returns:
        Trained model and training history
    """
    # Set up logging directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/ablation_{model_type}_{noise_type}_{timestamp}"
    
    print("Efficient Ablation Denoising Training Configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Noise type: {noise_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Noise intensity: {noise_intensity}")
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
    
    # Create efficient datasets with additive noise
    print("Creating efficient data pipelines with additive noise...")
    train_dataset = create_additive_noise_dataset(
        x_train_binary, 
        batch_size=batch_size, 
        noise_intensity=noise_intensity, 
        noise_type=noise_type
    )
    
    val_dataset = create_additive_noise_dataset(
        x_test_binary, 
        batch_size=batch_size, 
        noise_intensity=noise_intensity, 
        noise_type=noise_type
    )
    
    # Create model
    print("Creating ablation model...")
    model = create_ablation_model(model_type=model_type)
    
    # Create loss function and metrics
    loss_fn = get_ablation_loss(loss_type)
    metrics = get_ablation_metrics()
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create trainer
    trainer = EfficientAblationTrainer(
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
    
    # Add noise for evaluation
    noise_gen = AdditiveNoiseGenerator()
    noisy_samples = noise_gen.add_mixed_noise(test_samples, intensity=noise_intensity)
    
    # Get predictions
    noisy_input = np.expand_dims(noisy_samples, -1)
    denoised = model.predict(noisy_input, verbose=0)
    denoised = denoised.squeeze()
    
    # Compute comprehensive metrics
    metrics_result = compute_comprehensive_ablation_metrics(
        test_samples, denoised, noisy_samples
    )
    
    print("Ablation Denoising Results:")
    print(f"  Precision: {metrics_result['precision']:.4f}")
    print(f"  Recall: {metrics_result['recall']:.4f}")
    print(f"  F1 Score: {metrics_result['f1_score']:.4f}")
    print(f"  Specificity: {metrics_result['specificity']:.4f}")
    print(f"  Ablation Efficiency: {metrics_result['ablation_efficiency']:.4f}")
    print(f"  Pixel Conservation: {metrics_result['pixel_conservation_rate']:.4f}")
    
    # Save visualization
    viz_path = os.path.join(log_dir, "ablation_evaluation_samples.png")
    visualize_bitmap_comparison(
        test_samples[:5], 
        noisy_samples[:5], 
        denoised[:5],
        save_path=viz_path
    )
    
    print("Efficient ablation training completed successfully!")
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ablation denoising model efficiently on MNIST')
    parser.add_argument('--model', type=str, default='basic', 
                       choices=['basic', 'unet', 'attention', 'residual', 'vae'],
                       help='Model architecture')
    parser.add_argument('--noise', type=str, default='mixed',
                       choices=['random', 'blocks', 'mixed'],
                       help='Additive noise type')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--noise_intensity', type=float, default=0.5,
                       help='Noise intensity (0.0-1.0)')
    parser.add_argument('--loss', type=str, default='precision_bce',
                       choices=['precision_bce', 'weighted_bce', 'dice', 'fp_penalty', 'combined'],
                       help='Loss function type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Bitmap threshold')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                       help='Steps per epoch (None for full dataset)')
    parser.add_argument('--save_animation', action='store_true',
                       help='Save training animation showing ablation progress')
    parser.add_argument('--live_viz', action='store_true',
                       help='Show live visualization during training')
    
    args = parser.parse_args()
    
    # Calculate steps per epoch for ~1000 samples if not specified
    if args.steps_per_epoch is None and args.epochs > 10:
        args.steps_per_epoch = max(16, 1000 // args.batch_size)  # ~1000 samples per epoch
        print(f"Using {args.steps_per_epoch} steps per epoch (~{args.steps_per_epoch * args.batch_size} samples)")
    
    # Train model
    model, history = train_efficient_ablation_denoiser(
        model_type=args.model,
        noise_type=args.noise,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        noise_intensity=args.noise_intensity,
        loss_type=args.loss,
        bitmap_threshold=args.threshold,
        steps_per_epoch=args.steps_per_epoch,
        save_animation=args.save_animation,
        live_visualization=args.live_viz
    )
