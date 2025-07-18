"""
Visualization utilities for MNIST experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import threading
import time


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)):
    """
    Plot training history with loss and other metrics.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
        figsize: Figure size
    """
    n_metrics = len([k for k in history.keys() if not k.startswith('val_')])
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    metric_idx = 0
    for key, values in history.items():
        if key.startswith('val_'):
            continue
            
        ax = axes[metric_idx]
        
        # Plot training metric
        ax.plot(values, label=f'Train {key}', alpha=0.8)
        
        # Plot validation metric if available
        val_key = f'val_{key}'
        if val_key in history:
            ax.plot(history[val_key], label=f'Val {key}', alpha=0.8)
        
        ax.set_title(key.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        metric_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_image_grid(images: np.ndarray, 
                   titles: Optional[List[str]] = None,
                   rows: int = 2, 
                   cols: int = 5,
                   figsize: Tuple[int, int] = (15, 6),
                   cmap: str = 'gray',
                   save_path: Optional[str] = None):
    """
    Plot a grid of images.
    
    Args:
        images: Array of images to plot
        titles: Optional titles for each image
        rows: Number of rows
        cols: Number of columns
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i in range(min(len(images), rows * cols)):
        axes[i].imshow(images[i], cmap=cmap)
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_denoising_comparison(clean: np.ndarray,
                            noisy: np.ndarray, 
                            denoised: np.ndarray,
                            n_samples: int = 5,
                            figsize: Tuple[int, int] = (15, 9),
                            save_path: Optional[str] = None):
    """
    Plot comparison of clean, noisy, and denoised images.
    
    Args:
        clean: Clean images
        noisy: Noisy images
        denoised: Denoised images
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, n_samples, figsize=figsize)
    
    row_titles = ['Clean', 'Noisy', 'Denoised']
    
    for i in range(n_samples):
        # Clean images
        axes[0, i].imshow(clean[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Sample {i+1}')
        axes[0, i].axis('off')
        
        # Noisy images
        axes[1, i].imshow(noisy[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        
        # Denoised images
        axes[2, i].imshow(denoised[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
    
    # Add row labels
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, rotation=90, size='large')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


class LiveTrainingVisualizer:
    """
    Real-time visualization during training.
    """
    
    def __init__(self, 
                 update_freq: int = 10,
                 figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize the live visualizer.
        
        Args:
            update_freq: Update frequency in training steps
            figsize: Figure size
        """
        self.update_freq = update_freq
        self.figsize = figsize
        self.step_count = 0
        
        # Training history
        self.loss_history = []
        self.step_history = []
        
        # Current samples
        self.current_clean = None
        self.current_noisy = None
        self.current_denoised = None
        
        # Setup the plot
        self.setup_plot()
        
        # Threading for non-blocking updates
        self.update_thread = None
        self.should_update = False
        
    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        plt.ion()  # Turn on interactive mode
        
        self.fig = plt.figure(figsize=self.figsize)
        
        # Create grid layout
        gs = self.fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
        
        # Loss plot (spans 2 columns)
        self.loss_ax = self.fig.add_subplot(gs[0, :2])
        self.loss_ax.set_title('Training Loss')
        self.loss_ax.set_xlabel('Step')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True, alpha=0.3)
        
        # Sample images (3 rows, 4 columns for images)
        self.image_axes = []
        titles = ['Clean 1', 'Noisy 1', 'Denoised 1', 'Error 1',
                 'Clean 2', 'Noisy 2', 'Denoised 2', 'Error 2',
                 'Clean 3', 'Noisy 3', 'Denoised 3', 'Error 3']
        
        for i in range(3):
            row_axes = []
            for j in range(4):
                ax = self.fig.add_subplot(gs[i, j+2])
                ax.set_title(titles[i*4 + j])
                ax.axis('off')
                row_axes.append(ax)
            self.image_axes.append(row_axes)
        
        plt.show(block=False)
        
    def update_loss(self, loss: float):
        """Update loss history."""
        self.step_count += 1
        self.loss_history.append(loss)
        self.step_history.append(self.step_count)
        
        if self.step_count % self.update_freq == 0:
            self.should_update = True
    
    def update_samples(self, clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray):
        """Update current sample images."""
        self.current_clean = clean[:3]  # Take first 3 samples
        self.current_noisy = noisy[:3]
        self.current_denoised = denoised[:3]
        
        if self.step_count % self.update_freq == 0:
            self.should_update = True
    
    def refresh_display(self):
        """Refresh the display with current data."""
        if not self.should_update:
            return
            
        try:
            # Update loss plot
            if len(self.loss_history) > 0:
                self.loss_ax.clear()
                self.loss_ax.plot(self.step_history, self.loss_history, 'b-', alpha=0.7)
                self.loss_ax.set_title(f'Training Loss (Step {self.step_count})')
                self.loss_ax.set_xlabel('Step')
                self.loss_ax.set_ylabel('Loss')
                self.loss_ax.grid(True, alpha=0.3)
            
            # Update sample images
            if (self.current_clean is not None and 
                self.current_noisy is not None and 
                self.current_denoised is not None):
                
                for i in range(3):
                    # Clean image
                    self.image_axes[i][0].clear()
                    self.image_axes[i][0].imshow(self.current_clean[i], cmap='gray', vmin=0, vmax=1)
                    self.image_axes[i][0].set_title(f'Clean {i+1}')
                    self.image_axes[i][0].axis('off')
                    
                    # Noisy image
                    self.image_axes[i][1].clear()
                    self.image_axes[i][1].imshow(self.current_noisy[i], cmap='gray', vmin=0, vmax=1)
                    self.image_axes[i][1].set_title(f'Noisy {i+1}')
                    self.image_axes[i][1].axis('off')
                    
                    # Denoised image
                    self.image_axes[i][2].clear()
                    self.image_axes[i][2].imshow(self.current_denoised[i], cmap='gray', vmin=0, vmax=1)
                    self.image_axes[i][2].set_title(f'Denoised {i+1}')
                    self.image_axes[i][2].axis('off')
                    
                    # Error map (absolute difference)
                    error = np.abs(self.current_clean[i] - self.current_denoised[i])
                    self.image_axes[i][3].clear()
                    self.image_axes[i][3].imshow(error, cmap='hot', vmin=0, vmax=1)
                    self.image_axes[i][3].set_title(f'Error {i+1}')
                    self.image_axes[i][3].axis('off')
            
            # Refresh the display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            self.should_update = False
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def close(self):
        """Close the visualization."""
        plt.ioff()
        plt.close(self.fig)


def plot_embeddings_2d(embeddings: np.ndarray,
                      labels: np.ndarray,
                      method: str = 'tsne',
                      figsize: Tuple[int, int] = (10, 8),
                      save_path: Optional[str] = None):
    """
    Plot 2D visualization of embeddings using t-SNE or UMAP.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Corresponding labels
        method: Dimensionality reduction method ('tsne' or 'umap')
        figsize: Figure size
        save_path: Path to save the plot
    """
    try:
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == 'umap':
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensionality
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=figsize)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'{method.upper()} Visualization of Learned Embeddings')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError as e:
        print(f"Required library not installed: {e}")
        print("Install with: pip install scikit-learn umap-learn")


def plot_noise_examples(clean_images: np.ndarray,
                       noise_functions: Dict[str, callable],
                       n_samples: int = 3,
                       figsize: Tuple[int, int] = (15, 10)):
    """
    Plot examples of different noise types.
    
    Args:
        clean_images: Clean input images
        noise_functions: Dictionary of noise functions
        n_samples: Number of samples to show
        figsize: Figure size
    """
    n_noise_types = len(noise_functions)
    fig, axes = plt.subplots(n_samples, n_noise_types + 1, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    axes[0, 0].set_title('Clean', fontsize=12, fontweight='bold')
    for j, noise_name in enumerate(noise_functions.keys()):
        axes[0, j + 1].set_title(noise_name, fontsize=12, fontweight='bold')
    
    for i in range(n_samples):
        # Clean image
        axes[i, 0].imshow(clean_images[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        
        # Noisy versions
        for j, (noise_name, noise_fn) in enumerate(noise_functions.items()):
            noisy = noise_fn(clean_images[i:i+1])[0]  # Apply to single image
            axes[i, j + 1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.show()
