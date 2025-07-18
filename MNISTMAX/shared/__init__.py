"""
Shared utilities for MNISTMAX framework.

This package provides common functionality used across different components:
- data_processing: Data loading, preprocessing, and dataset creation utilities
- visualization: Plotting, live training visualization, and result display utilities
"""

# Import from submodules for backward compatibility
from .data_processing import *
from .visualization import *

__version__ = "1.0.0"

__all__ = [
    # Data processing utilities
    'load_mnist_data',
    'convert_to_bitmap',
    'preprocess_for_contrastive',
    'create_contrastive_dataset',
    'create_denoising_dataset',
    'create_contrastive_pairs',
    'save_representations',
    'load_representations',
    'compute_dataset_stats',
    
    # Visualization utilities
    'plot_training_history',
    'plot_image_grid',
    'plot_denoising_comparison',
    'LiveTrainingVisualizer',
    'plot_embeddings_2d',
    'plot_noise_examples'
]
