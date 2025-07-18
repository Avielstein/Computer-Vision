"""
Data utilities for MNIST processing across different tasks.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional


def load_mnist_data(normalize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST dataset with optional normalization.
    
    Args:
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)


def convert_to_bitmap(images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert grayscale images to pure binary (0/1) bitmaps.
    
    Args:
        images: Input images of shape (N, H, W) or (N, H, W, 1)
        threshold: Threshold for binarization
        
    Returns:
        Binary images with values 0 or 1
    """
    return (images > threshold).astype(np.float32)


def preprocess_for_contrastive(x_train: np.ndarray, 
                             x_test: np.ndarray,
                             add_channel_dim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess MNIST data for contrastive learning.
    
    Args:
        x_train: Training images
        x_test: Test images  
        add_channel_dim: Whether to add channel dimension for CNN
        
    Returns:
        Preprocessed training and test images
    """
    # Ensure float32 and normalized
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    if x_train.max() > 1.0:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    
    # Add channel dimension if needed
    if add_channel_dim and len(x_train.shape) == 3:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    
    return x_train, x_test


def create_contrastive_dataset(x_data: np.ndarray, 
                             batch_size: int = 256,
                             shuffle: bool = True) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset for unsupervised contrastive learning.
    
    Args:
        x_data: Input images
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices(x_data)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_denoising_dataset(x_clean: np.ndarray,
                           noise_fn,
                           batch_size: int = 256,
                           clean_ratio: float = 0.5) -> tf.data.Dataset:
    """
    Create a dataset for denoising autoencoder training.
    
    Args:
        x_clean: Clean images
        noise_fn: Function to add noise to images
        batch_size: Batch size
        clean_ratio: Ratio of clean images in each batch (rest will be noisy)
        
    Returns:
        TensorFlow dataset yielding (input, target) pairs
    """
    def generate_batch():
        """Generator function for mixed clean/noisy batches."""
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
                batch_input[~clean_mask] = noise_fn(batch_clean[~clean_mask])
                
                # Target is always clean
                batch_target = batch_clean
                
                yield batch_input, batch_target
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generate_batch,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 28, 28), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 28, 28), dtype=tf.float32)
        )
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)


def create_contrastive_pairs(x_data: np.ndarray,
                           batch_size: int = 256,
                           augmentation_fn=None) -> tf.data.Dataset:
    """
    Create contrastive pairs dataset for self-supervised learning.
    
    Args:
        x_data: Input images
        batch_size: Batch size
        augmentation_fn: Augmentation function to create positive pairs
        
    Returns:
        Dataset yielding (anchor, positive) pairs
    """
    def simple_augment(x):
        """Simple augmentation that works in graph mode."""
        # Add small amount of noise
        noise = tf.random.normal(tf.shape(x), stddev=0.05)
        augmented = tf.clip_by_value(x + noise, 0.0, 1.0)
        
        # Random horizontal flip (for variety)
        if tf.random.uniform([]) > 0.5:
            augmented = tf.image.flip_left_right(tf.expand_dims(augmented, -1))
            augmented = tf.squeeze(augmented, -1)
        
        return augmented
    
    def augment_pair(x):
        """Create augmented pair from single image."""
        # Create two different augmented versions
        anchor = simple_augment(x)
        positive = simple_augment(x)
        return anchor, positive
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(x_data)
    dataset = dataset.shuffle(buffer_size=10000)
    
    # Create pairs through augmentation
    dataset = dataset.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def save_representations(embeddings: np.ndarray, 
                        labels: np.ndarray,
                        images: np.ndarray,
                        save_path: str):
    """
    Save learned representations to file.
    
    Args:
        embeddings: Learned feature representations
        labels: Corresponding labels
        images: Original images
        save_path: Path to save file
    """
    np.savez_compressed(save_path, 
                       embeddings=embeddings,
                       labels=labels,
                       images=images)
    print(f"Representations saved to {save_path}")


def load_representations(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load saved representations from file.
    
    Args:
        filepath: Path to saved file
        
    Returns:
        (representations, labels)
    """
    data = np.load(filepath)
    return data['representations'], data['labels']


def compute_dataset_stats(images: np.ndarray) -> dict:
    """
    Compute basic statistics for a dataset.
    
    Args:
        images: Input images
        
    Returns:
        Dictionary with statistics
    """
    return {
        'shape': images.shape,
        'dtype': images.dtype,
        'min': float(images.min()),
        'max': float(images.max()),
        'mean': float(images.mean()),
        'std': float(images.std()),
        'n_samples': len(images)
    }
