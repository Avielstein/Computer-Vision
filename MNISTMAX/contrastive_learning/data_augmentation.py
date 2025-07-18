"""
Data augmentation utilities for contrastive learning on MNIST.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List


class ContrastiveAugmentation:
    """Data augmentation pipeline for contrastive learning."""
    
    def __init__(self, 
                 rotation_range: float = 20.0,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1,
                 zoom_range: float = 0.1,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 noise_factor: float = 0.1):
        """
        Initialize augmentation parameters.
        
        Args:
            rotation_range: Range of random rotations in degrees
            width_shift_range: Range of horizontal shifts as fraction of width
            height_shift_range: Range of vertical shifts as fraction of height
            zoom_range: Range of random zoom
            brightness_range: Range of brightness adjustment
            noise_factor: Standard deviation of Gaussian noise
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.noise_factor = noise_factor
    
    def random_rotation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random rotation to image."""
        # For MNIST, we'll use simple 90-degree rotations
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = tf.squeeze(image)
        
        # Add batch and channel dimensions for tf.image.rot90
        image_4d = tf.expand_dims(tf.expand_dims(image, 0), -1)
        
        # Apply rotation
        rotated = tf.image.rot90(image_4d, k=k)
        
        # Remove batch and channel dimensions
        return tf.squeeze(rotated)
    
    def random_shift(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random translation to image."""
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        
        # Calculate shift amounts
        dx = tf.random.uniform([], -self.width_shift_range, self.width_shift_range)
        dy = tf.random.uniform([], -self.height_shift_range, self.height_shift_range)
        
        # Convert to pixel values
        dx_pixels = tf.cast(dx * tf.cast(width, tf.float32), tf.int32)
        dy_pixels = tf.cast(dy * tf.cast(height, tf.float32), tf.int32)
        
        # Apply translation
        return tf.roll(tf.roll(image, dx_pixels, axis=1), dy_pixels, axis=0)
    
    def random_zoom(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random zoom to image."""
        zoom_factor = tf.random.uniform([], 1.0 - self.zoom_range, 1.0 + self.zoom_range)
        
        # Get image dimensions
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        
        # Calculate new dimensions
        new_height = tf.cast(tf.cast(height, tf.float32) * zoom_factor, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) * zoom_factor, tf.int32)
        
        # Resize and crop/pad to original size
        resized = tf.image.resize(image[..., tf.newaxis], [new_height, new_width])
        resized = tf.squeeze(resized, axis=-1)
        
        # Center crop or pad to original size
        return tf.image.resize_with_crop_or_pad(resized[..., tf.newaxis], height, width)[..., 0]
    
    def random_brightness(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random brightness adjustment."""
        brightness_factor = tf.random.uniform([], 
                                            self.brightness_range[0], 
                                            self.brightness_range[1])
        return tf.clip_by_value(image * brightness_factor, 0.0, 1.0)
    
    def add_noise(self, image: tf.Tensor) -> tf.Tensor:
        """Add Gaussian noise to image."""
        noise = tf.random.normal(tf.shape(image), stddev=self.noise_factor)
        return tf.clip_by_value(image + noise, 0.0, 1.0)
    
    def elastic_transform(self, image: tf.Tensor, alpha: float = 34, sigma: float = 4) -> tf.Tensor:
        """Apply elastic deformation to image."""
        # Generate random displacement fields
        random_state = np.random.RandomState(None)
        shape = image.shape
        
        dx = tf.random.normal(shape, stddev=sigma)
        dy = tf.random.normal(shape, stddev=sigma)
        
        # Apply Gaussian filter (approximation using conv2d)
        kernel_size = int(4 * sigma) | 1  # Ensure odd kernel size
        kernel = tf.constant(self._gaussian_kernel_2d(kernel_size, sigma), dtype=tf.float32)
        kernel = kernel[..., tf.newaxis, tf.newaxis]
        
        dx = tf.nn.conv2d(dx[tf.newaxis, ..., tf.newaxis], kernel, 
                         strides=[1, 1, 1, 1], padding='SAME')[0, ..., 0]
        dy = tf.nn.conv2d(dy[tf.newaxis, ..., tf.newaxis], kernel, 
                         strides=[1, 1, 1, 1], padding='SAME')[0, ..., 0]
        
        dx *= alpha
        dy *= alpha
        
        # Create coordinate grids
        x, y = tf.meshgrid(tf.range(shape[1], dtype=tf.float32), 
                          tf.range(shape[0], dtype=tf.float32))
        
        # Apply displacement
        indices = tf.stack([y + dy, x + dx], axis=-1)
        
        # Sample from original image using bilinear interpolation
        return tf.gather_nd(image, tf.cast(indices, tf.int32))
    
    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float) -> np.ndarray:
        """Generate 2D Gaussian kernel."""
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        return kernel / kernel.sum()
    
    def augment_pair(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate two different augmented versions of the same image.
        
        Args:
            image: Input image tensor of shape (28, 28)
            
        Returns:
            Tuple of two augmented images
        """
        # Ensure image is float32 and normalized
        image = tf.cast(image, tf.float32)
        if tf.reduce_max(image) > 1.0:
            image = image / 255.0
        
        # Generate first augmented view
        aug1 = image
        if tf.random.uniform([]) > 0.5:
            aug1 = self.random_rotation(aug1)
        if tf.random.uniform([]) > 0.5:
            aug1 = self.random_shift(aug1)
        if tf.random.uniform([]) > 0.5:
            aug1 = self.random_zoom(aug1)
        if tf.random.uniform([]) > 0.5:
            aug1 = self.random_brightness(aug1)
        if tf.random.uniform([]) > 0.5:
            aug1 = self.add_noise(aug1)
        
        # Generate second augmented view
        aug2 = image
        if tf.random.uniform([]) > 0.5:
            aug2 = self.random_rotation(aug2)
        if tf.random.uniform([]) > 0.5:
            aug2 = self.random_shift(aug2)
        if tf.random.uniform([]) > 0.5:
            aug2 = self.random_zoom(aug2)
        if tf.random.uniform([]) > 0.5:
            aug2 = self.random_brightness(aug2)
        if tf.random.uniform([]) > 0.5:
            aug2 = self.add_noise(aug2)
        
        return aug1, aug2
    
    def create_contrastive_dataset(self, 
                                 x_data: np.ndarray, 
                                 batch_size: int = 256) -> tf.data.Dataset:
        """
        Create a dataset for contrastive learning.
        
        Args:
            x_data: Input images of shape (N, 28, 28)
            batch_size: Batch size for the dataset
            
        Returns:
            TensorFlow dataset yielding pairs of augmented images
        """
        def augment_fn(image):
            aug1, aug2 = self.augment_pair(image)
            return aug1, aug2
        
        dataset = tf.data.Dataset.from_tensor_slices(x_data)
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_simclr_augmentation() -> ContrastiveAugmentation:
    """Create augmentation pipeline similar to SimCLR paper."""
    return ContrastiveAugmentation(
        rotation_range=30.0,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        brightness_range=(0.6, 1.4),
        noise_factor=0.15
    )


def create_mild_augmentation() -> ContrastiveAugmentation:
    """Create milder augmentation for MNIST."""
    return ContrastiveAugmentation(
        rotation_range=15.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.8, 1.2),
        noise_factor=0.05
    )


def create_augmentation_pipeline(strength: str = 'moderate'):
    """
    Create augmentation pipeline based on strength level.
    
    Args:
        strength: Augmentation strength ('light', 'moderate', 'strong')
        
    Returns:
        Augmentation function
    """
    if strength == 'light':
        augmenter = ContrastiveAugmentation(
            rotation_range=10.0,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            brightness_range=(0.9, 1.1),
            noise_factor=0.02
        )
    elif strength == 'moderate':
        augmenter = create_mild_augmentation()
    elif strength == 'strong':
        augmenter = create_simclr_augmentation()
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")
    
    def augment_fn(image):
        """Apply augmentation to a single image."""
        aug1, aug2 = augmenter.augment_pair(image)
        return aug1  # Return just one augmented version
    
    return augment_fn
