"""
Noise generation utilities for denoising autoencoder training.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Callable, Dict, Optional
import random


def salt_and_pepper_noise(images: np.ndarray, 
                         noise_prob: float = 0.1,
                         salt_prob: float = 0.5) -> np.ndarray:
    """
    Add salt and pepper noise to binary images.
    
    Args:
        images: Input images of shape (N, H, W) or (N, H, W, 1)
        noise_prob: Probability of a pixel being corrupted
        salt_prob: Probability of corrupted pixel being salt (1) vs pepper (0)
        
    Returns:
        Noisy images
    """
    noisy = images.copy()
    
    # Generate noise mask
    noise_mask = np.random.random(images.shape) < noise_prob
    
    # Generate salt/pepper values
    salt_mask = np.random.random(images.shape) < salt_prob
    
    # Apply noise
    noisy[noise_mask & salt_mask] = 1.0  # Salt
    noisy[noise_mask & ~salt_mask] = 0.0  # Pepper
    
    return noisy


def random_pixel_flip_noise(images: np.ndarray, 
                           flip_prob: float = 0.05) -> np.ndarray:
    """
    Randomly flip pixels in binary images.
    
    Args:
        images: Input binary images
        flip_prob: Probability of flipping each pixel
        
    Returns:
        Noisy images with flipped pixels
    """
    noisy = images.copy()
    
    # Generate flip mask
    flip_mask = np.random.random(images.shape) < flip_prob
    
    # Flip pixels (0 -> 1, 1 -> 0)
    noisy[flip_mask] = 1.0 - noisy[flip_mask]
    
    return noisy


def gaussian_noise_binary(images: np.ndarray, 
                         noise_std: float = 0.1,
                         threshold: float = 0.5) -> np.ndarray:
    """
    Add Gaussian noise and re-binarize.
    
    Args:
        images: Input binary images
        noise_std: Standard deviation of Gaussian noise
        threshold: Threshold for re-binarization
        
    Returns:
        Noisy binary images
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, images.shape)
    noisy = images + noise
    
    # Re-binarize
    noisy = (noisy > threshold).astype(np.float32)
    
    return noisy


def structured_noise_lines(images: np.ndarray, 
                          num_lines: int = 3,
                          line_width: int = 1) -> np.ndarray:
    """
    Add structured noise in the form of random lines.
    
    Args:
        images: Input images
        num_lines: Number of random lines to add
        line_width: Width of the lines
        
    Returns:
        Images with line noise
    """
    noisy = images.copy()
    
    for i in range(len(images)):
        img = noisy[i]
        h, w = img.shape[:2]
        
        for _ in range(num_lines):
            # Random line parameters
            if np.random.random() < 0.5:  # Horizontal line
                y = np.random.randint(0, h)
                x_start = np.random.randint(0, w // 2)
                x_end = np.random.randint(w // 2, w)
                
                for lw in range(line_width):
                    if y + lw < h:
                        img[y + lw, x_start:x_end] = np.random.choice([0.0, 1.0])
            else:  # Vertical line
                x = np.random.randint(0, w)
                y_start = np.random.randint(0, h // 2)
                y_end = np.random.randint(h // 2, h)
                
                for lw in range(line_width):
                    if x + lw < w:
                        img[y_start:y_end, x + lw] = np.random.choice([0.0, 1.0])
    
    return noisy


def block_noise(images: np.ndarray, 
               num_blocks: int = 5,
               block_size_range: Tuple[int, int] = (2, 6)) -> np.ndarray:
    """
    Add block-shaped noise to images.
    
    Args:
        images: Input images
        num_blocks: Number of noise blocks to add
        block_size_range: Range of block sizes (min, max)
        
    Returns:
        Images with block noise
    """
    noisy = images.copy()
    
    for i in range(len(images)):
        img = noisy[i]
        h, w = img.shape[:2]
        
        for _ in range(num_blocks):
            # Random block parameters
            block_h = np.random.randint(block_size_range[0], block_size_range[1] + 1)
            block_w = np.random.randint(block_size_range[0], block_size_range[1] + 1)
            
            y = np.random.randint(0, max(1, h - block_h))
            x = np.random.randint(0, max(1, w - block_w))
            
            # Fill block with random value
            block_value = np.random.choice([0.0, 1.0])
            img[y:y + block_h, x:x + block_w] = block_value
    
    return noisy


def speckle_noise(images: np.ndarray, 
                 noise_intensity: float = 0.1) -> np.ndarray:
    """
    Add speckle noise to images.
    
    Args:
        images: Input images
        noise_intensity: Intensity of speckle noise
        
    Returns:
        Images with speckle noise
    """
    noise = np.random.randn(*images.shape) * noise_intensity
    noisy = images + images * noise
    
    # Clip and re-binarize
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy > 0.5).astype(np.float32)
    
    return noisy


def erosion_dilation_noise(images: np.ndarray, 
                          erosion_prob: float = 0.3,
                          dilation_prob: float = 0.3,
                          kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological erosion/dilation as noise.
    
    Args:
        images: Input binary images
        erosion_prob: Probability of applying erosion
        dilation_prob: Probability of applying dilation
        kernel_size: Size of morphological kernel
        
    Returns:
        Images with morphological noise
    """
    try:
        import cv2
        
        noisy = images.copy()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for i in range(len(images)):
            img = (noisy[i] * 255).astype(np.uint8)
            
            if np.random.random() < erosion_prob:
                img = cv2.erode(img, kernel, iterations=1)
            elif np.random.random() < dilation_prob:
                img = cv2.dilate(img, kernel, iterations=1)
            
            noisy[i] = img.astype(np.float32) / 255.0
        
        return noisy
        
    except ImportError:
        print("OpenCV not available, using simple erosion/dilation approximation")
        return random_pixel_flip_noise(images, flip_prob=0.05)


class NoiseGenerator:
    """Configurable noise generator for training."""
    
    def __init__(self, noise_types: Dict[str, Dict] = None):
        """
        Initialize noise generator.
        
        Args:
            noise_types: Dictionary of noise types and their parameters
        """
        if noise_types is None:
            self.noise_types = {
                'salt_pepper': {'noise_prob': 0.1, 'salt_prob': 0.5},
                'pixel_flip': {'flip_prob': 0.05},
                'gaussian': {'noise_std': 0.1, 'threshold': 0.5},
                'lines': {'num_lines': 2, 'line_width': 1},
                'blocks': {'num_blocks': 3, 'block_size_range': (2, 4)},
                'speckle': {'noise_intensity': 0.1}
            }
        else:
            self.noise_types = noise_types
        
        # Map noise type names to functions
        self.noise_functions = {
            'salt_pepper': salt_and_pepper_noise,
            'pixel_flip': random_pixel_flip_noise,
            'gaussian': gaussian_noise_binary,
            'lines': structured_noise_lines,
            'blocks': block_noise,
            'speckle': speckle_noise,
            'morphological': erosion_dilation_noise
        }
    
    def add_noise(self, images: np.ndarray, 
                  noise_type: Optional[str] = None) -> np.ndarray:
        """
        Add noise to images.
        
        Args:
            images: Input images
            noise_type: Specific noise type to use (if None, random)
            
        Returns:
            Noisy images
        """
        if noise_type is None:
            # Randomly select noise type
            noise_type = np.random.choice(list(self.noise_types.keys()))
        
        if noise_type not in self.noise_functions:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noise_fn = self.noise_functions[noise_type]
        noise_params = self.noise_types.get(noise_type, {})
        
        return noise_fn(images, **noise_params)
    
    def add_mixed_noise(self, images: np.ndarray, 
                       num_noise_types: int = 2) -> np.ndarray:
        """
        Add multiple types of noise to images.
        
        Args:
            images: Input images
            num_noise_types: Number of different noise types to apply
            
        Returns:
            Images with mixed noise
        """
        noisy = images.copy()
        
        # Randomly select noise types
        selected_types = np.random.choice(
            list(self.noise_types.keys()), 
            size=min(num_noise_types, len(self.noise_types)),
            replace=False
        )
        
        for noise_type in selected_types:
            noisy = self.add_noise(noisy, noise_type)
        
        return noisy
    
    def create_noise_function(self, noise_type: str = None, 
                            mixed: bool = False) -> Callable:
        """
        Create a noise function for use with datasets.
        
        Args:
            noise_type: Specific noise type (if None, random)
            mixed: Whether to use mixed noise
            
        Returns:
            Noise function
        """
        if mixed:
            return lambda x: self.add_mixed_noise(x)
        else:
            return lambda x: self.add_noise(x, noise_type)


def create_default_noise_generator() -> NoiseGenerator:
    """Create a default noise generator with balanced parameters."""
    return NoiseGenerator()


def create_aggressive_noise_generator() -> NoiseGenerator:
    """Create a noise generator with more aggressive noise parameters."""
    aggressive_params = {
        'salt_pepper': {'noise_prob': 0.2, 'salt_prob': 0.5},
        'pixel_flip': {'flip_prob': 0.1},
        'gaussian': {'noise_std': 0.2, 'threshold': 0.5},
        'lines': {'num_lines': 4, 'line_width': 2},
        'blocks': {'num_blocks': 5, 'block_size_range': (3, 7)},
        'speckle': {'noise_intensity': 0.2}
    }
    return NoiseGenerator(aggressive_params)


def create_mild_noise_generator() -> NoiseGenerator:
    """Create a noise generator with mild noise parameters."""
    mild_params = {
        'salt_pepper': {'noise_prob': 0.05, 'salt_prob': 0.5},
        'pixel_flip': {'flip_prob': 0.02},
        'gaussian': {'noise_std': 0.05, 'threshold': 0.5},
        'lines': {'num_lines': 1, 'line_width': 1},
        'blocks': {'num_blocks': 2, 'block_size_range': (2, 3)},
        'speckle': {'noise_intensity': 0.05}
    }
    return NoiseGenerator(mild_params)


# TensorFlow versions for use in tf.data pipelines
@tf.function
def tf_salt_and_pepper_noise(images: tf.Tensor, 
                            noise_prob: float = 0.1,
                            salt_prob: float = 0.5) -> tf.Tensor:
    """TensorFlow version of salt and pepper noise."""
    noise_mask = tf.random.uniform(tf.shape(images)) < noise_prob
    salt_mask = tf.random.uniform(tf.shape(images)) < salt_prob
    
    # Apply noise
    noisy = tf.where(noise_mask & salt_mask, 1.0, images)
    noisy = tf.where(noise_mask & ~salt_mask, 0.0, noisy)
    
    return noisy


@tf.function
def tf_random_pixel_flip_noise(images: tf.Tensor, 
                              flip_prob: float = 0.05) -> tf.Tensor:
    """TensorFlow version of random pixel flip noise."""
    flip_mask = tf.random.uniform(tf.shape(images)) < flip_prob
    return tf.where(flip_mask, 1.0 - images, images)


def get_tf_noise_function(noise_type: str = 'salt_pepper') -> Callable:
    """
    Get TensorFlow-compatible noise function.
    
    Args:
        noise_type: Type of noise function to return
        
    Returns:
        TensorFlow noise function
    """
    tf_noise_functions = {
        'salt_pepper': tf_salt_and_pepper_noise,
        'pixel_flip': tf_random_pixel_flip_noise
    }
    
    if noise_type not in tf_noise_functions:
        print(f"TensorFlow version of {noise_type} not available, using salt_pepper")
        noise_type = 'salt_pepper'
    
    return tf_noise_functions[noise_type]
