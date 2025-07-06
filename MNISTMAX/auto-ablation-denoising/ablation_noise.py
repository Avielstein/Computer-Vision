"""
Additive-only noise generation for ablation denoising.
Only adds pixels (0 → 1), never removes them.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Union, Dict, Any


class AdditiveNoiseGenerator:
    """
    Generates additive-only noise for ablation denoising experiments.
    
    Core principle: Only add pixels (0 → 1), never remove pixels (1 → 0).
    This ensures that noisy_pixels ⊇ clean_pixels always holds.
    """
    
    def __init__(self, noise_params: Dict[str, Any] = None):
        """
        Initialize the additive noise generator.
        
        Args:
            noise_params: Dictionary of noise parameters for different noise types
        """
        self.noise_params = noise_params or {
            'random_pixels': {'add_prob': 0.1},
            'structured_lines': {'num_lines': 2, 'thickness': 1},
            'blocks': {'num_blocks': 3, 'block_size_range': (2, 5)},
            'gaussian_blobs': {'num_blobs': 5, 'blob_size_range': (1, 3)},
            'border_noise': {'border_width': 2, 'add_prob': 0.3}
        }
    
    def add_random_pixels(self, 
                         images: np.ndarray, 
                         add_prob: float = 0.1) -> np.ndarray:
        """
        Add random pixels to images (only where pixels are currently 0).
        
        Args:
            images: Input binary images of shape (N, H, W) or (N, H, W, 1)
            add_prob: Probability of adding a pixel at each 0-valued location
            
        Returns:
            Images with added random pixels
        """
        images = self._ensure_3d(images)
        noisy_images = images.copy()
        
        # Only add pixels where current value is 0
        zero_mask = (images == 0)
        add_mask = np.random.random(images.shape) < add_prob
        
        # Combine masks: only add where pixel is 0 AND random condition is met
        final_add_mask = zero_mask & add_mask
        
        noisy_images[final_add_mask] = 1.0
        
        return noisy_images
    
    def add_structured_lines(self, 
                           images: np.ndarray,
                           num_lines: int = 2,
                           thickness: int = 1) -> np.ndarray:
        """
        Add structured line noise (horizontal, vertical, or diagonal).
        
        Args:
            images: Input binary images
            num_lines: Number of lines to add per image
            thickness: Thickness of lines
            
        Returns:
            Images with added line noise
        """
        images = self._ensure_3d(images)
        noisy_images = images.copy()
        
        for i in range(len(images)):
            img = noisy_images[i]
            h, w = img.shape
            
            for _ in range(num_lines):
                # Random line type: 0=horizontal, 1=vertical, 2=diagonal
                line_type = np.random.randint(0, 3)
                
                if line_type == 0:  # Horizontal line
                    y = np.random.randint(0, h)
                    for t in range(thickness):
                        if y + t < h:
                            img[y + t, :] = np.maximum(img[y + t, :], 1.0)
                
                elif line_type == 1:  # Vertical line
                    x = np.random.randint(0, w)
                    for t in range(thickness):
                        if x + t < w:
                            img[:, x + t] = np.maximum(img[:, x + t], 1.0)
                
                else:  # Diagonal line
                    start_x = np.random.randint(0, w)
                    start_y = np.random.randint(0, h)
                    direction = np.random.choice([-1, 1])  # Diagonal direction
                    
                    for step in range(min(h, w)):
                        x = start_x + step
                        y = start_y + direction * step
                        
                        if 0 <= x < w and 0 <= y < h:
                            for t in range(thickness):
                                if x + t < w and y < h:
                                    img[y, x + t] = 1.0
                                if x < w and y + t < h:
                                    img[y + t, x] = 1.0
        
        return noisy_images
    
    def add_block_noise(self, 
                       images: np.ndarray,
                       num_blocks: int = 3,
                       block_size_range: Tuple[int, int] = (2, 5)) -> np.ndarray:
        """
        Add rectangular block noise.
        
        Args:
            images: Input binary images
            num_blocks: Number of blocks to add per image
            block_size_range: Range of block sizes (min, max)
            
        Returns:
            Images with added block noise
        """
        images = self._ensure_3d(images)
        noisy_images = images.copy()
        
        for i in range(len(images)):
            img = noisy_images[i]
            h, w = img.shape
            
            for _ in range(num_blocks):
                # Random block size
                block_h = np.random.randint(block_size_range[0], block_size_range[1] + 1)
                block_w = np.random.randint(block_size_range[0], block_size_range[1] + 1)
                
                # Random position
                start_y = np.random.randint(0, max(1, h - block_h))
                start_x = np.random.randint(0, max(1, w - block_w))
                
                # Add block (only where pixels are currently 0)
                block_region = img[start_y:start_y + block_h, start_x:start_x + block_w]
                img[start_y:start_y + block_h, start_x:start_x + block_w] = np.maximum(
                    block_region, 1.0
                )
        
        return noisy_images
    
    def add_gaussian_blobs(self, 
                          images: np.ndarray,
                          num_blobs: int = 5,
                          blob_size_range: Tuple[int, int] = (1, 3)) -> np.ndarray:
        """
        Add circular blob noise.
        
        Args:
            images: Input binary images
            num_blobs: Number of blobs to add per image
            blob_size_range: Range of blob radii
            
        Returns:
            Images with added blob noise
        """
        images = self._ensure_3d(images)
        noisy_images = images.copy()
        
        for i in range(len(images)):
            img = noisy_images[i]
            h, w = img.shape
            
            for _ in range(num_blobs):
                # Random blob center and radius
                center_y = np.random.randint(0, h)
                center_x = np.random.randint(0, w)
                radius = np.random.randint(blob_size_range[0], blob_size_range[1] + 1)
                
                # Create circular mask
                y_coords, x_coords = np.ogrid[:h, :w]
                mask = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 <= radius ** 2
                
                # Add blob (only where pixels are currently 0)
                img[mask] = np.maximum(img[mask], 1.0)
        
        return noisy_images
    
    def add_border_noise(self, 
                        images: np.ndarray,
                        border_width: int = 2,
                        add_prob: float = 0.3) -> np.ndarray:
        """
        Add noise around image borders.
        
        Args:
            images: Input binary images
            border_width: Width of border region
            add_prob: Probability of adding pixels in border region
            
        Returns:
            Images with added border noise
        """
        images = self._ensure_3d(images)
        noisy_images = images.copy()
        
        for i in range(len(images)):
            img = noisy_images[i]
            h, w = img.shape
            
            # Create border mask
            border_mask = np.zeros((h, w), dtype=bool)
            border_mask[:border_width, :] = True  # Top
            border_mask[-border_width:, :] = True  # Bottom
            border_mask[:, :border_width] = True  # Left
            border_mask[:, -border_width:] = True  # Right
            
            # Add noise in border region
            zero_mask = (img == 0)
            add_mask = np.random.random((h, w)) < add_prob
            
            final_add_mask = border_mask & zero_mask & add_mask
            img[final_add_mask] = 1.0
        
        return noisy_images
    
    def add_mixed_noise(self, 
                       images: np.ndarray,
                       noise_types: list = None,
                       intensity: float = 0.5) -> np.ndarray:
        """
        Add mixed additive noise using multiple noise types.
        
        Args:
            images: Input binary images
            noise_types: List of noise types to apply
            intensity: Overall noise intensity (0.0 to 1.0)
            
        Returns:
            Images with mixed additive noise
        """
        if noise_types is None:
            noise_types = ['random_pixels', 'structured_lines', 'blocks']
        
        images = self._ensure_3d(images)
        noisy_images = images.copy()
        
        # Scale noise parameters by intensity
        scaled_params = self._scale_noise_params(intensity)
        
        # Apply each noise type with some probability
        for noise_type in noise_types:
            if np.random.random() < 0.7:  # 70% chance to apply each noise type
                if noise_type == 'random_pixels':
                    noisy_images = self.add_random_pixels(
                        noisy_images, **scaled_params['random_pixels']
                    )
                elif noise_type == 'structured_lines':
                    noisy_images = self.add_structured_lines(
                        noisy_images, **scaled_params['structured_lines']
                    )
                elif noise_type == 'blocks':
                    noisy_images = self.add_block_noise(
                        noisy_images, **scaled_params['blocks']
                    )
                elif noise_type == 'gaussian_blobs':
                    noisy_images = self.add_gaussian_blobs(
                        noisy_images, **scaled_params['gaussian_blobs']
                    )
                elif noise_type == 'border_noise':
                    noisy_images = self.add_border_noise(
                        noisy_images, **scaled_params['border_noise']
                    )
        
        return noisy_images
    
    def _ensure_3d(self, images: np.ndarray) -> np.ndarray:
        """Ensure images are 3D (N, H, W)."""
        if len(images.shape) == 4 and images.shape[-1] == 1:
            return images.squeeze(-1)
        elif len(images.shape) == 2:
            return images[np.newaxis, ...]
        return images
    
    def _scale_noise_params(self, intensity: float) -> Dict[str, Any]:
        """Scale noise parameters by intensity factor."""
        scaled = {}
        
        scaled['random_pixels'] = {
            'add_prob': self.noise_params['random_pixels']['add_prob'] * intensity
        }
        
        scaled['structured_lines'] = {
            'num_lines': max(1, int(self.noise_params['structured_lines']['num_lines'] * intensity)),
            'thickness': max(1, int(self.noise_params['structured_lines']['thickness'] * intensity))
        }
        
        scaled['blocks'] = {
            'num_blocks': max(1, int(self.noise_params['blocks']['num_blocks'] * intensity)),
            'block_size_range': self.noise_params['blocks']['block_size_range']
        }
        
        scaled['gaussian_blobs'] = {
            'num_blobs': max(1, int(self.noise_params['gaussian_blobs']['num_blobs'] * intensity)),
            'blob_size_range': self.noise_params['gaussian_blobs']['blob_size_range']
        }
        
        scaled['border_noise'] = {
            'border_width': max(1, int(self.noise_params['border_noise']['border_width'] * intensity)),
            'add_prob': self.noise_params['border_noise']['add_prob'] * intensity
        }
        
        return scaled
    
    def generate_noise_progression(self, 
                                 clean_image: np.ndarray,
                                 noise_levels: list = None) -> Dict[str, np.ndarray]:
        """
        Generate a progression of noise levels for visualization.
        
        Args:
            clean_image: Single clean binary image
            noise_levels: List of noise intensities
            
        Returns:
            Dictionary mapping noise levels to noisy images
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        if len(clean_image.shape) == 2:
            clean_image = clean_image[np.newaxis, ...]
        
        progression = {}
        
        for level in noise_levels:
            if level == 0.0:
                progression[level] = clean_image[0]
            else:
                noisy = self.add_mixed_noise(clean_image, intensity=level)
                progression[level] = noisy[0]
        
        return progression


# TensorFlow functions for efficient training
@tf.function
def add_random_pixels_tf(images: tf.Tensor, add_prob: float = 0.1) -> tf.Tensor:
    """TensorFlow function to add random pixels."""
    zero_mask = tf.equal(images, 0.0)
    add_mask = tf.random.uniform(tf.shape(images)) < add_prob
    final_add_mask = tf.logical_and(zero_mask, add_mask)
    
    return tf.where(final_add_mask, 1.0, images)


@tf.function
def add_block_noise_tf(images: tf.Tensor, 
                      num_blocks: int = 3,
                      max_block_size: int = 5) -> tf.Tensor:
    """TensorFlow function to add block noise."""
    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    
    noisy_images = images
    
    for _ in range(num_blocks):
        # Random block size
        block_h = tf.random.uniform([], 2, max_block_size + 1, dtype=tf.int32)
        block_w = tf.random.uniform([], 2, max_block_size + 1, dtype=tf.int32)
        
        # Random position
        start_y = tf.random.uniform([], 0, height - block_h + 1, dtype=tf.int32)
        start_x = tf.random.uniform([], 0, width - block_w + 1, dtype=tf.int32)
        
        # Create block mask
        y_coords = tf.range(height)[:, None]
        x_coords = tf.range(width)[None, :]
        
        block_mask = tf.logical_and(
            tf.logical_and(y_coords >= start_y, y_coords < start_y + block_h),
            tf.logical_and(x_coords >= start_x, x_coords < start_x + block_w)
        )
        
        # Apply to all images in batch
        block_mask = tf.expand_dims(block_mask, 0)
        block_mask = tf.tile(block_mask, [batch_size, 1, 1])
        
        # Add block (only where pixels are currently 0)
        noisy_images = tf.where(block_mask, tf.maximum(noisy_images, 1.0), noisy_images)
    
    return noisy_images


def create_additive_noise_dataset(clean_images: np.ndarray,
                                batch_size: int = 32,
                                noise_intensity: float = 0.5,
                                noise_type: str = 'mixed') -> tf.data.Dataset:
    """
    Create a TensorFlow dataset with additive noise generation.
    
    Args:
        clean_images: Clean binary images
        batch_size: Batch size
        noise_intensity: Noise intensity level
        noise_type: Type of noise to apply
        
    Returns:
        Dataset yielding (noisy, clean) pairs
    """
    noise_gen = AdditiveNoiseGenerator()
    
    def add_noise_fn(clean_batch):
        """Add noise to a batch of clean images."""
        if noise_type == 'random':
            noisy_batch = add_random_pixels_tf(clean_batch, add_prob=0.1 * noise_intensity)
        elif noise_type == 'blocks':
            noisy_batch = add_block_noise_tf(clean_batch, num_blocks=int(3 * noise_intensity))
        else:  # mixed
            # Apply multiple noise types
            noisy_batch = clean_batch
            noisy_batch = add_random_pixels_tf(noisy_batch, add_prob=0.05 * noise_intensity)
            noisy_batch = add_block_noise_tf(noisy_batch, num_blocks=int(2 * noise_intensity))
        
        # Add channel dimension
        noisy_batch = tf.expand_dims(noisy_batch, -1)
        clean_batch = tf.expand_dims(clean_batch, -1)
        
        return noisy_batch, clean_batch
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(clean_images)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(add_noise_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Demo usage
    from shared.data_utils import load_mnist_data
    from shared.visualization import visualize_bitmap_comparison
    
    # Load sample data
    (x_train, _), (_, _) = load_mnist_data(normalize=True)
    from bitmap_utils import convert_to_bitmap
    x_binary = convert_to_bitmap(x_train[:5], threshold=0.5)
    
    # Create noise generator
    noise_gen = AdditiveNoiseGenerator()
    
    # Generate different types of noise
    random_noise = noise_gen.add_random_pixels(x_binary, add_prob=0.15)
    line_noise = noise_gen.add_structured_lines(x_binary, num_lines=3)
    block_noise = noise_gen.add_block_noise(x_binary, num_blocks=4)
    mixed_noise = noise_gen.add_mixed_noise(x_binary, intensity=0.7)
    
    # Visualize results
    print("Additive Noise Generation Demo")
    print("Original -> Random -> Lines -> Blocks -> Mixed")
    
    for i in range(5):
        print(f"\nImage {i + 1}:")
        print(f"  Original pixels: {x_binary[i].sum()}")
        print(f"  Random noise: {random_noise[i].sum()}")
        print(f"  Line noise: {line_noise[i].sum()}")
        print(f"  Block noise: {block_noise[i].sum()}")
        print(f"  Mixed noise: {mixed_noise[i].sum()}")
