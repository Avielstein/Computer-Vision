"""
Variable noise level system for progressive denoising training and evaluation.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List, Optional, Callable
from noise_generation import NoiseGenerator, salt_and_pepper_noise, random_pixel_flip_noise, gaussian_noise_binary


class VariableNoiseGenerator:
    """
    Generate noise with variable intensity levels from 0 (clean) to maximum corruption.
    """
    
    def __init__(self, max_noise_params: Dict[str, Dict] = None):
        """
        Initialize variable noise generator.
        
        Args:
            max_noise_params: Maximum noise parameters for each noise type
        """
        if max_noise_params is None:
            self.max_noise_params = {
                'salt_pepper': {'noise_prob': 0.3, 'salt_prob': 0.5},
                'pixel_flip': {'flip_prob': 0.15},
                'gaussian': {'noise_std': 0.25, 'threshold': 0.5},
                'mixed': {'intensity': 1.0}  # For mixed noise scaling
            }
        else:
            self.max_noise_params = max_noise_params
        
        # Define noise level ranges (0.0 = clean, 1.0 = maximum noise)
        self.noise_levels = {
            'clean': 0.0,
            'very_light': 0.1,
            'light': 0.25,
            'moderate': 0.5,
            'heavy': 0.75,
            'very_heavy': 0.9,
            'maximum': 1.0
        }
    
    def get_noise_params(self, noise_type: str, intensity: float) -> Dict:
        """
        Get noise parameters for given intensity level.
        
        Args:
            noise_type: Type of noise ('salt_pepper', 'pixel_flip', 'gaussian', 'mixed')
            intensity: Noise intensity from 0.0 (clean) to 1.0 (maximum)
            
        Returns:
            Dictionary of noise parameters
        """
        intensity = np.clip(intensity, 0.0, 1.0)
        
        if intensity == 0.0:
            # Return parameters that produce no noise
            return {key: 0.0 for key in self.max_noise_params[noise_type].keys()}
        
        max_params = self.max_noise_params[noise_type]
        scaled_params = {}
        
        for param, max_val in max_params.items():
            if param == 'salt_prob' or param == 'threshold':
                # These parameters don't scale linearly
                scaled_params[param] = max_val
            else:
                # Scale parameter by intensity
                scaled_params[param] = max_val * intensity
        
        return scaled_params
    
    def add_variable_noise(self, images: np.ndarray, 
                          noise_type: str = 'salt_pepper',
                          intensity: float = 0.5) -> np.ndarray:
        """
        Add noise with variable intensity.
        
        Args:
            images: Input images
            noise_type: Type of noise to add
            intensity: Noise intensity (0.0 = clean, 1.0 = maximum)
            
        Returns:
            Noisy images
        """
        if intensity <= 0.0:
            return images.copy()
        
        params = self.get_noise_params(noise_type, intensity)
        
        if noise_type == 'salt_pepper':
            return salt_and_pepper_noise(images, **params)
        elif noise_type == 'pixel_flip':
            return random_pixel_flip_noise(images, **params)
        elif noise_type == 'gaussian':
            return gaussian_noise_binary(images, **params)
        elif noise_type == 'mixed':
            return self._add_mixed_variable_noise(images, intensity)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def _add_mixed_variable_noise(self, images: np.ndarray, intensity: float) -> np.ndarray:
        """Add mixed noise with variable intensity."""
        noisy = images.copy()
        
        # Apply multiple noise types with scaled intensity
        noise_types = ['salt_pepper', 'pixel_flip', 'gaussian']
        
        for noise_type in noise_types:
            # Each noise type gets a fraction of the total intensity
            type_intensity = intensity * np.random.uniform(0.3, 0.7)
            if type_intensity > 0.05:  # Only apply if significant
                noisy = self.add_variable_noise(noisy, noise_type, type_intensity)
        
        return noisy
    
    def create_noise_schedule(self, num_levels: int = 10, 
                             noise_type: str = 'salt_pepper') -> List[Tuple[float, Dict]]:
        """
        Create a schedule of noise levels for progressive training/evaluation.
        
        Args:
            num_levels: Number of noise levels to create
            noise_type: Type of noise
            
        Returns:
            List of (intensity, params) tuples
        """
        intensities = np.linspace(0.0, 1.0, num_levels)
        schedule = []
        
        for intensity in intensities:
            params = self.get_noise_params(noise_type, intensity)
            schedule.append((intensity, params))
        
        return schedule
    
    def evaluate_on_noise_levels(self, model, test_images: np.ndarray, 
                                noise_type: str = 'salt_pepper',
                                num_levels: int = 10) -> Dict[float, Dict]:
        """
        Evaluate model performance across different noise levels.
        
        Args:
            model: Trained denoising model
            test_images: Clean test images
            noise_type: Type of noise to evaluate
            num_levels: Number of noise levels to test
            
        Returns:
            Dictionary mapping noise intensity to performance metrics
        """
        from bitmap_utils import compute_bitmap_metrics
        
        results = {}
        schedule = self.create_noise_schedule(num_levels, noise_type)
        
        for intensity, params in schedule:
            # Generate noisy images
            if intensity == 0.0:
                noisy_images = test_images.copy()
            else:
                noisy_images = self.add_variable_noise(test_images, noise_type, intensity)
            
            # Get model predictions
            if len(noisy_images.shape) == 3:
                noisy_input = np.expand_dims(noisy_images, -1)
            else:
                noisy_input = noisy_images
            
            denoised = model.predict(noisy_input, verbose=0)
            if len(denoised.shape) == 4:
                denoised = denoised.squeeze()
            
            # Compute metrics
            metrics = compute_bitmap_metrics(test_images, denoised)
            
            results[intensity] = {
                'pixel_accuracy': metrics['pixel_accuracy'],
                'dice_coefficient': metrics['dice_coefficient']['mean'],
                'jaccard_index': metrics['jaccard_index']['mean'],
                'noise_params': params
            }
        
        return results


class ProgressiveNoiseTrainer:
    """
    Trainer that progressively increases noise difficulty during training.
    """
    
    def __init__(self, noise_generator: VariableNoiseGenerator,
                 start_intensity: float = 0.1,
                 end_intensity: float = 0.8,
                 noise_type: str = 'salt_pepper'):
        """
        Initialize progressive noise trainer.
        
        Args:
            noise_generator: Variable noise generator
            start_intensity: Starting noise intensity
            end_intensity: Final noise intensity
            noise_type: Type of noise to use
        """
        self.noise_generator = noise_generator
        self.start_intensity = start_intensity
        self.end_intensity = end_intensity
        self.noise_type = noise_type
    
    def get_current_intensity(self, epoch: int, total_epochs: int) -> float:
        """
        Get current noise intensity based on training progress.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Current noise intensity
        """
        if total_epochs <= 1:
            return self.end_intensity
        
        progress = epoch / (total_epochs - 1)
        intensity = self.start_intensity + progress * (self.end_intensity - self.start_intensity)
        return np.clip(intensity, 0.0, 1.0)
    
    def create_progressive_dataset(self, clean_images: np.ndarray,
                                 epoch: int, total_epochs: int,
                                 batch_size: int = 32) -> tf.data.Dataset:
        """
        Create dataset with progressive noise for current epoch.
        
        Args:
            clean_images: Clean training images
            epoch: Current epoch
            total_epochs: Total epochs
            batch_size: Batch size
            
        Returns:
            TensorFlow dataset with appropriate noise level
        """
        current_intensity = self.get_current_intensity(epoch, total_epochs)
        
        # Generate noisy images for this epoch
        noisy_images = self.noise_generator.add_variable_noise(
            clean_images, self.noise_type, current_intensity
        )
        
        # Create dataset
        if len(clean_images.shape) == 3:
            clean_images = np.expand_dims(clean_images, -1)
            noisy_images = np.expand_dims(noisy_images, -1)
        
        dataset = tf.data.Dataset.from_tensor_slices((noisy_images, clean_images))
        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


@tf.function
def tf_variable_salt_pepper_noise(images: tf.Tensor, intensity: tf.Tensor) -> tf.Tensor:
    """
    TensorFlow function for variable salt and pepper noise.
    
    Args:
        images: Input images
        intensity: Noise intensity (0.0 to 1.0)
        
    Returns:
        Noisy images
    """
    # Scale noise probability by intensity
    max_noise_prob = 0.3
    noise_prob = intensity * max_noise_prob
    salt_prob = 0.5
    
    # Generate noise mask
    noise_mask = tf.random.uniform(tf.shape(images)) < noise_prob
    salt_mask = tf.random.uniform(tf.shape(images)) < salt_prob
    
    # Apply noise
    noisy = tf.where(noise_mask & salt_mask, 1.0, images)
    noisy = tf.where(noise_mask & ~salt_mask, 0.0, noisy)
    
    return noisy


@tf.function
def tf_variable_pixel_flip_noise(images: tf.Tensor, intensity: tf.Tensor) -> tf.Tensor:
    """
    TensorFlow function for variable pixel flip noise.
    
    Args:
        images: Input images
        intensity: Noise intensity (0.0 to 1.0)
        
    Returns:
        Noisy images
    """
    max_flip_prob = 0.15
    flip_prob = intensity * max_flip_prob
    
    flip_mask = tf.random.uniform(tf.shape(images)) < flip_prob
    return tf.where(flip_mask, 1.0 - images, images)


def create_variable_noise_dataset(clean_images: np.ndarray,
                                 noise_type: str = 'salt_pepper',
                                 intensity_range: Tuple[float, float] = (0.0, 1.0),
                                 batch_size: int = 32) -> tf.data.Dataset:
    """
    Create dataset with variable noise levels.
    
    Args:
        clean_images: Clean images
        noise_type: Type of noise
        intensity_range: Range of noise intensities (min, max)
        batch_size: Batch size
        
    Returns:
        TensorFlow dataset with variable noise
    """
    def apply_variable_noise(clean_image):
        # Random intensity for each sample
        intensity = tf.random.uniform([], intensity_range[0], intensity_range[1])
        
        if noise_type == 'salt_pepper':
            noisy_image = tf_variable_salt_pepper_noise(clean_image, intensity)
        elif noise_type == 'pixel_flip':
            noisy_image = tf_variable_pixel_flip_noise(clean_image, intensity)
        else:
            # Default to salt and pepper
            noisy_image = tf_variable_salt_pepper_noise(clean_image, intensity)
        
        # Add channel dimension if needed
        if len(clean_image.shape) == 2:
            clean_image = tf.expand_dims(clean_image, -1)
            noisy_image = tf.expand_dims(noisy_image, -1)
        
        return noisy_image, clean_image
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(clean_images)
    dataset = dataset.map(apply_variable_noise, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_noise_level_demo(clean_images: np.ndarray,
                           noise_type: str = 'salt_pepper',
                           num_levels: int = 6) -> Tuple[np.ndarray, List[float]]:
    """
    Create demonstration of different noise levels.
    
    Args:
        clean_images: Clean images to add noise to
        noise_type: Type of noise
        num_levels: Number of noise levels to demonstrate
        
    Returns:
        Tuple of (noisy_images_array, intensity_levels)
    """
    generator = VariableNoiseGenerator()
    intensities = np.linspace(0.0, 1.0, num_levels)
    
    demo_images = []
    
    for intensity in intensities:
        noisy = generator.add_variable_noise(clean_images, noise_type, intensity)
        demo_images.append(noisy)
    
    return np.array(demo_images), intensities.tolist()


if __name__ == "__main__":
    # Example usage
    from shared.data_utils import load_mnist_data
    from bitmap_utils import convert_to_bitmap
    
    # Load sample data
    (x_train, _), (x_test, _) = load_mnist_data(normalize=True)
    x_test_binary = convert_to_bitmap(x_test[:100], threshold=0.5)
    
    # Create variable noise generator
    noise_gen = VariableNoiseGenerator()
    
    # Demonstrate different noise levels
    demo_images, intensities = create_noise_level_demo(x_test_binary[:5], 'salt_pepper', 6)
    
    print("Noise intensity levels:", intensities)
    print("Demo images shape:", demo_images.shape)
    
    # Create noise schedule
    schedule = noise_gen.create_noise_schedule(10, 'salt_pepper')
    print("\nNoise schedule:")
    for i, (intensity, params) in enumerate(schedule):
        print(f"Level {i}: intensity={intensity:.2f}, params={params}")
