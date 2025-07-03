"""
Bitmap utilities for binary image processing.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def convert_to_bitmap(images: np.ndarray, 
                     threshold: float = 0.5,
                     method: str = 'threshold') -> np.ndarray:
    """
    Convert grayscale images to binary bitmaps.
    
    Args:
        images: Input grayscale images (0-1 range)
        threshold: Threshold for binarization
        method: Binarization method ('threshold', 'otsu', 'adaptive')
        
    Returns:
        Binary images with values 0 or 1
    """
    if method == 'threshold':
        return (images > threshold).astype(np.float32)
    
    elif method == 'otsu':
        try:
            import cv2
            binary_images = []
            for img in images:
                # Convert to uint8
                img_uint8 = (img * 255).astype(np.uint8)
                # Apply Otsu's thresholding
                _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_images.append(binary.astype(np.float32) / 255.0)
            return np.array(binary_images)
        except ImportError:
            print("OpenCV not available, using simple threshold")
            return (images > threshold).astype(np.float32)
    
    elif method == 'adaptive':
        try:
            import cv2
            binary_images = []
            for img in images:
                # Convert to uint8
                img_uint8 = (img * 255).astype(np.uint8)
                # Apply adaptive thresholding
                binary = cv2.adaptiveThreshold(
                    img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                binary_images.append(binary.astype(np.float32) / 255.0)
            return np.array(binary_images)
        except ImportError:
            print("OpenCV not available, using simple threshold")
            return (images > threshold).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown binarization method: {method}")


def bitmap_to_grayscale(binary_images: np.ndarray, 
                       smooth: bool = False,
                       sigma: float = 0.5) -> np.ndarray:
    """
    Convert binary images back to grayscale (for visualization).
    
    Args:
        binary_images: Binary images (0 or 1)
        smooth: Whether to apply Gaussian smoothing
        sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        Grayscale images
    """
    grayscale = binary_images.astype(np.float32)
    
    if smooth:
        try:
            import cv2
            smoothed = []
            for img in grayscale:
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(img, (5, 5), sigma)
                smoothed.append(blurred)
            return np.array(smoothed)
        except ImportError:
            print("OpenCV not available, returning unsmoothed images")
    
    return grayscale


def analyze_bitmap_properties(images: np.ndarray) -> dict:
    """
    Analyze properties of binary images.
    
    Args:
        images: Binary images
        
    Returns:
        Dictionary with analysis results
    """
    properties = {}
    
    # Basic statistics
    properties['n_images'] = len(images)
    properties['image_shape'] = images.shape[1:]
    properties['total_pixels'] = np.prod(images.shape[1:])
    
    # Pixel statistics
    white_pixels = np.sum(images == 1, axis=(1, 2))
    black_pixels = np.sum(images == 0, axis=(1, 2))
    
    properties['white_pixel_stats'] = {
        'mean': float(np.mean(white_pixels)),
        'std': float(np.std(white_pixels)),
        'min': int(np.min(white_pixels)),
        'max': int(np.max(white_pixels))
    }
    
    properties['black_pixel_stats'] = {
        'mean': float(np.mean(black_pixels)),
        'std': float(np.std(black_pixels)),
        'min': int(np.min(black_pixels)),
        'max': int(np.max(black_pixels))
    }
    
    # Density (ratio of white pixels)
    density = white_pixels / properties['total_pixels']
    properties['density_stats'] = {
        'mean': float(np.mean(density)),
        'std': float(np.std(density)),
        'min': float(np.min(density)),
        'max': float(np.max(density))
    }
    
    return properties


def compute_bitmap_metrics(clean: np.ndarray, 
                          denoised: np.ndarray) -> dict:
    """
    Compute metrics for bitmap denoising evaluation.
    
    Args:
        clean: Clean binary images
        denoised: Denoised binary images
        
    Returns:
        Dictionary with metrics
    """
    # Ensure binary
    clean_binary = (clean > 0.5).astype(np.float32)
    denoised_binary = (denoised > 0.5).astype(np.float32)
    
    # Pixel accuracy
    pixel_accuracy = np.mean(clean_binary == denoised_binary)
    
    # Per-image accuracy
    per_image_accuracy = np.mean(clean_binary == denoised_binary, axis=(1, 2))
    
    # Hamming distance (number of differing pixels)
    hamming_distance = np.sum(clean_binary != denoised_binary, axis=(1, 2))
    
    # Jaccard index (IoU for binary images)
    intersection = np.sum((clean_binary == 1) & (denoised_binary == 1), axis=(1, 2))
    union = np.sum((clean_binary == 1) | (denoised_binary == 1), axis=(1, 2))
    jaccard_index = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Dice coefficient
    dice_coeff = 2 * intersection / (np.sum(clean_binary, axis=(1, 2)) + np.sum(denoised_binary, axis=(1, 2)) + 1e-8)
    
    # Precision and Recall for white pixels
    true_positives = np.sum((clean_binary == 1) & (denoised_binary == 1), axis=(1, 2))
    false_positives = np.sum((clean_binary == 0) & (denoised_binary == 1), axis=(1, 2))
    false_negatives = np.sum((clean_binary == 1) & (denoised_binary == 0), axis=(1, 2))
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'pixel_accuracy': float(pixel_accuracy),
        'per_image_accuracy': {
            'mean': float(np.mean(per_image_accuracy)),
            'std': float(np.std(per_image_accuracy)),
            'min': float(np.min(per_image_accuracy)),
            'max': float(np.max(per_image_accuracy))
        },
        'hamming_distance': {
            'mean': float(np.mean(hamming_distance)),
            'std': float(np.std(hamming_distance)),
            'min': int(np.min(hamming_distance)),
            'max': int(np.max(hamming_distance))
        },
        'jaccard_index': {
            'mean': float(np.mean(jaccard_index)),
            'std': float(np.std(jaccard_index)),
            'min': float(np.min(jaccard_index)),
            'max': float(np.max(jaccard_index))
        },
        'dice_coefficient': {
            'mean': float(np.mean(dice_coeff)),
            'std': float(np.std(dice_coeff)),
            'min': float(np.min(dice_coeff)),
            'max': float(np.max(dice_coeff))
        },
        'precision': {
            'mean': float(np.mean(precision)),
            'std': float(np.std(precision))
        },
        'recall': {
            'mean': float(np.mean(recall)),
            'std': float(np.std(recall))
        },
        'f1_score': {
            'mean': float(np.mean(f1_score)),
            'std': float(np.std(f1_score))
        }
    }


def visualize_bitmap_comparison(clean: np.ndarray,
                               noisy: np.ndarray,
                               denoised: np.ndarray,
                               n_samples: int = 5,
                               figsize: Tuple[int, int] = (15, 9),
                               save_path: Optional[str] = None):
    """
    Visualize comparison of clean, noisy, and denoised binary images.
    
    Args:
        clean: Clean binary images
        noisy: Noisy binary images
        denoised: Denoised binary images
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(4, n_samples, figsize=figsize)
    
    row_titles = ['Clean', 'Noisy', 'Denoised', 'Error']
    
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
        
        # Error map (XOR of clean and denoised)
        error = np.abs(clean[i] - denoised[i])
        axes[3, i].imshow(error, cmap='Reds', vmin=0, vmax=1)
        axes[3, i].axis('off')
    
    # Add row labels
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, rotation=90, size='large')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_bitmap_dataset(images: np.ndarray,
                         labels: np.ndarray,
                         threshold: float = 0.5,
                         method: str = 'threshold') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a bitmap dataset from grayscale MNIST images.
    
    Args:
        images: Grayscale MNIST images
        labels: Corresponding labels
        threshold: Binarization threshold
        method: Binarization method
        
    Returns:
        Binary images and labels
    """
    # Convert to bitmap
    binary_images = convert_to_bitmap(images, threshold, method)
    
    return binary_images, labels


def save_bitmap_samples(images: np.ndarray,
                       filepath: str,
                       n_samples: int = 100):
    """
    Save bitmap samples for inspection.
    
    Args:
        images: Binary images
        filepath: Path to save samples
        n_samples: Number of samples to save
    """
    # Select random samples
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    samples = images[indices]
    
    # Create grid visualization
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples):
        axes[i].imshow(sample, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bitmap samples saved to {filepath}")


def bitmap_morphological_operations(images: np.ndarray,
                                   operation: str = 'opening',
                                   kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological operations to binary images.
    
    Args:
        images: Binary images
        operation: Type of operation ('opening', 'closing', 'erosion', 'dilation')
        kernel_size: Size of morphological kernel
        
    Returns:
        Processed binary images
    """
    try:
        import cv2
        
        processed = []
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for img in images:
            # Convert to uint8
            img_uint8 = (img * 255).astype(np.uint8)
            
            if operation == 'erosion':
                result = cv2.erode(img_uint8, kernel, iterations=1)
            elif operation == 'dilation':
                result = cv2.dilate(img_uint8, kernel, iterations=1)
            elif operation == 'opening':
                result = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                result = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            processed.append(result.astype(np.float32) / 255.0)
        
        return np.array(processed)
        
    except ImportError:
        print("OpenCV not available, returning original images")
        return images


# TensorFlow utilities for bitmap processing
@tf.function
def tf_convert_to_bitmap(images: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """TensorFlow version of bitmap conversion."""
    return tf.cast(images > threshold, tf.float32)


@tf.function
def tf_bitmap_metrics(clean: tf.Tensor, denoised: tf.Tensor) -> tf.Tensor:
    """TensorFlow version of bitmap accuracy computation."""
    clean_binary = tf.cast(clean > 0.5, tf.float32)
    denoised_binary = tf.cast(denoised > 0.5, tf.float32)
    
    # Pixel accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(clean_binary, denoised_binary), tf.float32))
    
    return accuracy


class BitmapProcessor:
    """Utility class for bitmap processing operations."""
    
    def __init__(self, threshold: float = 0.5, method: str = 'threshold'):
        """
        Initialize bitmap processor.
        
        Args:
            threshold: Default binarization threshold
            method: Default binarization method
        """
        self.threshold = threshold
        self.method = method
    
    def process(self, images: np.ndarray) -> np.ndarray:
        """Process images to bitmaps."""
        return convert_to_bitmap(images, self.threshold, self.method)
    
    def analyze(self, images: np.ndarray) -> dict:
        """Analyze bitmap properties."""
        return analyze_bitmap_properties(images)
    
    def evaluate(self, clean: np.ndarray, denoised: np.ndarray) -> dict:
        """Evaluate denoising performance."""
        return compute_bitmap_metrics(clean, denoised)
    
    def visualize(self, clean: np.ndarray, noisy: np.ndarray, 
                 denoised: np.ndarray, **kwargs):
        """Visualize bitmap comparison."""
        visualize_bitmap_comparison(clean, noisy, denoised, **kwargs)
