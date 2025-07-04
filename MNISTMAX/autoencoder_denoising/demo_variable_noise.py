"""
Demo script for variable noise levels - shows noise progression from clean to maximum corruption.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variable_noise import VariableNoiseGenerator, create_noise_level_demo
from shared.data_utils import load_mnist_data
from bitmap_utils import convert_to_bitmap


def visualize_noise_progression(clean_images, noise_type='salt_pepper', num_levels=8, save_path=None):
    """
    Visualize noise progression from clean to maximum corruption.
    
    Args:
        clean_images: Clean MNIST images
        noise_type: Type of noise to demonstrate
        num_levels: Number of noise levels to show
        save_path: Path to save the visualization
    """
    # Create noise level demo
    demo_images, intensities = create_noise_level_demo(clean_images[:5], noise_type, num_levels)
    
    # Create visualization
    fig, axes = plt.subplots(5, num_levels, figsize=(2*num_levels, 10))
    fig.suptitle(f'Variable {noise_type.replace("_", " ").title()} Noise Progression', fontsize=16)
    
    for img_idx in range(5):
        for level_idx in range(num_levels):
            ax = axes[img_idx, level_idx]
            ax.imshow(demo_images[level_idx, img_idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Intensity: {intensities[level_idx]:.2f}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Noise progression saved to: {save_path}")
    
    plt.show()
    return fig


def compare_noise_types(clean_images, num_levels=6, save_path=None):
    """
    Compare different noise types at various intensity levels.
    
    Args:
        clean_images: Clean MNIST images
        num_levels: Number of noise levels to show
        save_path: Path to save the comparison
    """
    noise_types = ['salt_pepper', 'pixel_flip', 'gaussian', 'mixed']
    intensities = np.linspace(0.0, 1.0, num_levels)
    
    fig, axes = plt.subplots(len(noise_types), num_levels, figsize=(2*num_levels, 2*len(noise_types)))
    fig.suptitle('Comparison of Noise Types at Different Intensities', fontsize=16)
    
    generator = VariableNoiseGenerator()
    
    # Use the same image for all comparisons
    test_image = clean_images[0:1]  # Single image
    
    for noise_idx, noise_type in enumerate(noise_types):
        for level_idx, intensity in enumerate(intensities):
            if intensity == 0.0:
                noisy_image = test_image.copy()
            else:
                noisy_image = generator.add_variable_noise(test_image, noise_type, intensity)
            
            ax = axes[noise_idx, level_idx]
            ax.imshow(noisy_image[0], cmap='gray', vmin=0, vmax=1)
            
            if noise_idx == 0:  # Top row
                ax.set_title(f'{intensity:.2f}', fontsize=12)
            if level_idx == 0:  # Left column
                ax.set_ylabel(noise_type.replace('_', ' ').title(), fontsize=12)
            
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Noise type comparison saved to: {save_path}")
    
    plt.show()
    return fig


def create_noise_level_chart():
    """Create a chart showing the defined noise levels."""
    generator = VariableNoiseGenerator()
    
    # Get noise level definitions
    levels = generator.noise_levels
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(levels.keys())
    values = list(levels.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel('Noise Intensity', fontsize=12)
    ax.set_title('Predefined Noise Intensity Levels', fontsize=14)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return fig


def demonstrate_progressive_training():
    """Demonstrate how noise intensity changes during progressive training."""
    from variable_noise import ProgressiveNoiseTrainer
    
    generator = VariableNoiseGenerator()
    trainer = ProgressiveNoiseTrainer(generator, start_intensity=0.1, end_intensity=0.8)
    
    total_epochs = 20
    epochs = range(total_epochs)
    intensities = [trainer.get_current_intensity(epoch, total_epochs) for epoch in epochs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, intensities, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Noise Intensity', fontsize=12)
    ax.set_title('Progressive Noise Training Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add annotations for key points
    ax.annotate(f'Start: {intensities[0]:.2f}', 
                xy=(0, intensities[0]), xytext=(2, intensities[0] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate(f'End: {intensities[-1]:.2f}', 
                xy=(total_epochs-1, intensities[-1]), xytext=(total_epochs-3, intensities[-1] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.show()
    
    return fig


def evaluate_model_on_noise_levels(model_path=None):
    """
    Evaluate a trained model on different noise levels (if model exists).
    
    Args:
        model_path: Path to trained model
    """
    if model_path and os.path.exists(model_path):
        import tensorflow as tf
        from variable_noise import VariableNoiseGenerator
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load test data
        (_, _), (x_test, _) = load_mnist_data(normalize=True)
        x_test_binary = convert_to_bitmap(x_test[:100], threshold=0.5)
        
        # Evaluate on different noise levels
        generator = VariableNoiseGenerator()
        results = generator.evaluate_on_noise_levels(model, x_test_binary, 'salt_pepper', 10)
        
        # Plot results
        intensities = list(results.keys())
        accuracies = [results[i]['pixel_accuracy'] for i in intensities]
        dice_scores = [results[i]['dice_coefficient'] for i in intensities]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Pixel accuracy
        ax1.plot(intensities, accuracies, 'b-o', linewidth=2)
        ax1.set_xlabel('Noise Intensity')
        ax1.set_ylabel('Pixel Accuracy')
        ax1.set_title('Model Performance vs Noise Level')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Dice coefficient
        ax2.plot(intensities, dice_scores, 'r-o', linewidth=2)
        ax2.set_xlabel('Noise Intensity')
        ax2.set_ylabel('Dice Coefficient')
        ax2.set_title('Dice Score vs Noise Level')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return fig, results
    else:
        print("No trained model found. Train a model first to see performance evaluation.")
        return None, None


def main():
    """Main demo function."""
    print("🎯 Variable Noise Level Demo")
    print("=" * 50)
    
    # Load sample data
    print("Loading MNIST data...")
    (x_train, _), (x_test, _) = load_mnist_data(normalize=True)
    x_test_binary = convert_to_bitmap(x_test[:20], threshold=0.5)
    
    # Create sample_data directory if it doesn't exist
    os.makedirs('../sample_data', exist_ok=True)
    
    print("\n1. Visualizing noise progression...")
    fig1 = visualize_noise_progression(
        x_test_binary, 
        noise_type='salt_pepper', 
        num_levels=8,
        save_path='../sample_data/noise_progression_salt_pepper.png'
    )
    
    print("\n2. Comparing different noise types...")
    fig2 = compare_noise_types(
        x_test_binary, 
        num_levels=6,
        save_path='../sample_data/noise_types_comparison.png'
    )
    
    print("\n3. Showing predefined noise levels...")
    fig3 = create_noise_level_chart()
    
    print("\n4. Demonstrating progressive training schedule...")
    fig4 = demonstrate_progressive_training()
    
    print("\n5. Checking for trained model to evaluate...")
    # Look for the latest trained model
    log_dirs = [d for d in os.listdir('../autoencoder_denoising/logs') if d.startswith('efficient_')]
    if log_dirs:
        latest_log = sorted(log_dirs)[-1]
        model_path = f'../autoencoder_denoising/logs/{latest_log}/best_model.weights.h5'
        fig5, results = evaluate_model_on_noise_levels(model_path)
        if results:
            print(f"Model evaluation completed. Results saved for {len(results)} noise levels.")
    else:
        print("No trained models found. Run training first to see performance evaluation.")
    
    print("\n✅ Demo completed! Check the sample_data folder for saved visualizations.")


if __name__ == "__main__":
    main()
