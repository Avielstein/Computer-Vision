"""
Create comprehensive animation showing variable noise levels and denoising results.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variable_noise import VariableNoiseGenerator, create_noise_level_demo
from shared.data_utils import load_mnist_data
from bitmap_utils import convert_to_bitmap


def create_comprehensive_noise_animation():
    """Create a comprehensive animation showing different noise levels and types."""
    
    print("🎬 Creating Comprehensive Noise Level Animation")
    print("=" * 60)
    
    # Load MNIST data
    print("Loading MNIST data...")
    (_, _), (x_test, _) = load_mnist_data(normalize=True)
    x_test_binary = convert_to_bitmap(x_test[:50], threshold=0.5)
    
    # Select diverse digits for demonstration
    digit_indices = [7, 23, 42, 15, 31, 8]  # Diverse set of digits
    demo_images = x_test_binary[digit_indices]
    
    # Create noise generator
    generator = VariableNoiseGenerator()
    
    # Define noise levels to demonstrate
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    noise_types = ['salt_pepper', 'pixel_flip', 'gaussian', 'mixed']
    
    # Create animation frames
    frames = []
    frame_dir = "../sample_data/animation_frames"
    os.makedirs(frame_dir, exist_ok=True)
    
    print(f"Creating {len(noise_levels)} frames for {len(noise_types)} noise types...")
    
    frame_count = 0
    
    # Create frames for each noise type
    for noise_type in noise_types:
        print(f"\nProcessing {noise_type} noise...")
        
        for level_idx, intensity in enumerate(noise_levels):
            frame_count += 1
            
            # Generate noisy images for this intensity
            if intensity == 0.0:
                noisy_images = demo_images.copy()
            else:
                noisy_images = generator.add_variable_noise(demo_images, noise_type, intensity)
            
            # Create visualization
            fig, axes = plt.subplots(2, 6, figsize=(18, 6))
            fig.suptitle(f'{noise_type.replace("_", " ").title()} Noise - Intensity: {intensity:.1f}', 
                        fontsize=20, fontweight='bold')
            
            # Top row: Clean images
            for i in range(6):
                axes[0, i].imshow(demo_images[i], cmap='gray', vmin=0, vmax=1)
                axes[0, i].set_title('Clean' if i == 0 else '', fontsize=14)
                axes[0, i].axis('off')
            
            # Bottom row: Noisy images
            for i in range(6):
                axes[1, i].imshow(noisy_images[i], cmap='gray', vmin=0, vmax=1)
                axes[1, i].set_title('Noisy' if i == 0 else '', fontsize=14)
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # Save frame
            frame_path = os.path.join(frame_dir, f"frame_{frame_count:03d}_{noise_type}_{intensity:.1f}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            
            frames.append(frame_path)
            
            if level_idx % 3 == 0:
                print(f"  Created frame {frame_count} (intensity {intensity:.1f})")
    
    print(f"\nCreated {len(frames)} frames total")
    
    # Create animated GIF
    print("Creating animated GIF...")
    
    # Load images
    images = []
    for frame_path in frames:
        img = Image.open(frame_path)
        images.append(img)
    
    # Create GIF with variable durations
    gif_path = "../sample_data/comprehensive_noise_animation.gif"
    
    # Set durations - slower for extreme values, faster for middle values
    durations = []
    frames_per_type = len(noise_levels)
    
    for i, frame_path in enumerate(frames):
        frame_in_sequence = i % frames_per_type
        intensity = noise_levels[frame_in_sequence]
        
        if intensity == 0.0 or intensity == 1.0:
            durations.append(1500)  # 1.5 seconds for clean and maximum noise
        elif intensity in [0.1, 0.9]:
            durations.append(1000)  # 1 second for near extremes
        else:
            durations.append(600)   # 0.6 seconds for middle values
    
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0
    )
    
    print(f"Comprehensive noise animation saved: {gif_path}")
    
    # Create a summary comparison image
    create_noise_level_summary(generator, demo_images)
    
    return gif_path


def create_noise_level_summary(generator, demo_images):
    """Create a summary image showing all noise levels side by side."""
    
    print("Creating noise level summary...")
    
    # Define key noise levels for summary
    key_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    noise_types = ['salt_pepper', 'pixel_flip', 'gaussian', 'mixed']
    
    # Use first 3 demo images for summary
    summary_images = demo_images[:3]
    
    # Create large summary figure
    fig, axes = plt.subplots(len(noise_types), len(key_levels), figsize=(18, 12))
    fig.suptitle('Variable Noise Levels Summary - Different Types and Intensities', 
                fontsize=20, fontweight='bold')
    
    for type_idx, noise_type in enumerate(noise_types):
        for level_idx, intensity in enumerate(key_levels):
            
            # Generate noisy version
            if intensity == 0.0:
                noisy_sample = summary_images[0].copy()
            else:
                noisy_sample = generator.add_variable_noise(
                    summary_images[0:1], noise_type, intensity
                )[0]
            
            # Plot
            ax = axes[type_idx, level_idx]
            ax.imshow(noisy_sample, cmap='gray', vmin=0, vmax=1)
            
            # Set titles
            if type_idx == 0:  # Top row
                ax.set_title(f'Intensity: {intensity:.1f}', fontsize=12, fontweight='bold')
            if level_idx == 0:  # Left column
                ax.set_ylabel(noise_type.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = "../sample_data/noise_levels_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Noise levels summary saved: {summary_path}")
    
    return summary_path


def create_progressive_noise_demo():
    """Create a demo showing progressive noise increase."""
    
    print("Creating progressive noise demonstration...")
    
    # Load sample data
    (_, _), (x_test, _) = load_mnist_data(normalize=True)
    x_test_binary = convert_to_bitmap(x_test[:10], threshold=0.5)
    
    # Select one clear digit
    demo_digit = x_test_binary[7:8]  # Single digit
    
    generator = VariableNoiseGenerator()
    
    # Create progression with many steps
    intensities = np.linspace(0.0, 1.0, 21)  # 21 steps from 0 to 1
    
    fig, axes = plt.subplots(3, 7, figsize=(21, 9))
    fig.suptitle('Progressive Noise Increase - Salt & Pepper Noise', fontsize=20, fontweight='bold')
    
    for i, intensity in enumerate(intensities):
        row = i // 7
        col = i % 7
        
        if row >= 3:  # Only use first 3 rows
            break
            
        # Generate noisy version
        if intensity == 0.0:
            noisy = demo_digit.copy()
        else:
            noisy = generator.add_variable_noise(demo_digit, 'salt_pepper', intensity)
        
        # Plot
        axes[row, col].imshow(noisy[0], cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'{intensity:.2f}', fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save progressive demo
    progressive_path = "../sample_data/progressive_noise_demo.png"
    plt.savefig(progressive_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Progressive noise demo saved: {progressive_path}")
    
    return progressive_path


def main():
    """Main function to create all noise demonstrations."""
    
    # Create sample_data directory
    os.makedirs("../sample_data", exist_ok=True)
    
    print("🎯 Creating Comprehensive Noise Level Demonstrations")
    print("=" * 70)
    
    # Create comprehensive animation
    gif_path = create_comprehensive_noise_animation()
    
    # Create progressive demo
    progressive_path = create_progressive_noise_demo()
    
    print("\n✅ All demonstrations created successfully!")
    print(f"📁 Files saved in: MNISTMAX/sample_data/")
    print(f"🎬 Main animation: comprehensive_noise_animation.gif")
    print(f"📊 Summary image: noise_levels_summary.png")
    print(f"📈 Progressive demo: progressive_noise_demo.png")
    print(f"🎨 Noise progression: noise_progression_salt_pepper.png")
    print(f"🔄 Noise comparison: noise_types_comparison.png")
    
    return {
        'animation': gif_path,
        'summary': '../sample_data/noise_levels_summary.png',
        'progressive': progressive_path
    }


if __name__ == "__main__":
    results = main()
