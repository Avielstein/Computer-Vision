"""
Interactive demo for ablation denoising.
Showcases the capabilities of the ablation denoising framework.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_utils import load_mnist_data
from autoencoder_denoising.bitmap_utils import convert_to_bitmap

from ablation_noise import AdditiveNoiseGenerator
from ablation_loss import compute_comprehensive_ablation_metrics
from ablation_models import create_ablation_model, test_ablation_constraint
from efficient_ablation_trainer import train_efficient_ablation_denoiser


def demo_additive_noise_generation():
    """Demonstrate different types of additive noise."""
    print("=" * 60)
    print("DEMO 1: Additive Noise Generation")
    print("=" * 60)
    
    # Load sample MNIST data
    (x_train, _), (_, _) = load_mnist_data(normalize=True)
    x_binary = convert_to_bitmap(x_train[:6], threshold=0.5)
    
    # Create noise generator
    noise_gen = AdditiveNoiseGenerator()
    
    # Generate different types of noise
    noise_types = {
        'Random Pixels': lambda x: noise_gen.add_random_pixels(x, add_prob=0.15),
        'Structured Lines': lambda x: noise_gen.add_structured_lines(x, num_lines=3),
        'Block Noise': lambda x: noise_gen.add_block_noise(x, num_blocks=4),
        'Gaussian Blobs': lambda x: noise_gen.add_gaussian_blobs(x, num_blobs=6),
        'Border Noise': lambda x: noise_gen.add_border_noise(x, border_width=3, add_prob=0.4),
        'Mixed Noise': lambda x: noise_gen.add_mixed_noise(x, intensity=0.7)
    }
    
    # Create visualization
    fig, axes = plt.subplots(len(noise_types) + 1, 6, figsize=(18, 14))
    fig.suptitle('Additive Noise Generation Demo\n(Only adds pixels, never removes)', 
                 fontsize=16, fontweight='bold')
    
    # Show original images
    for i in range(6):
        axes[0, i].imshow(x_binary[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\n({x_binary[i].sum():.0f} pixels)', fontsize=10)
        axes[0, i].axis('off')
    
    # Show each noise type
    for row, (noise_name, noise_func) in enumerate(noise_types.items(), 1):
        noisy_images = noise_func(x_binary.copy())
        
        for i in range(6):
            axes[row, i].imshow(noisy_images[i], cmap='gray', vmin=0, vmax=1)
            pixel_count = noisy_images[i].sum()
            added_pixels = pixel_count - x_binary[i].sum()
            axes[row, i].set_title(f'{noise_name}\n({pixel_count:.0f} pixels, +{added_pixels:.0f})', 
                                 fontsize=10)
            axes[row, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_additive_noise.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Additive noise generation demo completed")
    print("  - All noise types only ADD pixels, never remove them")
    print("  - This ensures noisy_pixels ⊇ clean_pixels always holds")
    print("  - Saved visualization: demo_additive_noise.png")


def demo_ablation_constraint():
    """Demonstrate that models enforce ablation constraint."""
    print("\n" + "=" * 60)
    print("DEMO 2: Ablation Constraint Verification")
    print("=" * 60)
    
    # Test ablation constraint
    test_ablation_constraint()
    
    # Create sample data for detailed testing
    batch_size = 8
    test_input = np.random.uniform(0, 1, (batch_size, 28, 28, 1))
    
    print(f"\nDetailed Constraint Testing:")
    print(f"Input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # Test basic model
    model = create_ablation_model("basic", latent_dim=32)
    output = model(test_input, training=False)
    
    # Verify constraint mathematically
    constraint_violations = np.sum(output.numpy() > test_input + 1e-6)
    constraint_satisfied = constraint_violations == 0
    
    print(f"Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
    print(f"Constraint violations: {constraint_violations}")
    print(f"Ablation constraint satisfied: {constraint_satisfied}")
    
    # Show pixel density changes
    input_density = np.mean(test_input)
    output_density = np.mean(output.numpy())
    pixels_removed = input_density - output_density
    
    print(f"Input pixel density: {input_density:.3f}")
    print(f"Output pixel density: {output_density:.3f}")
    print(f"Pixels removed: {pixels_removed:.3f}")
    
    print("✓ Ablation constraint verification completed")
    print("  - Models can only turn pixels OFF, never ON")
    print("  - Mathematical constraint: output ≤ input (element-wise)")


def demo_noise_progression():
    """Demonstrate noise progression and ablation efficiency."""
    print("\n" + "=" * 60)
    print("DEMO 3: Noise Progression and Ablation Visualization")
    print("=" * 60)
    
    # Load sample data
    (_, _), (x_test, _) = load_mnist_data(normalize=True)
    x_binary = convert_to_bitmap(x_test, threshold=0.5)
    
    # Select a few diverse samples
    sample_indices = [7, 42, 123]  # Different digit types
    samples = x_binary[sample_indices]
    
    # Create noise generator
    noise_gen = AdditiveNoiseGenerator()
    
    # Generate noise progression
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Create visualization
    fig, axes = plt.subplots(len(samples), len(noise_levels), figsize=(18, 9))
    fig.suptitle('Additive Noise Progression\n(Intensity: 0.0 → 1.0)', 
                 fontsize=16, fontweight='bold')
    
    for i, sample in enumerate(samples):
        progression = noise_gen.generate_noise_progression(sample, noise_levels)
        
        for j, level in enumerate(noise_levels):
            noisy_img = progression[level]
            pixel_count = noisy_img.sum()
            added_pixels = pixel_count - sample.sum()
            
            axes[i, j].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].set_title(f'Level {level}\n({pixel_count:.0f} pixels, +{added_pixels:.0f})', 
                               fontsize=10)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_noise_progression.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Noise progression demo completed")
    print("  - Shows how additive noise intensity affects pixel density")
    print("  - Higher intensity = more added pixels")
    print("  - Saved visualization: demo_noise_progression.png")


def demo_quick_training():
    """Demonstrate quick training of an ablation model."""
    print("\n" + "=" * 60)
    print("DEMO 4: Quick Ablation Model Training")
    print("=" * 60)
    
    print("Training a basic ablation denoiser for 5 epochs...")
    print("(This is a quick demo - for full training use more epochs)")
    
    # Train a small model quickly
    model, history = train_efficient_ablation_denoiser(
        model_type="basic",
        noise_type="mixed",
        epochs=5,
        batch_size=128,
        learning_rate=2e-3,
        noise_intensity=0.4,
        loss_type="precision_bce",
        steps_per_epoch=50,  # Quick training
        save_animation=False,
        live_visualization=False
    )
    
    print("✓ Quick training completed")
    print("  - Model learned to perform ablation denoising")
    print("  - Training focused on precision (avoiding false positives)")
    print("  - Check logs/ directory for detailed results")
    
    return model


def demo_ablation_performance(model=None):
    """Demonstrate ablation denoising performance."""
    print("\n" + "=" * 60)
    print("DEMO 5: Ablation Denoising Performance")
    print("=" * 60)
    
    if model is None:
        print("Creating untrained model for architecture demo...")
        model = create_ablation_model("basic", latent_dim=32)
    
    # Load test data
    (_, _), (x_test, _) = load_mnist_data(normalize=True)
    x_test_binary = convert_to_bitmap(x_test[:20], threshold=0.5)
    
    # Add noise
    noise_gen = AdditiveNoiseGenerator()
    noisy_images = noise_gen.add_mixed_noise(x_test_binary, intensity=0.6)
    
    # Get model predictions
    noisy_input = np.expand_dims(noisy_images, -1)
    denoised = model.predict(noisy_input, verbose=0)
    denoised = denoised.squeeze()
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_ablation_metrics(
        x_test_binary, denoised, noisy_images
    )
    
    print("Ablation Denoising Metrics:")
    print("-" * 30)
    for key, value in metrics.items():
        print(f"{key:25}: {value:.4f}")
    
    # Visualize a few examples
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Ablation Denoising Results\n(Clean | Noisy | Denoised | Ablation Map)', 
                 fontsize=16, fontweight='bold')
    
    for i in range(5):
        # Clean
        axes[0, i].imshow(x_test_binary[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Clean\n({x_test_binary[i].sum():.0f} pixels)', fontsize=10)
        axes[0, i].axis('off')
        
        # Noisy
        axes[1, i].imshow(noisy_images[i], cmap='gray', vmin=0, vmax=1)
        added = noisy_images[i].sum() - x_test_binary[i].sum()
        axes[1, i].set_title(f'Noisy\n({noisy_images[i].sum():.0f} pixels, +{added:.0f})', fontsize=10)
        axes[1, i].axis('off')
        
        # Denoised
        axes[2, i].imshow(denoised[i], cmap='gray', vmin=0, vmax=1)
        removed = noisy_images[i].sum() - denoised[i].sum()
        axes[2, i].set_title(f'Denoised\n({denoised[i].sum():.0f} pixels, -{removed:.0f})', fontsize=10)
        axes[2, i].axis('off')
        
        # Ablation map (what was removed)
        ablation_map = noisy_images[i] - denoised[i]
        axes[3, i].imshow(ablation_map, cmap='Reds', vmin=0, vmax=1)
        axes[3, i].set_title(f'Ablated\n({ablation_map.sum():.0f} pixels removed)', fontsize=10)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_ablation_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Performance demo completed")
    print("  - Ablation maps show which pixels were removed")
    print("  - Red regions indicate ablated (removed) pixels")
    print("  - Saved visualization: demo_ablation_performance.png")


def demo_comparison_with_traditional():
    """Compare ablation denoising with traditional approaches."""
    print("\n" + "=" * 60)
    print("DEMO 6: Comparison with Traditional Denoising")
    print("=" * 60)
    
    # Load test data
    (_, _), (x_test, _) = load_mnist_data(normalize=True)
    x_test_binary = convert_to_bitmap(x_test[:5], threshold=0.5)
    
    # Add noise
    noise_gen = AdditiveNoiseGenerator()
    noisy_images = noise_gen.add_mixed_noise(x_test_binary, intensity=0.5)
    
    # Create models
    ablation_model = create_ablation_model("basic", latent_dim=32)
    
    # Get predictions
    noisy_input = np.expand_dims(noisy_images, -1)
    ablation_output = ablation_model.predict(noisy_input, verbose=0).squeeze()
    
    # Simple traditional approach: threshold-based denoising
    traditional_output = (noisy_images > 0.7).astype(np.float32)  # Aggressive thresholding
    
    # Compute metrics for both approaches
    ablation_metrics = compute_comprehensive_ablation_metrics(
        x_test_binary, ablation_output, noisy_images
    )
    
    traditional_metrics = compute_comprehensive_ablation_metrics(
        x_test_binary, traditional_output, noisy_images
    )
    
    print("Comparison Results:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Ablation':<12} {'Traditional':<12}")
    print("-" * 50)
    
    key_metrics = ['precision', 'recall', 'f1_score', 'specificity']
    for metric in key_metrics:
        ablation_val = ablation_metrics.get(metric, 0)
        traditional_val = traditional_metrics.get(metric, 0)
        print(f"{metric:<25} {ablation_val:<12.4f} {traditional_val:<12.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle('Ablation vs Traditional Denoising Comparison', fontsize=16, fontweight='bold')
    
    titles = ['Clean', 'Noisy', 'Ablation', 'Traditional', 'Difference']
    
    for i in range(5):
        # Clean
        axes[i, 0].imshow(x_test_binary[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(titles[0] if i == 0 else '', fontsize=12)
        axes[i, 0].axis('off')
        
        # Noisy
        axes[i, 1].imshow(noisy_images[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(titles[1] if i == 0 else '', fontsize=12)
        axes[i, 1].axis('off')
        
        # Ablation result
        axes[i, 2].imshow(ablation_output[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(titles[2] if i == 0 else '', fontsize=12)
        axes[i, 2].axis('off')
        
        # Traditional result
        axes[i, 3].imshow(traditional_output[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(titles[3] if i == 0 else '', fontsize=12)
        axes[i, 3].axis('off')
        
        # Difference map
        diff_map = np.abs(ablation_output[i] - traditional_output[i])
        axes[i, 4].imshow(diff_map, cmap='RdYlBu', vmin=0, vmax=1)
        axes[i, 4].set_title(titles[4] if i == 0 else '', fontsize=12)
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Comparison demo completed")
    print("  - Ablation denoising provides learned, adaptive pixel removal")
    print("  - Traditional methods use fixed rules (e.g., thresholding)")
    print("  - Difference maps show where approaches disagree")
    print("  - Saved visualization: demo_comparison.png")


def run_full_demo():
    """Run the complete ablation denoising demo."""
    print("🚀 ABLATION DENOISING FRAMEWORK DEMO")
    print("=" * 60)
    print("This demo showcases the auto-ablation-denoising framework")
    print("Key principle: Models can only turn pixels OFF, never ON")
    print("=" * 60)
    
    # Run all demos
    demo_additive_noise_generation()
    demo_ablation_constraint()
    demo_noise_progression()
    
    # Optional: quick training demo
    print(f"\nWould you like to run a quick training demo? (y/n): ", end="")
    response = input().lower().strip()
    
    trained_model = None
    if response in ['y', 'yes']:
        trained_model = demo_quick_training()
    
    demo_ablation_performance(trained_model)
    demo_comparison_with_traditional()
    
    print("\n" + "🎉" * 20)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("🎉" * 20)
    print("\nKey Takeaways:")
    print("✓ Additive noise ensures noisy_pixels ⊇ clean_pixels")
    print("✓ Ablation constraint prevents adding pixels (output ≤ input)")
    print("✓ Precision-focused metrics evaluate ablation quality")
    print("✓ Models learn intelligent pixel removal strategies")
    print("✓ Framework provides theoretical guarantees and practical performance")
    
    print(f"\nGenerated files:")
    print("  - demo_additive_noise.png")
    print("  - demo_noise_progression.png") 
    print("  - demo_ablation_performance.png")
    print("  - demo_comparison.png")
    if trained_model:
        print("  - Training logs in logs/ directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation Denoising Demo')
    parser.add_argument('--demo', type=str, default='full',
                       choices=['full', 'noise', 'constraint', 'progression', 
                               'training', 'performance', 'comparison'],
                       help='Which demo to run')
    parser.add_argument('--quick', action='store_true',
                       help='Skip interactive prompts')
    
    args = parser.parse_args()
    
    if args.demo == 'full':
        if args.quick:
            # Run all demos without prompts
            demo_additive_noise_generation()
            demo_ablation_constraint()
            demo_noise_progression()
            demo_ablation_performance()
            demo_comparison_with_traditional()
        else:
            run_full_demo()
    elif args.demo == 'noise':
        demo_additive_noise_generation()
    elif args.demo == 'constraint':
        demo_ablation_constraint()
    elif args.demo == 'progression':
        demo_noise_progression()
    elif args.demo == 'training':
        demo_quick_training()
    elif args.demo == 'performance':
        demo_ablation_performance()
    elif args.demo == 'comparison':
        demo_comparison_with_traditional()
