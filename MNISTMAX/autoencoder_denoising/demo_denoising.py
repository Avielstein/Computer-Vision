"""
Interactive demo for denoising autoencoders.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_utils import load_mnist_data
from shared.visualization import plot_noise_examples, plot_denoising_comparison
from noise_generation import (
    NoiseGenerator, create_default_noise_generator, create_aggressive_noise_generator, 
    create_mild_noise_generator, salt_and_pepper_noise, random_pixel_flip_noise,
    gaussian_noise_binary, structured_noise_lines, block_noise
)
from bitmap_utils import convert_to_bitmap, compute_bitmap_metrics, visualize_bitmap_comparison
from denoising_models import create_denoising_model, get_denoising_loss, get_denoising_metrics
from train_denoiser import DenoisingTrainer


def demo_noise_types():
    """Demonstrate different types of noise."""
    print("🎨 Noise Types Demonstration")
    print("=" * 40)
    
    # Load sample images
    (x_train, _), _ = load_mnist_data(normalize=True)
    
    # Convert to binary
    x_binary = convert_to_bitmap(x_train[:5], threshold=0.5)
    
    # Define noise functions
    noise_functions = {
        'Salt & Pepper': lambda x: salt_and_pepper_noise(x, noise_prob=0.1),
        'Pixel Flip': lambda x: random_pixel_flip_noise(x, flip_prob=0.05),
        'Gaussian': lambda x: gaussian_noise_binary(x, noise_std=0.1),
        'Lines': lambda x: structured_noise_lines(x, num_lines=3),
        'Blocks': lambda x: block_noise(x, num_blocks=4)
    }
    
    # Show noise examples
    plot_noise_examples(x_binary, noise_functions, n_samples=3)
    
    print("Different noise types demonstrated!")
    return x_binary, noise_functions


def demo_quick_training():
    """Quick training demo with minimal epochs."""
    print("\n🚀 Quick Training Demo")
    print("=" * 40)
    
    # Load data
    (x_train, _), (x_test, _) = load_mnist_data(normalize=True)
    
    # Use subset for quick demo
    x_train_subset = convert_to_bitmap(x_train[:1000], threshold=0.5)
    x_test_subset = convert_to_bitmap(x_test[:200], threshold=0.5)
    
    print(f"Using {len(x_train_subset)} training and {len(x_test_subset)} test samples")
    
    # Create simple model
    model = create_denoising_model(model_type="basic")
    
    # Create noise generator
    noise_generator = create_mild_noise_generator()
    
    # Create trainer
    trainer = DenoisingTrainer(
        model=model,
        noise_generator=noise_generator,
        loss_fn=get_denoising_loss("binary_crossentropy"),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=get_denoising_metrics(),
        log_dir="demo_logs/quick_training",
        live_visualization=False  # Disable for demo
    )
    
    # Quick training
    print("Training for 3 epochs...")
    history = trainer.train(
        x_train=x_train_subset,
        x_val=x_test_subset,
        epochs=3,
        batch_size=32,
        clean_ratio=0.5,
        verbose=1
    )
    
    # Test the model
    print("\nTesting the trained model...")
    
    # Get some test samples
    test_samples = x_test_subset[:5]
    noisy_samples = noise_generator.add_noise(test_samples, 'salt_pepper')
    
    # Add channel dimension
    test_input = np.expand_dims(noisy_samples, -1)
    
    # Get predictions
    denoised = model.predict(test_input, verbose=0)
    denoised = denoised.squeeze()
    
    # Visualize results
    visualize_bitmap_comparison(test_samples, noisy_samples, denoised, n_samples=5)
    
    # Compute metrics
    metrics = compute_bitmap_metrics(test_samples, denoised)
    print(f"\nQuick Training Results:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
    print(f"  Dice Coefficient: {metrics['dice_coefficient']['mean']:.3f}")
    print(f"  IoU Score: {metrics['jaccard_index']['mean']:.3f}")
    
    return model, history


def demo_model_comparison():
    """Compare different model architectures."""
    print("\n🏛️  Model Architecture Comparison")
    print("=" * 40)
    
    # Load small dataset
    (x_train, _), (x_test, _) = load_mnist_data(normalize=True)
    x_train_small = convert_to_bitmap(x_train[:500], threshold=0.5)
    x_test_small = convert_to_bitmap(x_test[:100], threshold=0.5)
    
    # Model types to compare
    model_types = ['basic', 'unet']  # Skip complex models for quick demo
    results = {}
    
    noise_generator = create_default_noise_generator()
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model...")
        
        # Create model
        model = create_denoising_model(model_type=model_type)
        
        # Create trainer
        trainer = DenoisingTrainer(
            model=model,
            noise_generator=noise_generator,
            loss_fn=get_denoising_loss("binary_crossentropy"),
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=get_denoising_metrics(),
            log_dir=f"demo_logs/comparison_{model_type}",
            live_visualization=False
        )
        
        # Quick training
        history = trainer.train(
            x_train=x_train_small,
            x_val=x_test_small,
            epochs=2,
            batch_size=32,
            clean_ratio=0.5,
            verbose=0
        )
        
        # Evaluate
        test_samples = x_test_small[:20]
        noisy_samples = noise_generator.add_noise(test_samples, 'salt_pepper')
        test_input = np.expand_dims(noisy_samples, -1)
        denoised = model.predict(test_input, verbose=0).squeeze()
        
        metrics = compute_bitmap_metrics(test_samples, denoised)
        
        results[model_type] = {
            'final_loss': history['val_loss'][-1],
            'pixel_accuracy': metrics['pixel_accuracy'],
            'dice_coefficient': metrics['dice_coefficient']['mean']
        }
        
        print(f"  Final Loss: {results[model_type]['final_loss']:.3f}")
        print(f"  Pixel Accuracy: {results[model_type]['pixel_accuracy']:.3f}")
        print(f"  Dice Coefficient: {results[model_type]['dice_coefficient']:.3f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = list(results.keys())
    losses = [results[m]['final_loss'] for m in models]
    accuracies = [results[m]['pixel_accuracy'] for m in models]
    dice_scores = [results[m]['dice_coefficient'] for m in models]
    
    axes[0].bar(models, losses, alpha=0.7, color='skyblue')
    axes[0].set_title('Final Validation Loss')
    axes[0].set_ylabel('Loss')
    
    axes[1].bar(models, accuracies, alpha=0.7, color='lightcoral')
    axes[1].set_title('Pixel Accuracy')
    axes[1].set_ylabel('Accuracy')
    
    axes[2].bar(models, dice_scores, alpha=0.7, color='lightgreen')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_ylabel('Dice Score')
    
    plt.tight_layout()
    plt.show()
    
    return results


def demo_noise_robustness():
    """Test model robustness to different noise types."""
    print("\n🛡️  Noise Robustness Test")
    print("=" * 40)
    
    # Load data and train a model quickly
    (x_train, _), (x_test, _) = load_mnist_data(normalize=True)
    x_train_small = convert_to_bitmap(x_train[:800], threshold=0.5)
    x_test_small = convert_to_bitmap(x_test[:100], threshold=0.5)
    
    # Train a model
    model = create_denoising_model(model_type="unet")
    noise_generator = create_default_noise_generator()
    
    trainer = DenoisingTrainer(
        model=model,
        noise_generator=noise_generator,
        loss_fn=get_denoising_loss("binary_crossentropy"),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=get_denoising_metrics(),
        log_dir="demo_logs/robustness",
        live_visualization=False
    )
    
    print("Training model for robustness test...")
    trainer.train(
        x_train=x_train_small,
        x_val=x_test_small,
        epochs=5,
        batch_size=32,
        clean_ratio=0.5,
        verbose=0
    )
    
    # Test on different noise types
    noise_types = ['salt_pepper', 'pixel_flip', 'gaussian', 'blocks']
    test_samples = x_test_small[:20]
    
    results = {}
    
    for noise_type in noise_types:
        print(f"\nTesting on {noise_type} noise...")
        
        # Add noise
        noisy_samples = noise_generator.add_noise(test_samples, noise_type)
        test_input = np.expand_dims(noisy_samples, -1)
        
        # Get predictions
        denoised = model.predict(test_input, verbose=0).squeeze()
        
        # Compute metrics
        metrics = compute_bitmap_metrics(test_samples, denoised)
        results[noise_type] = metrics['pixel_accuracy']
        
        print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
        
        # Show examples for first noise type
        if noise_type == noise_types[0]:
            visualize_bitmap_comparison(
                test_samples[:3], 
                noisy_samples[:3], 
                denoised[:3], 
                n_samples=3
            )
    
    # Plot robustness results
    plt.figure(figsize=(10, 6))
    noise_names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(noise_names, accuracies, alpha=0.7, color='orange')
    plt.title('Model Robustness to Different Noise Types')
    plt.xlabel('Noise Type')
    plt.ylabel('Pixel Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results


def demo_interactive_denoising():
    """Interactive denoising demonstration."""
    print("\n🎮 Interactive Denoising Demo")
    print("=" * 40)
    
    # Load some test images
    (x_train, _), _ = load_mnist_data(normalize=True)
    x_binary = convert_to_bitmap(x_train[:10], threshold=0.5)
    
    # Create a simple trained model (or load if available)
    model = create_denoising_model(model_type="basic")
    
    # For demo purposes, we'll create a simple "trained" model
    # In practice, you would load a pre-trained model
    print("Note: Using a basic model for demonstration.")
    print("For best results, train the model first using train_denoiser.py")
    
    # Create noise generator
    noise_generator = create_default_noise_generator()
    
    # Interactive loop
    while True:
        print("\nChoose an option:")
        print("1. Add salt & pepper noise")
        print("2. Add pixel flip noise")
        print("3. Add gaussian noise")
        print("4. Add block noise")
        print("5. Add mixed noise")
        print("6. Exit")
        
        try:
            choice = input("Enter choice (1-6): ").strip()
            
            if choice == '6':
                break
            elif choice in ['1', '2', '3', '4', '5']:
                # Select random image
                img_idx = np.random.randint(0, len(x_binary))
                clean_img = x_binary[img_idx:img_idx+1]
                
                # Add noise based on choice
                if choice == '1':
                    noisy_img = noise_generator.add_noise(clean_img, 'salt_pepper')
                    noise_name = "Salt & Pepper"
                elif choice == '2':
                    noisy_img = noise_generator.add_noise(clean_img, 'pixel_flip')
                    noise_name = "Pixel Flip"
                elif choice == '3':
                    noisy_img = noise_generator.add_noise(clean_img, 'gaussian')
                    noise_name = "Gaussian"
                elif choice == '4':
                    noisy_img = noise_generator.add_noise(clean_img, 'blocks')
                    noise_name = "Block"
                elif choice == '5':
                    noisy_img = noise_generator.add_mixed_noise(clean_img, num_noise_types=2)
                    noise_name = "Mixed"
                
                # "Denoise" with model (basic model won't be very effective)
                test_input = np.expand_dims(noisy_img, -1)
                denoised_img = model.predict(test_input, verbose=0).squeeze()
                
                # Show results
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                axes[0].imshow(clean_img[0], cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Original Clean')
                axes[0].axis('off')
                
                axes[1].imshow(noisy_img[0], cmap='gray', vmin=0, vmax=1)
                axes[1].set_title(f'Noisy ({noise_name})')
                axes[1].axis('off')
                
                axes[2].imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
                axes[2].set_title('Denoised')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # Compute metrics
                metrics = compute_bitmap_metrics(clean_img, denoised_img.reshape(1, 28, 28))
                print(f"Denoising Results:")
                print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
                print(f"  Dice Coefficient: {metrics['dice_coefficient']['mean']:.3f}")
                
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run all demonstrations."""
    print("🎯 MNISTMAX Denoising Autoencoder Demo")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # 1. Demonstrate noise types
        x_binary, noise_functions = demo_noise_types()
        
        # 2. Quick training demo
        model, history = demo_quick_training()
        
        # 3. Model comparison
        comparison_results = demo_model_comparison()
        
        # 4. Noise robustness test
        robustness_results = demo_noise_robustness()
        
        # 5. Interactive demo
        print("\n" + "="*60)
        print("All automated demos completed!")
        print("\nWould you like to try the interactive denoising demo?")
        response = input("Enter 'y' for yes, any other key to exit: ").strip().lower()
        
        if response == 'y':
            demo_interactive_denoising()
        
        print("\n🎉 All demonstrations completed!")
        print("\nTo train a full model, run:")
        print("python train_denoiser.py --model unet --epochs 50 --batch_size 32")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install tensorflow matplotlib numpy")


if __name__ == "__main__":
    main()
