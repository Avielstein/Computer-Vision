"""
Example usage of the contrastive learning framework for MNIST.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Import our modules
from data_augmentation import create_mild_augmentation
from contrastive_loss import create_contrastive_loss
from models import create_contrastive_model
from train_contrastive import ContrastiveTrainer, evaluate_representations


def quick_demo():
    """Quick demonstration of contrastive learning on MNIST."""
    print("🚀 MNISTMAX Contrastive Learning Demo")
    print("=" * 50)
    
    # Load and prepare data
    print("📊 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Use subset for quick demo
    x_train = x_train[:5000].astype('float32') / 255.0
    y_train = y_train[:5000]
    x_test = x_test[:1000].astype('float32') / 255.0
    y_test = y_test[:1000]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Create datasets with labels for supervised contrastive learning
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(128).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(128).prefetch(tf.data.AUTOTUNE)
    
    # Create model components
    print("\n🏗️  Creating model...")
    model = create_contrastive_model(
        encoder_type="simple_cnn",
        embedding_dim=64,
        projection_dim=32
    )
    
    loss_fn = create_contrastive_loss("ntxent", temperature=0.1)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    augmentation = create_mild_augmentation()
    
    print(f"Model: {model.encoder_type}")
    print(f"Embedding dim: {model.embedding_dim}")
    print(f"Projection dim: {model.projection_dim}")
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        augmentation=augmentation,
        log_dir="demo_logs"
    )
    
    # Train for a few epochs
    print("\n🎯 Training contrastive model...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,  # Quick demo
        verbose=1
    )
    
    # Visualize training
    print("\n📈 Training results:")
    trainer.plot_training_history()
    
    # Evaluate representations
    print("\n🔍 Evaluating learned representations...")
    x_train_eval = x_train.reshape(-1, 28, 28, 1)
    x_test_eval = x_test.reshape(-1, 28, 28, 1)
    
    results = evaluate_representations(
        model, x_train_eval, y_train, x_test_eval, y_test
    )
    
    print("\n✅ Demo completed!")
    print(f"Final test accuracy: {results['test_accuracy']:.3f}")
    
    return model, history, results


def visualize_augmentations():
    """Visualize the data augmentations."""
    print("\n🎨 Visualizing Data Augmentations")
    print("-" * 40)
    
    # Load sample image
    (x_train, _), _ = keras.datasets.mnist.load_data()
    sample_image = x_train[0].astype('float32') / 255.0
    
    # Create augmentation
    augmentation = create_mild_augmentation()
    
    # Generate augmented pairs
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Data Augmentation Examples', fontsize=16)
    
    for i in range(5):
        aug1, aug2 = augmentation.augment_pair(sample_image)
        
        axes[0, i].imshow(aug1, cmap='gray')
        axes[0, i].set_title(f'Augmentation 1 - {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(aug2, cmap='gray')
        axes[1, i].set_title(f'Augmentation 2 - {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_encoders():
    """Compare different encoder architectures."""
    print("\n🏛️  Comparing Encoder Architectures")
    print("-" * 40)
    
    encoders = ['simple_cnn', 'resnet']  # Skip 'vit' for quick demo
    results = {}
    
    # Load small dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[:2000].astype('float32') / 255.0
    y_train = y_train[:2000]
    x_test = x_test[:500].astype('float32') / 255.0
    y_test = y_test[:500]
    
    for encoder_type in encoders:
        print(f"\nTesting {encoder_type}...")
        
        # Create model
        model = create_contrastive_model(
            encoder_type=encoder_type,
            embedding_dim=64,
            projection_dim=32
        )
        
        # Quick training setup
        loss_fn = create_contrastive_loss("ntxent", temperature=0.1)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        augmentation = create_mild_augmentation()
        
        trainer = ContrastiveTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            augmentation=augmentation,
            log_dir=f"demo_logs_{encoder_type}"
        )
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(64)
        
        # Train briefly
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            verbose=0
        )
        
        # Evaluate
        x_train_eval = x_train.reshape(-1, 28, 28, 1)
        x_test_eval = x_test.reshape(-1, 28, 28, 1)
        
        eval_results = evaluate_representations(
            model, x_train_eval, y_train, x_test_eval, y_test
        )
        
        results[encoder_type] = {
            'final_loss': history['val_loss'][-1],
            'test_accuracy': eval_results['test_accuracy']
        }
        
        print(f"{encoder_type}: Loss={results[encoder_type]['final_loss']:.3f}, "
              f"Accuracy={results[encoder_type]['test_accuracy']:.3f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    encoders_list = list(results.keys())
    losses = [results[enc]['final_loss'] for enc in encoders_list]
    accuracies = [results[enc]['test_accuracy'] for enc in encoders_list]
    
    ax1.bar(encoders_list, losses, alpha=0.7, color='skyblue')
    ax1.set_title('Final Validation Loss')
    ax1.set_ylabel('Loss')
    
    ax2.bar(encoders_list, accuracies, alpha=0.7, color='lightcoral')
    ax2.set_title('Test Accuracy')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Run all demonstrations."""
    print("🎯 MNISTMAX: Contrastive Learning Framework")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    try:
        # 1. Quick demo
        model, history, results = quick_demo()
        
        # 2. Visualize augmentations
        visualize_augmentations()
        
        # 3. Compare encoders
        comparison_results = compare_encoders()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nTo run full training, use:")
        print("python train_contrastive.py --encoder simple_cnn --epochs 100 --evaluate")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install tensorflow matplotlib numpy")


if __name__ == "__main__":
    main()
