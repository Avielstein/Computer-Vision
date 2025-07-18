"""
Train a contrastive learning model to create a pretrained encoder for MNISTMAX framework.
This encoder can then be used in other components like denoising autoencoders.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from shared.data_processing.data_utils import load_mnist_data, create_contrastive_pairs, save_representations
from shared.visualization.visualization import plot_training_history, plot_embeddings_2d
from contrastive_learning.models import create_contrastive_model
from contrastive_learning.contrastive_loss import NTXentLoss
from contrastive_learning.data_augmentation import create_augmentation_pipeline


def create_pretrained_encoder_config():
    """Create configuration for pretrained encoder training."""
    return {
        'encoder_type': 'simple_cnn',  # Start with simple CNN for reliability
        'embedding_dim': 128,
        'projection_dim': 64,
        'batch_size': 256,
        'epochs': 50,
        'learning_rate': 1e-3,
        'temperature': 0.1,
        'augmentation_strength': 'moderate',
        'save_representations': True,
        'save_encoder_only': True,
        'log_dir': None  # Will be set automatically
    }


def train_pretrained_encoder(config=None, verbose=1):
    """
    Train a contrastive learning model to create a pretrained encoder.
    
    Args:
        config: Configuration dictionary (uses default if None)
        verbose: Verbosity level
        
    Returns:
        Tuple of (encoder_model, full_model, training_history, representations)
    """
    if config is None:
        config = create_pretrained_encoder_config()
    
    # Set up logging directory
    if config['log_dir'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['log_dir'] = f"logs/pretrained_encoder_{config['encoder_type']}_{timestamp}"
    
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['log_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose >= 1:
        print("=" * 60)
        print("MNISTMAX Pretrained Encoder Training")
        print("=" * 60)
        print(f"Encoder type: {config['encoder_type']}")
        print(f"Embedding dim: {config['embedding_dim']}")
        print(f"Projection dim: {config['projection_dim']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Temperature: {config['temperature']}")
        print(f"Log directory: {config['log_dir']}")
        print()
    
    # Load MNIST data
    if verbose >= 1:
        print("Loading MNIST dataset...")
    
    (x_train, y_train), (x_test, y_test) = load_mnist_data(normalize=True)
    
    # Reshape for CNN if needed
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    
    if verbose >= 1:
        print(f"Training samples: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Input shape: {x_train.shape[1:]}")
        print()
    
    # Create augmentation pipeline
    if verbose >= 1:
        print("Creating data augmentation pipeline...")
    
    augmentation_pipeline = create_augmentation_pipeline(
        strength=config['augmentation_strength']
    )
    
    # Create contrastive pairs dataset
    if verbose >= 1:
        print("Creating contrastive learning dataset...")
    
    train_dataset = create_contrastive_pairs(
        x_train, 
        batch_size=config['batch_size'],
        augmentation_fn=augmentation_pipeline
    )
    
    val_dataset = create_contrastive_pairs(
        x_test[:5000],  # Use subset for validation
        batch_size=config['batch_size'],
        augmentation_fn=augmentation_pipeline
    )
    
    # Create contrastive model
    if verbose >= 1:
        print("Creating contrastive learning model...")
    
    model = create_contrastive_model(
        encoder_type=config['encoder_type'],
        embedding_dim=config['embedding_dim'],
        projection_dim=config['projection_dim'],
        input_shape=x_train.shape[1:]
    )
    
    if verbose >= 1:
        print(f"Model parameters: {model.count_params():,}")
        print()
    
    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    # Create custom training step for contrastive learning
    loss_fn = NTXentLoss(temperature=config['temperature'])
    
    @tf.function
    def train_step(batch):
        anchor, positive = batch
        
        with tf.GradientTape() as tape:
            # Get embeddings and projections
            anchor_embeddings, anchor_projections = model(anchor, training=True)
            positive_embeddings, positive_projections = model(positive, training=True)
            
            # Compute contrastive loss on projections
            loss = loss_fn(anchor_projections, positive_projections)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    @tf.function
    def val_step(batch):
        anchor, positive = batch
        
        # Get embeddings and projections
        anchor_embeddings, anchor_projections = model(anchor, training=False)
        positive_embeddings, positive_projections = model(positive, training=False)
        
        # Compute contrastive loss on projections
        loss = loss_fn(anchor_projections, positive_projections)
        
        return loss
    
    # Train model with custom training loop
    if verbose >= 1:
        print("Starting contrastive learning training...")
        print("-" * 40)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        if verbose >= 1:
            print(f"Epoch {epoch + 1}/{config['epochs']}")
        
        # Training
        train_losses = []
        for batch in train_dataset:
            loss = train_step(batch)
            train_losses.append(float(loss))
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        val_losses = []
        for batch in val_dataset:
            loss = val_step(batch)
            val_losses.append(float(loss))
        
        avg_val_loss = np.mean(val_losses)
        
        # Record history
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if verbose >= 1:
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(os.path.join(config['log_dir'], 'best_model.weights.h5'))
            patience_counter = 0
            if verbose >= 1:
                print(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            if verbose >= 1:
                print(f"  Early stopping after {epoch + 1} epochs")
            break
        
        # Learning rate reduction
        if patience_counter >= 5:
            new_lr = optimizer.learning_rate * 0.5
            if new_lr >= 1e-6:
                optimizer.learning_rate.assign(new_lr)
                if verbose >= 1:
                    print(f"  Reduced learning rate to {float(new_lr):.6f}")
                patience_counter = 0  # Reset patience after LR reduction
    
    # Load best weights
    model.load_weights(os.path.join(config['log_dir'], 'best_model.weights.h5'))
    
    if verbose >= 1:
        print("\nTraining completed!")
        print("-" * 40)
    
    # Plot training history
    if verbose >= 1:
        print("Plotting training history...")
    
    plot_training_history(
        history,
        save_path=os.path.join(config['log_dir'], 'training_history.png')
    )
    
    # Extract encoder for reuse
    encoder = model.encoder
    
    # Save encoder separately
    if config['save_encoder_only']:
        encoder_path = os.path.join(config['log_dir'], 'pretrained_encoder.weights.h5')
        encoder.save_weights(encoder_path)
        if verbose >= 1:
            print(f"Pretrained encoder saved: {encoder_path}")
    
    # Generate and save representations
    representations = None
    if config['save_representations']:
        if verbose >= 1:
            print("Generating learned representations...")
        
        # Use test set for representation generation
        test_embeddings = encoder.predict(x_test, verbose=0)
        
        # Save representations
        representations_path = os.path.join(config['log_dir'], 'learned_representations.npz')
        save_representations(
            embeddings=test_embeddings,
            labels=y_test,
            images=x_test,
            save_path=representations_path
        )
        
        representations = {
            'embeddings': test_embeddings,
            'labels': y_test,
            'images': x_test
        }
        
        if verbose >= 1:
            print(f"Representations saved: {representations_path}")
        
        # Create t-SNE visualization
        if verbose >= 1:
            print("Creating t-SNE visualization...")
        
        try:
            plot_embeddings_2d(
                test_embeddings[:2000],  # Use subset for speed
                y_test[:2000],
                method='tsne',
                save_path=os.path.join(config['log_dir'], 'embeddings_tsne.png')
            )
        except ImportError:
            if verbose >= 1:
                print("Skipping t-SNE visualization (scikit-learn not available)")
    
    # Evaluate representation quality with linear probing
    if verbose >= 1:
        print("Evaluating representation quality with linear probing...")
    
    # Create linear classifier on top of frozen encoder
    linear_classifier = keras.Sequential([
        encoder,
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Freeze encoder weights
    encoder.trainable = False
    
    # Compile and train linear classifier
    linear_classifier.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train linear classifier
    linear_history = linear_classifier.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=256,
        verbose=1 if verbose >= 1 else 0
    )
    
    # Get final accuracy
    final_accuracy = max(linear_history.history['val_accuracy'])
    
    if verbose >= 1:
        print(f"\nLinear probing accuracy: {final_accuracy:.4f}")
        print("This indicates the quality of learned representations.")
    
    # Save linear probing results
    linear_results = {
        'final_accuracy': float(final_accuracy),
        'history': linear_history.history
    }
    
    with open(os.path.join(config['log_dir'], 'linear_probing_results.json'), 'w') as f:
        json.dump(linear_results, f, indent=2)
    
    # Save complete model
    model.save_weights(os.path.join(config['log_dir'], 'full_contrastive_model.weights.h5'))
    
    if verbose >= 1:
        print("\n" + "=" * 60)
        print("Pretrained Encoder Training Summary")
        print("=" * 60)
        print(f"✅ Encoder type: {config['encoder_type']}")
        print(f"✅ Embedding dimension: {config['embedding_dim']}")
        print(f"✅ Linear probing accuracy: {final_accuracy:.4f}")
        print(f"✅ Model saved to: {config['log_dir']}")
        print(f"✅ Encoder weights: pretrained_encoder.weights.h5")
        if config['save_representations']:
            print(f"✅ Representations: learned_representations.npz")
        print("\nThis encoder can now be used in other MNISTMAX components!")
        print("=" * 60)
    
    return encoder, model, history, representations


def load_pretrained_encoder(encoder_path, encoder_type='simple_cnn', embedding_dim=128):
    """
    Load a pretrained encoder from saved weights.
    
    Args:
        encoder_path: Path to saved encoder weights
        encoder_type: Type of encoder architecture
        embedding_dim: Embedding dimension
        
    Returns:
        Loaded encoder model
    """
    # Create encoder architecture
    from contrastive_learning.models import create_contrastive_model
    
    dummy_model = create_contrastive_model(
        encoder_type=encoder_type,
        embedding_dim=embedding_dim,
        input_shape=(28, 28, 1)
    )
    
    encoder = dummy_model.encoder
    
    # Load weights
    encoder.load_weights(encoder_path)
    
    return encoder


def demo_pretrained_encoder_usage():
    """Demonstrate how to use the pretrained encoder in other components."""
    print("Demo: Using pretrained encoder in a denoising autoencoder")
    print("-" * 50)
    
    # This would be used in other components like:
    # 1. Load pretrained encoder
    # encoder = load_pretrained_encoder('logs/pretrained_encoder_xxx/pretrained_encoder.weights.h5')
    
    # 2. Use as feature extractor in denoising autoencoder
    # inputs = keras.Input(shape=(28, 28, 1))
    # features = encoder(inputs)  # Extract learned features
    # decoder_output = decoder(features)  # Decode to clean image
    
    # 3. Fine-tune or freeze encoder as needed
    # encoder.trainable = False  # Freeze for transfer learning
    
    print("See other MNISTMAX components for actual usage examples.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pretrained encoder for MNISTMAX framework')
    parser.add_argument('--encoder_type', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'resnet', 'vit'],
                       help='Type of encoder architecture')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=64,
                       help='Projection dimension for contrastive learning')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for contrastive loss')
    parser.add_argument('--augmentation', type=str, default='moderate',
                       choices=['light', 'moderate', 'strong'],
                       help='Data augmentation strength')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo of pretrained encoder usage')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_pretrained_encoder_usage()
    else:
        # Create configuration from arguments
        config = {
            'encoder_type': args.encoder_type,
            'embedding_dim': args.embedding_dim,
            'projection_dim': args.projection_dim,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'temperature': args.temperature,
            'augmentation_strength': args.augmentation,
            'save_representations': True,
            'save_encoder_only': True,
            'log_dir': None
        }
        
        # Train pretrained encoder
        encoder, model, history, representations = train_pretrained_encoder(config)
