"""
Enhanced triplet-based contrastive learning with:
- Fixed PCA axis limits for smooth GIF animation
- Training/validation accuracy tracking
- Advanced callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Extended training up to 100 epochs
- Comprehensive metrics dashboard
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import imageio
from tqdm import tqdm
import pandas as pd

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from shared.data_processing.data_utils import load_mnist_data, save_representations
    from shared.visualization.visualization import plot_training_history
except ImportError:
    # Fallback to direct TensorFlow/Keras imports
    def load_mnist_data(normalize=True):
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if normalize:
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
        return (x_train, y_train), (x_test, y_test)
    
    def save_representations(embeddings, labels, images, save_path):
        np.savez(save_path, embeddings=embeddings, labels=labels, images=images)
    
    def plot_training_history(history, save_path):
        pass  # Will use our custom dashboard instead


class TripletDataGenerator:
    """Generate triplet data for contrastive learning."""
    
    def __init__(self, x_data, y_data, batch_size=64):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.n_classes = len(np.unique(y_data))
        
        # Group indices by class for efficient sampling
        self.class_indices = {}
        for i in range(self.n_classes):
            self.class_indices[i] = np.where(y_data == i)[0]
    
    def apply_consistent_augmentation(self, images):
        """Apply the same augmentation to all images in the batch."""
        # Random rotation (same for all images)
        angle = tf.random.uniform([], -10, 10)
        
        # Apply rotation using tf.image.rot90 for simplicity
        # Convert angle to number of 90-degree rotations
        num_rotations = tf.cast(tf.round(angle / 90), tf.int32)
        num_rotations = tf.clip_by_value(num_rotations, 0, 3)
        
        # Apply rotation
        rotated = tf.image.rot90(images, k=num_rotations)
        
        # Random translation (same for all)
        dx = tf.random.uniform([], -2, 2)
        dy = tf.random.uniform([], -2, 2)
        
        # Apply translation by rolling pixels
        translated = tf.roll(tf.roll(rotated, tf.cast(dx, tf.int32), axis=2), 
                           tf.cast(dy, tf.int32), axis=1)
        
        # Random brightness (same for all)
        brightness_factor = tf.random.uniform([], 0.9, 1.1)
        brightened = tf.clip_by_value(translated * brightness_factor, 0.0, 1.0)
        
        # Small amount of noise (different for each image for variety)
        noise = tf.random.normal(tf.shape(brightened), stddev=0.01)
        final = tf.clip_by_value(brightened + noise, 0.0, 1.0)
        
        return final
    
    def generate_triplets(self):
        """Generate triplets (anchor, positive, negative)."""
        while True:
            anchors = []
            positives = []
            negatives = []
            
            for _ in range(self.batch_size):
                # Sample anchor class
                anchor_class = np.random.randint(0, self.n_classes)
                
                # Sample anchor and positive from same class
                if len(self.class_indices[anchor_class]) < 2:
                    # If class has only one sample, duplicate it
                    anchor_idx = positive_idx = self.class_indices[anchor_class][0]
                else:
                    anchor_idx, positive_idx = np.random.choice(
                        self.class_indices[anchor_class], 2, replace=False
                    )
                
                # Sample negative from different class
                negative_class = np.random.choice(
                    [c for c in range(self.n_classes) if c != anchor_class]
                )
                negative_idx = np.random.choice(self.class_indices[negative_class])
                
                anchors.append(self.x_data[anchor_idx])
                positives.append(self.x_data[positive_idx])
                negatives.append(self.x_data[negative_idx])
            
            # Convert to tensors
            anchors = tf.stack(anchors)
            positives = tf.stack(positives)
            negatives = tf.stack(negatives)
            
            # Apply consistent augmentation to all three sets
            all_images = tf.concat([anchors, positives, negatives], axis=0)
            augmented = self.apply_consistent_augmentation(all_images)
            
            # Split back into triplets
            batch_size = tf.shape(anchors)[0]
            aug_anchors = augmented[:batch_size]
            aug_positives = augmented[batch_size:2*batch_size]
            aug_negatives = augmented[2*batch_size:]
            
            yield (aug_anchors, aug_positives, aug_negatives)


class FastClusteringVisualizer:
    """Create fast PCA clustering visualizations with fixed axis limits for smooth animation."""
    
    def __init__(self, test_images, test_labels, save_dir, n_samples_per_class=25):
        # Sample exactly n_samples_per_class from each digit class with fixed seed
        np.random.seed(42)  # Fixed seed for reproducible sampling
        self.test_images = []
        self.test_labels = []
        
        for digit in range(10):
            digit_indices = np.where(test_labels == digit)[0]
            selected_indices = np.random.choice(
                digit_indices, 
                min(n_samples_per_class, len(digit_indices)), 
                replace=False
            )
            self.test_images.extend(test_images[selected_indices])
            self.test_labels.extend([digit] * len(selected_indices))
        
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)
        
        self.save_dir = save_dir
        self.frames_dir = os.path.join(save_dir, 'clustering_frames')
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Fixed colors for each digit class
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Initialize PCA with fixed random state
        self.pca = PCA(n_components=2, random_state=42)
        
        # Storage for all embeddings to compute global axis limits
        self.all_embeddings = []
        self.fixed_xlim = None
        self.fixed_ylim = None
        
        print(f"📊 Visualization using {len(self.test_images)} samples ({len(self.test_images)//10} per class)")
        
    def store_embeddings(self, embeddings):
        """Store embeddings for computing global axis limits."""
        self.all_embeddings.append(embeddings.copy())
    
    def compute_fixed_axis_limits(self):
        """Compute fixed axis limits from all stored embeddings."""
        if not self.all_embeddings:
            return
        
        print("🔧 Computing fixed axis limits for smooth animation...")
        
        # Stack all embeddings
        global_embeddings = np.vstack(self.all_embeddings)
        
        # Fit PCA on all embeddings
        global_pca_embeddings = self.pca.fit_transform(global_embeddings)
        
        # Compute global min/max with padding
        padding = 0.15  # 15% padding for better visualization
        x_min, x_max = global_pca_embeddings[:, 0].min(), global_pca_embeddings[:, 0].max()
        y_min, y_max = global_pca_embeddings[:, 1].min(), global_pca_embeddings[:, 1].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Set fixed limits with padding
        self.fixed_xlim = (x_min - padding * x_range, x_max + padding * x_range)
        self.fixed_ylim = (y_min - padding * y_range, y_max + padding * y_range)
        
        print(f"   📐 Fixed X limits: {self.fixed_xlim[0]:.2f} to {self.fixed_xlim[1]:.2f}")
        print(f"   📐 Fixed Y limits: {self.fixed_ylim[0]:.2f} to {self.fixed_ylim[1]:.2f}")
        
    def create_clustering_plot(self, embeddings, epoch):
        """Create PCA clustering plot for current epoch with fixed axis limits."""
        # Store embeddings for global axis computation
        self.store_embeddings(embeddings)
        
        # Apply PCA transform
        if not hasattr(self, 'pca_fitted'):
            reduced_2d = self.pca.fit_transform(embeddings)
            self.pca_fitted = True
        else:
            reduced_2d = self.pca.transform(embeddings)
        
        # Create plot with consistent styling
        plt.figure(figsize=(12, 9))
        
        # Plot each digit class with consistent colors
        for digit in range(10):
            mask = self.test_labels == digit
            if np.any(mask):
                plt.scatter(
                    reduced_2d[mask, 0], 
                    reduced_2d[mask, 1],
                    c=[self.colors[digit]], 
                    label=f'Digit {digit}',
                    alpha=0.8,
                    s=60,
                    edgecolors='white',
                    linewidth=0.8
                )
        
        # Apply fixed axis limits if computed
        if self.fixed_xlim is not None and self.fixed_ylim is not None:
            plt.xlim(self.fixed_xlim)
            plt.ylim(self.fixed_ylim)
        
        # Consistent styling
        plt.title(f'Epoch {epoch}: Learned Representations (PCA)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('First Principal Component', fontsize=14, fontweight='bold')
        plt.ylabel('Second Principal Component', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(self.frames_dir, f'epoch_{epoch:03d}_pca.png')
        plt.savefig(frame_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return frame_path
    
    def create_animation(self):
        """Create smooth GIF animation from saved frames with fixed axis limits."""
        frame_files = sorted([
            f for f in os.listdir(self.frames_dir) 
            if f.endswith('_pca.png')
        ])
        
        if not frame_files:
            print("No PCA frames found for animation")
            return
        
        print(f"🎬 Creating smooth PCA animation from {len(frame_files)} frames...")
        
        # Read frames
        frames = []
        for frame_file in tqdm(frame_files, desc="Loading frames"):
            frame_path = os.path.join(self.frames_dir, frame_file)
            frames.append(imageio.imread(frame_path))
        
        # Create GIF with slower duration for better viewing
        gif_path = os.path.join(self.save_dir, 'clustering_evolution_pca.gif')
        imageio.mimsave(gif_path, frames, duration=1.2, loop=0)
        print(f"✅ Smooth clustering animation saved: {gif_path}")
        print(f"   📊 {len(frames)} frames at 1.2s per frame")


def create_metrics_dashboard(history, csv_path, save_path):
    """Create comprehensive training metrics dashboard."""
    
    # Load CSV data if available
    df = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚀 MNISTMAX Enhanced Triplet Training Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax1.set_title('📉 Loss Curves', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Triplet Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Highlight best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax1.annotate(f'Best: Epoch {best_epoch}\nVal Loss: {best_val_loss:.4f}', 
                xy=(best_epoch, best_val_loss), xytext=(best_epoch + len(epochs)*0.1, best_val_loss + 0.1),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, fontweight='bold', color='green')
    
    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax2.set_title('🎯 Triplet Accuracy Curves', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Triplet Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Highlight best accuracy
    best_acc_epoch = np.argmax(history['val_accuracy']) + 1
    best_val_acc = max(history['val_accuracy'])
    ax2.axvline(x=best_acc_epoch, color='purple', linestyle='--', alpha=0.7, linewidth=2)
    ax2.annotate(f'Best: Epoch {best_acc_epoch}\nVal Acc: {best_val_acc:.4f}', 
                xy=(best_acc_epoch, best_val_acc), xytext=(best_acc_epoch + len(epochs)*0.1, best_val_acc - 0.1),
                arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                fontsize=10, fontweight='bold', color='purple')
    
    # Plot 3: Learning rate schedule (if available in CSV)
    ax3 = axes[1, 0]
    if df is not None and 'learning_rate' in df.columns:
        ax3.semilogy(df['epoch'], df['learning_rate'], 'g-', linewidth=2, marker='d', markersize=4)
        ax3.set_title('📉 Learning Rate Schedule', fontsize=16, fontweight='bold', pad=15)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Highlight LR reductions
        lr_changes = df['learning_rate'].diff() < 0
        if lr_changes.any():
            change_epochs = df[lr_changes]['epoch'].values
            for change_epoch in change_epochs:
                ax3.axvline(x=change_epoch, color='orange', linestyle=':', alpha=0.7)
    else:
        ax3.text(0.5, 0.5, 'Learning Rate\nSchedule\n(No CSV data)', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=14, fontweight='bold', alpha=0.6)
        ax3.set_title('📉 Learning Rate Schedule', fontsize=16, fontweight='bold', pad=15)
    
    # Plot 4: Training summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    
    improvement_loss = history['loss'][0] - final_train_loss
    improvement_acc = final_train_acc - history['accuracy'][0]
    
    # Create summary text
    summary_text = f"""📊 TRAINING SUMMARY

🏆 Best Results:
   • Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})
   • Best Val Acc: {best_val_acc:.4f} (Epoch {best_acc_epoch})

📈 Final Metrics:
   • Train Loss: {final_train_loss:.4f}
   • Val Loss: {final_val_loss:.4f}
   • Train Acc: {final_train_acc:.4f}
   • Val Acc: {final_val_acc:.4f}

🚀 Improvements:
   • Loss: {improvement_loss:.4f}
   • Accuracy: {improvement_acc:.4f}

📊 Total Epochs: {len(history['loss'])}"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Training dashboard saved: {save_path}")


def create_small_dataset(x_train, y_train, samples_per_class=100):
    """Create a smaller balanced dataset for faster training."""
    small_x = []
    small_y = []
    
    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0]
        selected_indices = np.random.choice(
            digit_indices, 
            min(samples_per_class, len(digit_indices)), 
            replace=False
        )
        small_x.extend(x_train[selected_indices])
        small_y.extend([digit] * len(selected_indices))
    
    return np.array(small_x), np.array(small_y)


def train_fast_triplet_contrastive(config=None, verbose=1):
    """
    Train enhanced triplet-based contrastive learning with comprehensive features.
    
    Args:
        config: Configuration dictionary
        verbose: Verbosity level
        
    Returns:
        Tuple of (encoder, model, history, representations)
    """
    if config is None:
        config = {
            'encoder_type': 'simple_cnn',
            'embedding_dim': 128,
            'batch_size': 64,
            'epochs': 50,  # Increased default for extended training
            'learning_rate': 1e-3,
            'margin': 1.0,
            'samples_per_class': 100,  # For training
            'viz_samples_per_class': 25,  # For visualization
            'log_dir': None
        }
    
    # Set up logging directory
    if config['log_dir'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['log_dir'] = f"logs/enhanced_triplet_{timestamp}"
    
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['log_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose >= 1:
        print("=" * 70)
        print("🚀 MNISTMAX Enhanced Triplet Contrastive Learning")
        print("=" * 70)
        print(f"📊 Encoder type: {config['encoder_type']}")
        print(f"🔢 Embedding dim: {config['embedding_dim']}")
        print(f"📦 Batch size: {config['batch_size']}")
        print(f"🔄 Epochs: {config['epochs']}")
        print(f"📏 Margin: {config['margin']}")
        print(f"🎯 Training samples per class: {config['samples_per_class']}")
        print(f"📈 Viz samples per class: {config['viz_samples_per_class']}")
        print(f"💾 Log directory: {config['log_dir']}")
        print()
    
    # Load MNIST data
    if verbose >= 1:
        print("📥 Loading MNIST dataset...")
    
    (x_train_full, y_train_full), (x_test, y_test) = load_mnist_data(normalize=True)
    
    # Create small training dataset
    x_train, y_train = create_small_dataset(
        x_train_full, y_train_full, config['samples_per_class']
    )
    
    # Reshape for CNN
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    
    if verbose >= 1:
        print(f"   Training samples: {len(x_train):,} (reduced from {len(x_train_full):,})")
        print(f"   Test samples: {len(x_test):,}")
        print(f"   Input shape: {x_train.shape[1:]}")
        print()
    
    # Create triplet data generator
    if verbose >= 1:
        print("🔄 Creating triplet data generator...")
    
    train_generator = TripletDataGenerator(x_train, y_train, config['batch_size'])
    val_generator = TripletDataGenerator(x_test[:1000], y_test[:1000], config['batch_size'])
    
    # Create model
    if verbose >= 1:
        print("🏗️  Creating triplet model...")
    
    # Create encoder
    input_shape = x_train.shape[1:]
    inputs = keras.Input(shape=input_shape)
    
    if config['encoder_type'] == 'simple_cnn':
        x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        embeddings = keras.layers.Dense(config['embedding_dim'], name='embeddings')(x)
    
    encoder = keras.Model(inputs, embeddings, name='encoder')
    
    if verbose >= 1:
        print(f"   Encoder parameters: {encoder.count_params():,}")
        print()
    
    # Create triplet model
    anchor_input = keras.Input(shape=input_shape, name='anchor')
    positive_input = keras.Input(shape=input_shape, name='positive')
    negative_input = keras.Input(shape=input_shape, name='negative')
    
    anchor_embedding = encoder(anchor_input)
    positive_embedding = encoder(positive_input)
    negative_embedding = encoder(negative_input)
    
    triplet_model = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding],
        name='triplet_model'
    )
    
    # Create loss and optimizer
    margin = config['margin']
    
    def triplet_loss_fn(anchor, positive, negative):
        """Custom triplet loss function."""
        # Normalize embeddings
        anchor = tf.nn.l2_normalize(anchor, axis=1)
        positive = tf.nn.l2_normalize(positive, axis=1)
        negative = tf.nn.l2_normalize(negative, axis=1)
        
        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
        # Compute triplet loss
        loss = tf.maximum(0.0, pos_dist - neg_dist + margin)
        return tf.reduce_mean(loss)
    
    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    # Initialize clustering visualizer
    visualizer = FastClusteringVisualizer(
        x_test, y_test, config['log_dir'], config['viz_samples_per_class']
    )
    
    # Enhanced training loop with callbacks and accuracy tracking
    if verbose >= 1:
        print("🎯 Starting enhanced triplet contrastive training...")
        print("-" * 70)
    
    # Enhanced history tracking
    history = {
        'loss': [], 'val_loss': [], 
        'accuracy': [], 'val_accuracy': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Calculate steps per epoch
    steps_per_epoch = len(x_train) // config['batch_size']
    val_steps = min(50, len(x_test) // config['batch_size'])  # More validation steps
    
    # Setup callbacks-like behavior
    early_stopping_patience = min(20, config['epochs'] // 3)  # Adaptive patience
    lr_reduce_patience = max(7, early_stopping_patience // 3)
    lr_reduce_factor = 0.5
    min_lr = 1e-7
    current_lr = config['learning_rate']
    
    # CSV logging setup
    csv_path = os.path.join(config['log_dir'], 'training_metrics.csv')
    csv_data = []
    
    if verbose >= 1:
        print(f"📋 Training setup:")
        print(f"   🔄 Steps per epoch: {steps_per_epoch}")
        print(f"   🔍 Validation steps: {val_steps}")
        print(f"   ⏰ Early stopping patience: {early_stopping_patience}")
        print(f"   📉 LR reduce patience: {lr_reduce_patience}")
        print()
    
    for epoch in range(config['epochs']):
        if verbose >= 1:
            print(f"\n📅 Epoch {epoch + 1}/{config['epochs']} (LR: {current_lr:.2e})")
        
        # Training phase
        train_losses = []
        train_correct = 0
        train_total = 0
        train_gen = train_generator.generate_triplets()
        
        train_pbar = tqdm(range(steps_per_epoch), desc="🏋️  Training", leave=False)
        
        for batch_idx in train_pbar:
            anchor, positive, negative = next(train_gen)
            
            with tf.GradientTape() as tape:
                anchor_emb, positive_emb, negative_emb = triplet_model(
                    [anchor, positive, negative], training=True
                )
                loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            
            gradients = tape.gradient(loss, triplet_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, triplet_model.trainable_variables))
            train_losses.append(float(loss))
            
            # Calculate triplet accuracy (positive closer than negative)
            anchor_norm = tf.nn.l2_normalize(anchor_emb, axis=1)
            positive_norm = tf.nn.l2_normalize(positive_emb, axis=1)
            negative_norm = tf.nn.l2_normalize(negative_emb, axis=1)
            
            pos_dist = tf.reduce_sum(tf.square(anchor_norm - positive_norm), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor_norm - negative_norm), axis=-1)
            
            correct = tf.reduce_sum(tf.cast(pos_dist < neg_dist, tf.int32))
            train_correct += int(correct)
            train_total += config['batch_size']
            
            # Update progress bar
            current_acc = train_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix({
                'loss': f'{float(loss):.4f}',
                'acc': f'{current_acc:.3f}'
            })
        
        avg_train_loss = np.mean(train_losses)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        val_losses = []
        val_correct = 0
        val_total = 0
        val_gen = val_generator.generate_triplets()
        
        val_pbar = tqdm(range(val_steps), desc="🔍 Validation", leave=False)
        
        for batch_idx in val_pbar:
            anchor, positive, negative = next(val_gen)
            anchor_emb, positive_emb, negative_emb = triplet_model(
                [anchor, positive, negative], training=False
            )
            loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            val_losses.append(float(loss))
            
            # Calculate validation triplet accuracy
            anchor_norm = tf.nn.l2_normalize(anchor_emb, axis=1)
            positive_norm = tf.nn.l2_normalize(positive_emb, axis=1)
            negative_norm = tf.nn.l2_normalize(negative_emb, axis=1)
            
            pos_dist = tf.reduce_sum(tf.square(anchor_norm - positive_norm), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor_norm - negative_norm), axis=-1)
            
            correct = tf.reduce_sum(tf.cast(pos_dist < neg_dist, tf.int32))
            val_correct += int(correct)
            val_total += config['batch_size']
            
            # Update progress bar
            current_val_acc = val_correct / val_total if val_total > 0 else 0
            val_pbar.set_postfix({
                'loss': f'{float(loss):.4f}',
                'acc': f'{current_val_acc:.3f}'
            })
        
        avg_val_loss = np.mean(val_losses)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # Record history
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        
        # CSV logging
        csv_data.append({
            'epoch': epoch + 1,
            'loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'learning_rate': current_lr
        })
        
        # Model checkpointing and early stopping logic
        improved = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            encoder.save_weights(os.path.join(config['log_dir'], 'best_encoder.weights.h5'))
            patience_counter = 0
            improved = True
            best_indicator = "🌟 NEW BEST!"
        else:
            patience_counter += 1
            best_indicator = ""
        
        # Learning rate reduction
        if patience_counter >= lr_reduce_patience and current_lr > min_lr:
            current_lr = max(current_lr * lr_reduce_factor, min_lr)
            optimizer.learning_rate.assign(current_lr)
            if verbose >= 1:
                print(f"   📉 Reduced learning rate to {current_lr:.2e}")
        
        if verbose >= 1:
            print(f"   📊 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {best_indicator}")
            print(f"   🎯 Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
            if patience_counter > 0:
                print(f"   ⏰ Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            if verbose >= 1:
                print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                print(f"   🎯 Best validation loss: {best_val_loss:.4f}")
            break
        
        # Create clustering visualization
        if verbose >= 1:
            print("   🎨 Creating PCA visualization...")
        
        test_embeddings = encoder.predict(visualizer.test_images, verbose=0)
        visualizer.create_clustering_plot(test_embeddings, epoch + 1)
    
    # Save CSV metrics
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        if verbose >= 1:
            print(f"📊 Training metrics saved to: {csv_path}")
    
    # Load best weights
    encoder.load_weights(os.path.join(config['log_dir'], 'best_encoder.weights.h5'))
    
    if verbose >= 1:
        print(f"\n🎉 Training completed!")
        print(f"   🏆 Best validation loss: {best_val_loss:.4f}")
        print(f"   📊 Total epochs: {len(history['loss'])}")
        print("🔧 Computing fixed axis limits for smooth animation...")
    
    # Compute fixed axis limits for smooth animation
    visualizer.compute_fixed_axis_limits()
    
    # Clear existing frames and regenerate with fixed axis limits
    if verbose >= 1:
        print("🎨 Regenerating plots with fixed axis limits...")
    
    # Clear existing frames
    import glob
    existing_frames = glob.glob(os.path.join(visualizer.frames_dir, "*.png"))
    for frame in existing_frames:
        os.remove(frame)
    
    # Regenerate only the correct number of frames
    for epoch_idx, embeddings in enumerate(visualizer.all_embeddings):
        # Apply PCA transform with the fitted PCA
        reduced_2d = visualizer.pca.transform(embeddings)
        
        # Create plot with fixed axis limits
        plt.figure(figsize=(12, 9))
        
        # Plot each digit class with consistent colors
        for digit in range(10):
            mask = visualizer.test_labels == digit
            if np.any(mask):
                plt.scatter(
                    reduced_2d[mask, 0], 
                    reduced_2d[mask, 1],
                    c=[visualizer.colors[digit]], 
                    label=f'Digit {digit}',
                    alpha=0.8,
                    s=60,
                    edgecolors='white',
                    linewidth=0.8
                )
        
        # Apply fixed axis limits
        plt.xlim(visualizer.fixed_xlim)
        plt.ylim(visualizer.fixed_ylim)
        
        # Consistent styling
        plt.title(f'Epoch {epoch_idx + 1}: Learned Representations (PCA)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('First Principal Component', fontsize=14, fontweight='bold')
        plt.ylabel('Second Principal Component', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(visualizer.frames_dir, f'epoch_{epoch_idx + 1:03d}_pca.png')
        plt.savefig(frame_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Create final animation
    if verbose >= 1:
        print("🎬 Creating smooth clustering animation...")
    
    visualizer.create_animation()
    
    # Create comprehensive training metrics dashboard
    create_metrics_dashboard(
        history, 
        csv_path,
        save_path=os.path.join(config['log_dir'], 'training_dashboard.png')
    )
    
    # Generate final representations
    if verbose >= 1:
        print("💾 Generating final representations...")
    
    test_embeddings = encoder.predict(x_test, verbose=0)
    representations = {
        'embeddings': test_embeddings,
        'labels': y_test,
        'images': x_test
    }
    
    # Save representations
    save_representations(
        embeddings=test_embeddings,
        labels=y_test,
        images=x_test,
        save_path=os.path.join(config['log_dir'], 'learned_representations.npz')
    )
    
    # Linear probing evaluation
    if verbose >= 1:
        print("🧪 Evaluating with linear probing...")
    
    linear_classifier = keras.Sequential([
        encoder,
        keras.layers.Dense(10, activation='softmax')
    ])
    
    encoder.trainable = False
    linear_classifier.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    linear_history = linear_classifier.fit(
        x_train_full, y_train_full,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=256,
        verbose=1 if verbose >= 1 else 0
    )
    
    final_accuracy = max(linear_history.history['val_accuracy'])
    
    if verbose >= 1:
        print(f"\n🎯 Linear probing accuracy: {final_accuracy:.4f}")
        print("=" * 70)
        print("✅ Enhanced triplet contrastive learning completed!")
        print(f"🎬 Smooth PCA clustering animation saved")
        print(f"📊 Comprehensive training dashboard created")
        print(f"🎯 Linear probing accuracy: {final_accuracy:.4f}")
        print("=" * 70)
    
    return encoder, triplet_model, history, representations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced triplet contrastive learning')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (up to 100)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet loss margin')
    parser.add_argument('--samples_per_class', type=int, default=100, help='Training samples per class')
    parser.add_argument('--viz_samples_per_class', type=int, default=25, help='Visualization samples per class')
    
    args = parser.parse_args()
    
    config = {
        'encoder_type': 'simple_cnn',
        'embedding_dim': args.embedding_dim,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': 1e-3,
        'margin': args.margin,
        'samples_per_class': args.samples_per_class,
        'viz_samples_per_class': args.viz_samples_per_class,
        'log_dir': None
    }
    
    encoder, model, history, representations = train_fast_triplet_contrastive(config)
