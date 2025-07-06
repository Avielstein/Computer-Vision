"""
Ablation-constrained model architectures for denoising.
Models can only turn pixels off (1 → 0), never on (0 → 1).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class AblationConstraint(layers.Layer):
    """
    Layer that enforces ablation constraint: output ≤ input element-wise.
    Ensures the model can only turn pixels off, never on.
    """
    
    def __init__(self, **kwargs):
        super(AblationConstraint, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        Apply ablation constraint.
        
        Args:
            inputs: Tuple of (original_input, model_output)
            
        Returns:
            Constrained output where output ≤ input
        """
        original_input, model_output = inputs
        
        # Ensure output is never greater than input
        # This enforces the ablation constraint: can only turn pixels off
        constrained_output = tf.minimum(model_output, original_input)
        
        return constrained_output


class AblationGate(layers.Layer):
    """
    Gating mechanism that decides which pixels to ablate.
    Outputs ablation probabilities that are multiplied with input.
    """
    
    def __init__(self, **kwargs):
        super(AblationGate, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        Apply ablation gating.
        
        Args:
            inputs: Tuple of (original_input, ablation_probabilities)
            
        Returns:
            Gated output: original_input * ablation_probabilities
        """
        original_input, ablation_probs = inputs
        
        # Element-wise multiplication: keep pixels with high ablation probability
        # ablation_probs close to 1.0 = keep pixel
        # ablation_probs close to 0.0 = remove pixel
        return original_input * ablation_probs


class BasicAblationDenoiser(keras.Model):
    """
    Basic ablation denoiser with encoder-decoder architecture.
    Uses ablation gates to decide which pixels to remove.
    """
    
    def __init__(self, 
                 latent_dim: int = 64,
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(BasicAblationDenoiser, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
            layers.Dropout(dropout_rate)
        ])
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(7 * 7 * 128, activation='relu'),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')  # Ablation probabilities
        ])
        
        # Ablation gate
        self.ablation_gate = AblationGate()
    
    def call(self, inputs, training=None):
        """
        Forward pass with ablation constraint.
        
        Args:
            inputs: Input noisy images
            training: Training mode flag
            
        Returns:
            Denoised images (ablated)
        """
        # Encode input
        encoded = self.encoder(inputs, training=training)
        
        # Decode to ablation probabilities
        ablation_probs = self.decoder(encoded, training=training)
        
        # Apply ablation gate
        denoised = self.ablation_gate([inputs, ablation_probs])
        
        return denoised


class UNetAblationDenoiser(keras.Model):
    """
    U-Net architecture adapted for ablation denoising.
    Uses skip connections and ablation constraints.
    """
    
    def __init__(self, 
                 filters: list = [32, 64, 128, 256],
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(UNetAblationDenoiser, self).__init__(**kwargs)
        
        self.filters = filters
        self.dropout_rate = dropout_rate
        
        # Encoder blocks
        self.encoder_blocks = []
        for i, f in enumerate(filters):
            block = keras.Sequential([
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.Dropout(dropout_rate) if i > 0 else layers.Lambda(lambda x: x)
            ])
            self.encoder_blocks.append(block)
        
        # Pooling layers
        self.pool = layers.MaxPooling2D(2)
        
        # Bottleneck
        self.bottleneck = keras.Sequential([
            layers.Conv2D(filters[-1] * 2, 3, activation='relu', padding='same'),
            layers.Conv2D(filters[-1] * 2, 3, activation='relu', padding='same'),
            layers.Dropout(dropout_rate)
        ])
        
        # Decoder blocks
        self.decoder_blocks = []
        self.upconv_blocks = []
        
        for i, f in enumerate(reversed(filters)):
            # Upconvolution
            upconv = layers.Conv2DTranspose(f, 2, strides=2, padding='same')
            self.upconv_blocks.append(upconv)
            
            # Decoder block
            decoder = keras.Sequential([
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.Dropout(dropout_rate) if i < len(filters) - 1 else layers.Lambda(lambda x: x)
            ])
            self.decoder_blocks.append(decoder)
        
        # Final ablation probability layer
        self.final_conv = layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        
        # Ablation gate
        self.ablation_gate = AblationGate()
    
    def call(self, inputs, training=None):
        """
        U-Net forward pass with ablation constraint.
        
        Args:
            inputs: Input noisy images
            training: Training mode flag
            
        Returns:
            Denoised images (ablated)
        """
        # Store skip connections
        skip_connections = []
        x = inputs
        
        # Encoder path
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x, training=training)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            x = upconv(x)
            
            # Concatenate skip connection
            if i < len(skip_connections):
                skip = skip_connections[i]
                x = layers.Concatenate()([x, skip])
            
            x = decoder_block(x, training=training)
        
        # Generate ablation probabilities
        ablation_probs = self.final_conv(x)
        
        # Apply ablation gate
        denoised = self.ablation_gate([inputs, ablation_probs])
        
        return denoised


class AttentionAblationDenoiser(keras.Model):
    """
    Attention-based ablation denoiser.
    Uses self-attention to decide which pixels to ablate.
    """
    
    def __init__(self, 
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 ff_dim: int = 128,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super(AttentionAblationDenoiser, self).__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.input_proj = layers.Conv2D(embed_dim, 1, padding='same')
        
        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        
        # Output projection to ablation probabilities
        self.output_proj = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.Conv2D(16, 3, activation='relu', padding='same'),
            layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        ])
        
        # Ablation gate
        self.ablation_gate = AblationGate()
    
    def call(self, inputs, training=None):
        """
        Attention-based forward pass with ablation constraint.
        
        Args:
            inputs: Input noisy images
            training: Training mode flag
            
        Returns:
            Denoised images (ablated)
        """
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Project input to embedding dimension
        x = self.input_proj(inputs)  # (batch, height, width, embed_dim)
        
        # Reshape for attention: (batch, seq_len, embed_dim)
        x_flat = tf.reshape(x, [batch_size, height * width, self.embed_dim])
        
        # Self-attention
        attn_output = self.attention(x_flat, x_flat, training=training)
        x_flat = self.layernorm1(x_flat + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x_flat, training=training)
        x_flat = self.layernorm2(x_flat + ffn_output)
        
        # Reshape back to spatial dimensions
        x = tf.reshape(x_flat, [batch_size, height, width, self.embed_dim])
        
        # Generate ablation probabilities
        ablation_probs = self.output_proj(x, training=training)
        
        # Apply ablation gate
        denoised = self.ablation_gate([inputs, ablation_probs])
        
        return denoised


class ResidualAblationDenoiser(keras.Model):
    """
    Residual network adapted for ablation denoising.
    Uses residual blocks with ablation constraints.
    """
    
    def __init__(self, 
                 num_blocks: int = 6,
                 filters: int = 64,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super(ResidualAblationDenoiser, self).__init__(**kwargs)
        
        self.num_blocks = num_blocks
        self.filters = filters
        self.dropout_rate = dropout_rate
        
        # Initial convolution
        self.initial_conv = layers.Conv2D(filters, 3, activation='relu', padding='same')
        
        # Residual blocks
        self.residual_blocks = []
        for _ in range(num_blocks):
            block = self._create_residual_block(filters, dropout_rate)
            self.residual_blocks.append(block)
        
        # Final layers for ablation probabilities
        self.final_layers = keras.Sequential([
            layers.Conv2D(filters // 2, 3, activation='relu', padding='same'),
            layers.Conv2D(filters // 4, 3, activation='relu', padding='same'),
            layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        ])
        
        # Ablation gate
        self.ablation_gate = AblationGate()
    
    def _create_residual_block(self, filters: int, dropout_rate: float):
        """Create a residual block."""
        def residual_block(x, training=None):
            shortcut = x
            
            # First convolution
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x, training=training)
            x = layers.ReLU()(x)
            x = layers.Dropout(dropout_rate)(x, training=training)
            
            # Second convolution
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x, training=training)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)
            
            return x
        
        return residual_block
    
    def call(self, inputs, training=None):
        """
        Residual network forward pass with ablation constraint.
        
        Args:
            inputs: Input noisy images
            training: Training mode flag
            
        Returns:
            Denoised images (ablated)
        """
        # Initial convolution
        x = self.initial_conv(inputs)
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x, training=training)
        
        # Generate ablation probabilities
        ablation_probs = self.final_layers(x, training=training)
        
        # Apply ablation gate
        denoised = self.ablation_gate([inputs, ablation_probs])
        
        return denoised


class VariationalAblationDenoiser(keras.Model):
    """
    Variational autoencoder adapted for ablation denoising.
    Uses probabilistic latent space for ablation decisions.
    """
    
    def __init__(self, 
                 latent_dim: int = 32,
                 beta: float = 1.0,
                 **kwargs):
        super(VariationalAblationDenoiser, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu')
        ])
        
        # Latent space
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(7 * 7 * 64, activation='relu'),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ])
        
        # Ablation gate
        self.ablation_gate = AblationGate()
    
    def encode(self, x, training=None):
        """Encode input to latent parameters."""
        h = self.encoder(x, training=training)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar, training=None):
        """Reparameterization trick."""
        if training:
            eps = tf.random.normal(tf.shape(mu))
            return mu + tf.exp(0.5 * logvar) * eps
        else:
            return mu
    
    def decode(self, z, training=None):
        """Decode latent vector to ablation probabilities."""
        return self.decoder(z, training=training)
    
    def call(self, inputs, training=None):
        """
        VAE forward pass with ablation constraint.
        
        Args:
            inputs: Input noisy images
            training: Training mode flag
            
        Returns:
            Denoised images (ablated) and KL divergence loss
        """
        # Encode
        mu, logvar = self.encode(inputs, training=training)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar, training=training)
        
        # Decode to ablation probabilities
        ablation_probs = self.decode(z, training=training)
        
        # Apply ablation gate
        denoised = self.ablation_gate([inputs, ablation_probs])
        
        # Compute KL divergence loss
        if training:
            kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            kl_loss = tf.reduce_mean(kl_loss)
            self.add_loss(self.beta * kl_loss)
        
        return denoised


def create_ablation_model(model_type: str = "basic", **kwargs) -> keras.Model:
    """
    Create an ablation denoising model.
    
    Args:
        model_type: Type of model architecture
        **kwargs: Additional model parameters
        
    Returns:
        Ablation denoising model
    """
    if model_type == "basic":
        return BasicAblationDenoiser(**kwargs)
    elif model_type == "unet":
        return UNetAblationDenoiser(**kwargs)
    elif model_type == "attention":
        return AttentionAblationDenoiser(**kwargs)
    elif model_type == "residual":
        return ResidualAblationDenoiser(**kwargs)
    elif model_type == "vae":
        return VariationalAblationDenoiser(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_ablation_constraint():
    """Test that models enforce ablation constraint."""
    print("Testing Ablation Constraint...")
    
    # Create test data
    batch_size = 4
    test_input = tf.random.uniform((batch_size, 28, 28, 1), 0, 1)
    
    # Test each model type
    model_types = ["basic", "unet", "attention", "residual", "vae"]
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model:")
        
        try:
            model = create_ablation_model(model_type)
            output = model(test_input, training=False)
            
            # Check ablation constraint: output <= input
            constraint_satisfied = tf.reduce_all(output <= test_input + 1e-6)  # Small epsilon for numerical precision
            
            print(f"  Input range: [{tf.reduce_min(test_input):.3f}, {tf.reduce_max(test_input):.3f}]")
            print(f"  Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
            print(f"  Ablation constraint satisfied: {constraint_satisfied}")
            print(f"  Model parameters: {model.count_params():,}")
            
        except Exception as e:
            print(f"  Error testing {model_type}: {e}")


if __name__ == "__main__":
    # Test ablation models
    test_ablation_constraint()
    
    # Demo usage
    print("\n" + "="*50)
    print("Ablation Model Demo")
    print("="*50)
    
    # Create sample noisy input
    batch_size = 2
    noisy_input = tf.random.uniform((batch_size, 28, 28, 1), 0, 1)
    
    # Test basic ablation model
    model = create_ablation_model("basic", latent_dim=32)
    
    print(f"\nBasic Ablation Model:")
    print(f"Input shape: {noisy_input.shape}")
    
    # Forward pass
    denoised_output = model(noisy_input, training=False)
    
    print(f"Output shape: {denoised_output.shape}")
    print(f"Input pixel density: {tf.reduce_mean(noisy_input):.3f}")
    print(f"Output pixel density: {tf.reduce_mean(denoised_output):.3f}")
    print(f"Pixels removed: {tf.reduce_mean(noisy_input - denoised_output):.3f}")
    
    # Verify constraint
    constraint_check = tf.reduce_all(denoised_output <= noisy_input + 1e-6)
    print(f"Ablation constraint satisfied: {constraint_check}")
