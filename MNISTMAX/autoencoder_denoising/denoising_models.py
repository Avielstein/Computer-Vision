"""
Denoising autoencoder models for binary image restoration.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, List
import numpy as np


class DenoisingAutoencoder(keras.Model):
    """Basic denoising autoencoder for binary images."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 latent_dim: int = 64,
                 **kwargs):
        """
        Initialize denoising autoencoder.
        
        Args:
            input_shape: Shape of input images
            latent_dim: Dimension of latent space
        """
        super().__init__(**kwargs)
        self.input_shape_val = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ], name='encoder')
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(7 * 7 * 128, activation='relu'),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(128, 3, activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2DTranspose(64, 3, activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2DTranspose(32, 3, activation='relu', padding='same'),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ], name='decoder')
    
    def call(self, inputs, training=None):
        """Forward pass."""
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def encode(self, inputs, training=None):
        """Encode inputs to latent space."""
        return self.encoder(inputs, training=training)
    
    def decode(self, latent, training=None):
        """Decode from latent space."""
        return self.decoder(latent, training=training)


class UNetDenoiser(keras.Model):
    """U-Net style denoising autoencoder with skip connections."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 filters: List[int] = [32, 64, 128],  # Reduced depth for 28x28
                 **kwargs):
        """
        Initialize U-Net denoiser.
        
        Args:
            input_shape: Shape of input images
            filters: Number of filters for each level
        """
        super().__init__(**kwargs)
        self.input_shape_val = input_shape
        self.filters = filters
        
        # Encoder (downsampling path)
        self.encoder_blocks = []
        for i, f in enumerate(filters):
            block = keras.Sequential([
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.BatchNormalization()
            ], name=f'encoder_block_{i}')
            self.encoder_blocks.append(block)
        
        self.downsample_layers = []
        for i in range(len(filters) - 1):
            self.downsample_layers.append(layers.MaxPooling2D(2, name=f'downsample_{i}'))
        
        # Decoder (upsampling path)
        self.decoder_blocks = []
        self.upsample_layers = []
        
        for i, f in enumerate(reversed(filters[:-1])):
            # Upsampling layer
            upsample = layers.UpSampling2D(2, name=f'upsample_{i}')
            self.upsample_layers.append(upsample)
            
            # Decoder block
            block = keras.Sequential([
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.Conv2D(f, 3, activation='relu', padding='same'),
                layers.BatchNormalization()
            ], name=f'decoder_block_{i}')
            self.decoder_blocks.append(block)
        
        # Final output layer
        self.output_layer = layers.Conv2D(1, 1, activation='sigmoid', padding='same', 
                                        name='output')
    
    def call(self, inputs, training=None):
        """Forward pass with skip connections."""
        # Encoder path
        skip_connections = []
        x = inputs
        
        for i, encoder_block in enumerate(self.encoder_blocks[:-1]):
            x = encoder_block(x, training=training)
            skip_connections.append(x)
            x = self.downsample_layers[i](x)
        
        # Bottleneck
        x = self.encoder_blocks[-1](x, training=training)
        
        # Decoder path
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_layers, self.decoder_blocks)):
            x = upsample(x, training=training)
            
            # Skip connection with size matching
            skip = skip_connections[-(i+1)]
            
            # Ensure skip connection matches upsampled size
            if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
                # Resize skip connection to match upsampled tensor
                skip = tf.image.resize(skip, [x.shape[1], x.shape[2]], method='nearest')
            
            x = layers.Concatenate()([x, skip])
            
            x = decoder_block(x, training=training)
        
        # Output
        output = self.output_layer(x, training=training)
        return output


class ResidualDenoisingAutoencoder(keras.Model):
    """Denoising autoencoder with residual connections."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 num_residual_blocks: int = 4,
                 filters: int = 64,
                 **kwargs):
        """
        Initialize residual denoising autoencoder.
        
        Args:
            input_shape: Shape of input images
            num_residual_blocks: Number of residual blocks
            filters: Number of filters in residual blocks
        """
        super().__init__(**kwargs)
        self.input_shape_val = input_shape
        self.num_residual_blocks = num_residual_blocks
        self.filters = filters
        
        # Initial convolution
        self.initial_conv = layers.Conv2D(filters, 7, padding='same', activation='relu')
        
        # Residual blocks
        self.residual_blocks = []
        for i in range(num_residual_blocks):
            block = self._create_residual_block(filters, f'residual_block_{i}')
            self.residual_blocks.append(block)
        
        # Final convolution
        self.final_conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
    
    def _create_residual_block(self, filters: int, name: str):
        """Create a residual block."""
        def residual_block(x, training=None):
            shortcut = x
            
            # First conv
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x, training=training)
            x = layers.ReLU()(x)
            
            # Second conv
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x, training=training)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)
            
            return x
        
        return residual_block
    
    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.initial_conv(inputs, training=training)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        # Final output
        output = self.final_conv(x, training=training)
        return output


class AttentionDenoisingAutoencoder(keras.Model):
    """Denoising autoencoder with attention mechanism."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 filters: int = 64,
                 **kwargs):
        """
        Initialize attention-based denoising autoencoder.
        
        Args:
            input_shape: Shape of input images
            filters: Number of filters
        """
        super().__init__(**kwargs)
        self.input_shape_val = input_shape
        self.filters = filters
        
        # Feature extraction
        self.feature_extractor = keras.Sequential([
            layers.Conv2D(filters, 3, activation='relu', padding='same'),
            layers.Conv2D(filters, 3, activation='relu', padding='same'),
            layers.BatchNormalization()
        ])
        
        # Attention mechanism
        self.attention_conv = layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        
        # Reconstruction
        self.reconstructor = keras.Sequential([
            layers.Conv2D(filters, 3, activation='relu', padding='same'),
            layers.Conv2D(filters, 3, activation='relu', padding='same'),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ])
    
    def call(self, inputs, training=None):
        """Forward pass with attention."""
        # Extract features
        features = self.feature_extractor(inputs, training=training)
        
        # Compute attention weights
        attention_weights = self.attention_conv(features, training=training)
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Reconstruct
        output = self.reconstructor(attended_features, training=training)
        
        return output


class VariationalDenoisingAutoencoder(keras.Model):
    """Variational denoising autoencoder for probabilistic denoising."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 latent_dim: int = 64,
                 **kwargs):
        """
        Initialize variational denoising autoencoder.
        
        Args:
            input_shape: Shape of input images
            latent_dim: Dimension of latent space
        """
        super().__init__(**kwargs)
        self.input_shape_val = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.Flatten()
        ])
        
        # Latent space
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(7 * 7 * 128, activation='relu'),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(128, 3, activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2DTranspose(64, 3, activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2DTranspose(32, 3, activation='relu', padding='same'),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ])
    
    def sampling(self, args):
        """Reparameterization trick."""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs, training=None):
        """Forward pass."""
        # Encode
        encoded = self.encoder_conv(inputs, training=training)
        z_mean = self.z_mean(encoded)
        z_log_var = self.z_log_var(encoded)
        
        # Sample
        z = self.sampling([z_mean, z_log_var])
        
        # Decode
        reconstructed = self.decoder(z, training=training)
        
        # Add KL loss
        if training:
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            self.add_loss(kl_loss * 0.001)  # Weight the KL loss
        
        return reconstructed


def create_denoising_model(model_type: str = "unet",
                          input_shape: Tuple[int, int, int] = (28, 28, 1),
                          **kwargs) -> keras.Model:
    """
    Factory function to create denoising models.
    
    Args:
        model_type: Type of model ("basic", "unet", "residual", "attention", "vae")
        input_shape: Shape of input images
        **kwargs: Additional arguments for specific models
        
    Returns:
        Configured denoising model
    """
    if model_type.lower() == "basic":
        model = DenoisingAutoencoder(input_shape=input_shape, **kwargs)
    elif model_type.lower() == "unet":
        model = UNetDenoiser(input_shape=input_shape, **kwargs)
    elif model_type.lower() == "residual":
        model = ResidualDenoisingAutoencoder(input_shape=input_shape, **kwargs)
    elif model_type.lower() == "attention":
        model = AttentionDenoisingAutoencoder(input_shape=input_shape, **kwargs)
    elif model_type.lower() == "vae":
        model = VariationalDenoisingAutoencoder(input_shape=input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Build the model
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    
    return model


class DenoisingLoss:
    """Custom loss functions for denoising."""
    
    @staticmethod
    def binary_crossentropy_loss(y_true, y_pred):
        """Binary cross-entropy loss for binary images."""
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    
    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1e-6):
        """Dice loss for binary segmentation."""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    @staticmethod
    def combined_loss(y_true, y_pred, alpha=0.5):
        """Combined binary cross-entropy and dice loss."""
        bce = DenoisingLoss.binary_crossentropy_loss(y_true, y_pred)
        dice = DenoisingLoss.dice_loss(y_true, y_pred)
        return alpha * bce + (1 - alpha) * dice
    
    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance."""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Compute focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.log(p_t)
        
        return tf.reduce_mean(focal_loss)
    
    @staticmethod
    def perceptual_loss(y_true, y_pred, feature_extractor=None):
        """Perceptual loss using feature differences."""
        if feature_extractor is None:
            # Simple gradient-based perceptual loss
            true_grad_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
            pred_grad_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
            
            true_grad_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
            pred_grad_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
            
            grad_loss_x = tf.reduce_mean(tf.abs(true_grad_x - pred_grad_x))
            grad_loss_y = tf.reduce_mean(tf.abs(true_grad_y - pred_grad_y))
            
            return grad_loss_x + grad_loss_y
        else:
            # Use provided feature extractor
            true_features = feature_extractor(y_true)
            pred_features = feature_extractor(y_pred)
            return tf.reduce_mean(tf.abs(true_features - pred_features))


def get_denoising_loss(loss_type: str = "binary_crossentropy") -> callable:
    """
    Get loss function for denoising.
    
    Args:
        loss_type: Type of loss function
        
    Returns:
        Loss function
    """
    loss_functions = {
        'binary_crossentropy': DenoisingLoss.binary_crossentropy_loss,
        'dice': DenoisingLoss.dice_loss,
        'combined': DenoisingLoss.combined_loss,
        'focal': DenoisingLoss.focal_loss,
        'perceptual': DenoisingLoss.perceptual_loss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_functions[loss_type]


# Metrics for evaluation
class DenoisingMetrics:
    """Custom metrics for denoising evaluation."""
    
    @staticmethod
    def pixel_accuracy(y_true, y_pred):
        """Pixel-wise accuracy for binary images."""
        y_true_binary = tf.cast(y_true > 0.5, tf.float32)
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_true_binary, y_pred_binary), tf.float32))
    
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Dice coefficient for binary images."""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    @staticmethod
    def iou_score(y_true, y_pred, smooth=1e-6):
        """Intersection over Union score."""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)


def get_denoising_metrics() -> List[callable]:
    """Get list of metrics for denoising evaluation."""
    return [
        DenoisingMetrics.pixel_accuracy,
        DenoisingMetrics.dice_coefficient,
        DenoisingMetrics.iou_score
    ]
