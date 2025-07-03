"""
Neural network architectures for contrastive learning on MNIST.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, List


class ResNetBlock(layers.Layer):
    """Residual block for ResNet architecture."""
    
    def __init__(self, filters: int, kernel_size: int = 3, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = layers.Conv2D(filters, kernel_size, stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, 1, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection
        if stride != 1:
            self.shortcut = layers.Conv2D(filters, 1, stride, padding='same')
            self.shortcut_bn = layers.BatchNormalization()
        else:
            self.shortcut = None
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
        
        x = x + shortcut
        return tf.nn.relu(x)


class SimpleCNNEncoder(keras.Model):
    """Simple CNN encoder for MNIST."""
    
    def __init__(self, embedding_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        
        self.conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D(2)
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D(2)
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D(2)
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(embedding_dim)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        return x


class ResNetEncoder(keras.Model):
    """ResNet-based encoder for MNIST."""
    
    def __init__(self, embedding_dim: int = 128, num_blocks: List[int] = [2, 2, 2], **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        
        # Initial convolution
        self.conv1 = layers.Conv2D(64, 7, 2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(3, 2, padding='same')
        
        # Residual blocks
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        
        # Final layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(embedding_dim)
    
    def _make_layer(self, filters: int, num_blocks: int, stride: int):
        """Create a layer with multiple residual blocks."""
        blocks = []
        blocks.append(ResNetBlock(filters, stride=stride))
        for _ in range(1, num_blocks):
            blocks.append(ResNetBlock(filters, stride=1))
        return keras.Sequential(blocks)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        
        x = self.global_pool(x)
        x = self.fc(x)
        
        return x


class VisionTransformerEncoder(keras.Model):
    """Vision Transformer encoder for MNIST."""
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 patch_size: int = 4,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 mlp_dim: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        
        # Calculate number of patches
        self.num_patches = (28 // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = layers.Conv2D(embedding_dim, patch_size, patch_size)
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='pos_embed'
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            self._create_transformer_block() for _ in range(num_layers)
        ]
        
        # Final layer norm and projection
        self.norm = layers.LayerNormalization()
        self.head = layers.Dense(embedding_dim)
    
    def _create_transformer_block(self):
        """Create a transformer block."""
        return keras.Sequential([
            layers.LayerNormalization(),
            layers.MultiHeadAttention(self.num_heads, self.embedding_dim // self.num_heads),
            layers.LayerNormalization(),
            layers.Dense(self.mlp_dim, activation='gelu'),
            layers.Dense(self.embedding_dim),
        ])
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Patch embedding
        x = self.patch_embed(inputs)  # (batch_size, num_patches_h, num_patches_w, embedding_dim)
        x = tf.reshape(x, (batch_size, self.num_patches, self.embedding_dim))
        
        # Add class token
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        x = tf.concat([cls_tokens, x], axis=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            # Multi-head attention
            attn_output = block.layers[1](
                block.layers[0](x), 
                block.layers[0](x),
                training=training
            )
            x = x + attn_output
            
            # MLP
            mlp_output = block.layers[3](block.layers[2](x))
            mlp_output = block.layers[4](mlp_output)
            x = x + mlp_output
        
        # Use class token for final representation
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x


class ProjectionHead(keras.Model):
    """Projection head for contrastive learning."""
    
    def __init__(self, 
                 projection_dim: int = 64,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layers_list = []
        for i in range(num_layers - 1):
            self.layers_list.append(layers.Dense(hidden_dim, activation='relu'))
            self.layers_list.append(layers.BatchNormalization())
        
        # Final projection layer (no activation)
        self.layers_list.append(layers.Dense(projection_dim))
        
        self.projection = keras.Sequential(self.layers_list)
    
    def call(self, inputs, training=None):
        return self.projection(inputs, training=training)


class ContrastiveModel(keras.Model):
    """Complete contrastive learning model with encoder and projection head."""
    
    def __init__(self, 
                 encoder_type: str = "simple_cnn",
                 embedding_dim: int = 128,
                 projection_dim: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        # Create encoder
        if encoder_type == "simple_cnn":
            self.encoder = SimpleCNNEncoder(embedding_dim)
        elif encoder_type == "resnet":
            self.encoder = ResNetEncoder(embedding_dim)
        elif encoder_type == "vit":
            self.encoder = VisionTransformerEncoder(embedding_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Create projection head
        self.projection_head = ProjectionHead(projection_dim)
    
    def call(self, inputs, training=None):
        # Get embeddings from encoder
        embeddings = self.encoder(inputs, training=training)
        
        # Project to contrastive space
        projections = self.projection_head(embeddings, training=training)
        
        return embeddings, projections
    
    def get_embeddings(self, inputs, training=None):
        """Get only the embeddings (for downstream tasks)."""
        return self.encoder(inputs, training=training)
    
    def get_projections(self, inputs, training=None):
        """Get only the projections (for contrastive loss)."""
        embeddings = self.encoder(inputs, training=training)
        return self.projection_head(embeddings, training=training)


def create_contrastive_model(encoder_type: str = "simple_cnn",
                           embedding_dim: int = 128,
                           projection_dim: int = 64,
                           input_shape: Tuple[int, int, int] = (28, 28, 1)) -> ContrastiveModel:
    """
    Factory function to create contrastive learning models.
    
    Args:
        encoder_type: Type of encoder ("simple_cnn", "resnet", "vit")
        embedding_dim: Dimension of the embedding space
        projection_dim: Dimension of the projection space
        input_shape: Shape of input images
        
    Returns:
        Configured contrastive model
    """
    model = ContrastiveModel(
        encoder_type=encoder_type,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim
    )
    
    # Build the model by calling it with dummy input
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    
    return model


class LinearClassifier(keras.Model):
    """Linear classifier for evaluating learned representations."""
    
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        return self.classifier(inputs, training=training)


def create_downstream_model(encoder: keras.Model, 
                          num_classes: int = 10,
                          freeze_encoder: bool = True) -> keras.Model:
    """
    Create a downstream classification model using pretrained encoder.
    
    Args:
        encoder: Pretrained encoder model
        num_classes: Number of classes for classification
        freeze_encoder: Whether to freeze encoder weights
        
    Returns:
        Complete model for downstream classification
    """
    if freeze_encoder:
        encoder.trainable = False
    
    inputs = keras.Input(shape=(28, 28, 1))
    embeddings = encoder(inputs)
    outputs = layers.Dense(num_classes, activation='softmax')(embeddings)
    
    model = keras.Model(inputs, outputs)
    return model
