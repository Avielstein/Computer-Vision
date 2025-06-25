"""
Contrastive loss functions for self-supervised learning.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


class NTXentLoss(tf.keras.losses.Loss):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used in SimCLR and other contrastive learning methods.
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 name: str = "ntxent_loss",
                 **kwargs):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for scaling
            name: Name of the loss function
        """
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
    
    def call(self, z_i: tf.Tensor, z_j: tf.Tensor) -> tf.Tensor:
        """
        Compute NT-Xent loss between two sets of representations.
        
        Args:
            z_i: First set of representations, shape (batch_size, embedding_dim)
            z_j: Second set of representations, shape (batch_size, embedding_dim)
            
        Returns:
            Scalar loss value
        """
        batch_size = tf.shape(z_i)[0]
        
        # Normalize representations
        z_i = tf.nn.l2_normalize(z_i, axis=1)
        z_j = tf.nn.l2_normalize(z_j, axis=1)
        
        # Concatenate representations
        representations = tf.concat([z_i, z_j], axis=0)  # Shape: (2*batch_size, embedding_dim)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(representations, representations, transpose_b=True)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels for positive pairs
        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = tf.range(2 * batch_size)
        labels = tf.where(labels < batch_size, labels + batch_size, labels - batch_size)
        
        # Mask to remove self-similarity (diagonal elements)
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        similarity_matrix = tf.where(mask, -np.inf, similarity_matrix)
        
        # Compute cross-entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=similarity_matrix
        )
        
        return tf.reduce_mean(loss)


class SupConLoss(tf.keras.losses.Loss):
    """
    Supervised Contrastive Loss.
    Extends contrastive learning to supervised settings.
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 base_temperature: float = 0.07,
                 name: str = "supcon_loss",
                 **kwargs):
        """
        Initialize Supervised Contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling
            base_temperature: Base temperature for normalization
            name: Name of the loss function
        """
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def call(self, features: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature representations, shape (batch_size, embedding_dim)
            labels: Ground truth labels, shape (batch_size,)
            
        Returns:
            Scalar loss value
        """
        batch_size = tf.shape(features)[0]
        
        # Normalize features
        features = tf.nn.l2_normalize(features, axis=1)
        
        # Compute similarity matrix
        anchor_dot_contrast = tf.matmul(features, features, transpose_b=True) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = tf.expand_dims(labels, 1)
        mask = tf.equal(labels, tf.transpose(labels))
        mask = tf.cast(mask, tf.float32)
        
        # Mask out self-contrast cases
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
        mask = mask * logits_mask
        
        # For numerical stability
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max
        
        # Compute log probabilities
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        
        # Compute mean of log-likelihood over positive pairs
        # Add small epsilon to avoid division by zero
        mask_sum = tf.reduce_sum(mask, axis=1)
        mask_sum = tf.maximum(mask_sum, 1e-8)  # Avoid division by zero
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / mask_sum
        
        # Only compute loss for samples that have positive pairs
        valid_samples = mask_sum > 0
        mean_log_prob_pos = tf.where(valid_samples, mean_log_prob_pos, 0.0)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(tf.cast(valid_samples, tf.float32)), 1.0)
        
        return loss


class InfoNCELoss(tf.keras.losses.Loss):
    """
    InfoNCE Loss for contrastive learning.
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 name: str = "infonce_loss",
                 **kwargs):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling
            name: Name of the loss function
        """
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
    
    def call(self, query: tf.Tensor, positive: tf.Tensor, negatives: tf.Tensor) -> tf.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query representations, shape (batch_size, embedding_dim)
            positive: Positive representations, shape (batch_size, embedding_dim)
            negatives: Negative representations, shape (batch_size, num_negatives, embedding_dim)
            
        Returns:
            Scalar loss value
        """
        # Normalize representations
        query = tf.nn.l2_normalize(query, axis=-1)
        positive = tf.nn.l2_normalize(positive, axis=-1)
        negatives = tf.nn.l2_normalize(negatives, axis=-1)
        
        # Compute positive similarities
        pos_sim = tf.reduce_sum(query * positive, axis=-1) / self.temperature  # (batch_size,)
        
        # Compute negative similarities
        neg_sim = tf.reduce_sum(
            tf.expand_dims(query, 1) * negatives, axis=-1
        ) / self.temperature  # (batch_size, num_negatives)
        
        # Concatenate positive and negative similarities
        logits = tf.concat([tf.expand_dims(pos_sim, 1), neg_sim], axis=1)
        
        # Labels: positive is always at index 0
        labels = tf.zeros(tf.shape(query)[0], dtype=tf.int32)
        
        # Compute cross-entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        
        return tf.reduce_mean(loss)


class TripletLoss(tf.keras.losses.Loss):
    """
    Triplet Loss for contrastive learning.
    """
    
    def __init__(self, 
                 margin: float = 1.0,
                 name: str = "triplet_loss",
                 **kwargs):
        """
        Initialize Triplet loss.
        
        Args:
            margin: Margin for triplet loss
            name: Name of the loss function
        """
        super().__init__(name=name, **kwargs)
        self.margin = margin
    
    def call(self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor) -> tf.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor representations, shape (batch_size, embedding_dim)
            positive: Positive representations, shape (batch_size, embedding_dim)
            negative: Negative representations, shape (batch_size, embedding_dim)
            
        Returns:
            Scalar loss value
        """
        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
        # Compute triplet loss
        loss = tf.maximum(0.0, pos_dist - neg_dist + self.margin)
        
        return tf.reduce_mean(loss)


def cosine_similarity_loss(z_i: tf.Tensor, z_j: tf.Tensor) -> tf.Tensor:
    """
    Simple cosine similarity loss for contrastive learning.
    
    Args:
        z_i: First set of representations
        z_j: Second set of representations
        
    Returns:
        Scalar loss value (negative cosine similarity)
    """
    # Normalize representations
    z_i = tf.nn.l2_normalize(z_i, axis=1)
    z_j = tf.nn.l2_normalize(z_j, axis=1)
    
    # Compute cosine similarity
    cosine_sim = tf.reduce_sum(z_i * z_j, axis=1)
    
    # Return negative similarity (we want to maximize similarity)
    return -tf.reduce_mean(cosine_sim)


def create_contrastive_loss(loss_type: str = "ntxent", **kwargs) -> tf.keras.losses.Loss:
    """
    Factory function to create contrastive loss functions.
    
    Args:
        loss_type: Type of loss ("ntxent", "supcon", "infonce", "triplet")
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Configured loss function
    """
    if loss_type.lower() == "ntxent":
        return NTXentLoss(**kwargs)
    elif loss_type.lower() == "supcon":
        return SupConLoss(**kwargs)
    elif loss_type.lower() == "infonce":
        return InfoNCELoss(**kwargs)
    elif loss_type.lower() == "triplet":
        return TripletLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Utility functions for mining hard negatives
def hard_negative_mining(embeddings: tf.Tensor, 
                        labels: tf.Tensor, 
                        num_negatives: int = 10) -> tf.Tensor:
    """
    Mine hard negatives for contrastive learning.
    
    Args:
        embeddings: Feature embeddings, shape (batch_size, embedding_dim)
        labels: Ground truth labels, shape (batch_size,)
        num_negatives: Number of hard negatives to mine
        
    Returns:
        Indices of hard negatives, shape (batch_size, num_negatives)
    """
    batch_size = tf.shape(embeddings)[0]
    
    # Compute pairwise distances
    distances = tf.linalg.norm(
        tf.expand_dims(embeddings, 1) - tf.expand_dims(embeddings, 0),
        axis=-1
    )
    
    # Create mask for different classes
    labels_equal = tf.equal(
        tf.expand_dims(labels, 1), 
        tf.expand_dims(labels, 0)
    )
    negative_mask = tf.logical_not(labels_equal)
    
    # Set distances of same class to infinity
    distances = tf.where(negative_mask, distances, tf.float32.max)
    
    # Get indices of smallest distances (hardest negatives)
    _, hard_negative_indices = tf.nn.top_k(-distances, k=num_negatives)
    
    return hard_negative_indices
