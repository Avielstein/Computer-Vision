"""
Precision-focused loss functions and metrics for ablation denoising.
Only evaluates on pixels that should be "on" in the target.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Callable


class AblationLoss:
    """
    Loss functions specifically designed for ablation denoising.
    
    Core principle: Only evaluate loss on pixels that should be "on" in the target,
    since we can only turn pixels off, never on.
    """
    
    @staticmethod
    def precision_focused_bce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Binary cross-entropy loss focused on precision.
        Evaluates loss on pixels that should be "on" AND penalizes false positives.
        
        Args:
            y_true: Target binary images (clean)
            y_pred: Predicted binary images (denoised)
            
        Returns:
            Precision-focused BCE loss
        """
        # Standard BCE on all pixels
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Additional penalty for false positives (keeping noise pixels)
        false_positive_mask = tf.logical_and(tf.equal(y_true, 0.0), tf.greater(y_pred, 0.5))
        fp_penalty = tf.reduce_mean(tf.cast(false_positive_mask, tf.float32)) * 2.0
        
        # Additional penalty for being too conservative (not removing enough)
        # Encourage the model to be more aggressive in noise removal
        conservation_penalty = tf.reduce_mean(y_pred) * 0.5
        
        return tf.reduce_mean(bce_loss) + fp_penalty + conservation_penalty
    
    @staticmethod
    def weighted_precision_bce(y_true: tf.Tensor, y_pred: tf.Tensor, 
                              false_positive_weight: float = 2.0) -> tf.Tensor:
        """
        Weighted BCE that penalizes false positives more heavily.
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            false_positive_weight: Weight for false positive penalty
            
        Returns:
            Weighted BCE loss
        """
        # Standard BCE
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Additional penalty for false positives (predicting 1 when true is 0)
        false_positive_mask = tf.logical_and(tf.equal(y_true, 0.0), tf.greater(y_pred, 0.5))
        false_positive_penalty = tf.cast(false_positive_mask, tf.float32) * false_positive_weight
        
        # Combine losses
        weighted_loss = bce + false_positive_penalty
        return tf.reduce_mean(weighted_loss)
    
    @staticmethod
    def ablation_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, 
                          smooth: float = 1e-6) -> tf.Tensor:
        """
        Dice loss adapted for ablation denoising.
        Focuses on overlap between predicted and true "on" pixels.
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Ablation-adapted Dice loss
        """
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Compute intersection (true positives)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        
        # Compute union (focus on predicted positives and true positives)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        # Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice
    
    @staticmethod
    def false_positive_penalty(y_true: tf.Tensor, y_pred: tf.Tensor,
                              penalty_weight: float = 1.0) -> tf.Tensor:
        """
        Direct penalty for false positives (pixels that should be off but are predicted on).
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            penalty_weight: Weight for the penalty
            
        Returns:
            False positive penalty loss
        """
        # Identify false positives: predicted on (>0.5) but should be off (=0)
        false_positives = tf.logical_and(
            tf.equal(y_true, 0.0),
            tf.greater(y_pred, 0.5)
        )
        
        # Count false positives per sample
        fp_count = tf.reduce_sum(tf.cast(false_positives, tf.float32), axis=[1, 2, 3])
        
        # Normalize by image size and apply penalty weight
        image_size = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
        normalized_fp = fp_count / image_size
        
        return penalty_weight * tf.reduce_mean(normalized_fp)
    
    @staticmethod
    def combined_ablation_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
                              bce_weight: float = 1.0,
                              dice_weight: float = 0.8,
                              fp_weight: float = 1.0,
                              recall_weight: float = 0.5) -> tf.Tensor:
        """
        Enhanced combined loss function for ablation denoising.
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            bce_weight: Weight for precision-focused BCE
            dice_weight: Weight for Dice loss
            fp_weight: Weight for false positive penalty
            recall_weight: Weight for recall penalty (encourages noise removal)
            
        Returns:
            Combined ablation loss
        """
        bce_loss = AblationLoss.precision_focused_bce(y_true, y_pred)
        dice_loss = AblationLoss.ablation_dice_loss(y_true, y_pred)
        fp_loss = AblationLoss.false_positive_penalty(y_true, y_pred)
        
        # Add recall penalty to encourage more aggressive denoising
        # Penalize when model keeps too many pixels (high recall but poor denoising)
        predicted_density = tf.reduce_mean(y_pred)
        target_density = tf.reduce_mean(y_true)
        recall_penalty = tf.maximum(0.0, predicted_density - target_density * 1.2) * recall_weight
        
        total_loss = (bce_weight * bce_loss + 
                     dice_weight * dice_loss + 
                     fp_weight * fp_loss +
                     recall_penalty)
        
        return total_loss


class AblationMetrics:
    """
    Metrics specifically designed for evaluating ablation denoising performance.
    """
    
    @staticmethod
    def precision_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Precision: Of the pixels we predicted as "on", how many should actually be "on"?
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            
        Returns:
            Precision score
        """
        # Convert predictions to binary
        y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        
        # True positives: predicted on AND should be on
        true_positives = tf.reduce_sum(y_true * y_pred_binary)
        
        # Predicted positives: all pixels predicted as on
        predicted_positives = tf.reduce_sum(y_pred_binary)
        
        # Precision = TP / (TP + FP) = TP / Predicted Positives
        precision = tf.cond(
            tf.greater(predicted_positives, 0),
            lambda: true_positives / predicted_positives,
            lambda: tf.constant(1.0)  # Perfect precision if no predictions
        )
        
        return precision
    
    @staticmethod
    def recall_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Recall: Of the pixels that should be "on", how many did we correctly keep "on"?
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            
        Returns:
            Recall score
        """
        # Convert predictions to binary
        y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        
        # True positives: predicted on AND should be on
        true_positives = tf.reduce_sum(y_true * y_pred_binary)
        
        # Actual positives: all pixels that should be on
        actual_positives = tf.reduce_sum(y_true)
        
        # Recall = TP / (TP + FN) = TP / Actual Positives
        recall = tf.cond(
            tf.greater(actual_positives, 0),
            lambda: true_positives / actual_positives,
            lambda: tf.constant(1.0)  # Perfect recall if no actual positives
        )
        
        return recall
    
    @staticmethod
    def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        F1 score: Harmonic mean of precision and recall.
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            
        Returns:
            F1 score
        """
        precision = AblationMetrics.precision_score(y_true, y_pred)
        recall = AblationMetrics.recall_score(y_true, y_pred)
        
        f1 = tf.cond(
            tf.greater(precision + recall, 0),
            lambda: 2.0 * (precision * recall) / (precision + recall),
            lambda: tf.constant(0.0)
        )
        
        return f1
    
    @staticmethod
    def specificity_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Specificity: Of the pixels that should be "off", how many did we correctly turn "off"?
        This is crucial for ablation denoising.
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            
        Returns:
            Specificity score
        """
        # Convert predictions to binary
        y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        
        # True negatives: predicted off AND should be off
        true_negatives = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred_binary))
        
        # Actual negatives: all pixels that should be off
        actual_negatives = tf.reduce_sum(1.0 - y_true)
        
        # Specificity = TN / (TN + FP) = TN / Actual Negatives
        specificity = tf.cond(
            tf.greater(actual_negatives, 0),
            lambda: true_negatives / actual_negatives,
            lambda: tf.constant(1.0)  # Perfect specificity if no actual negatives
        )
        
        return specificity
    
    @staticmethod
    def false_positive_rate(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        False Positive Rate: Fraction of pixels incorrectly predicted as "on".
        
        Args:
            y_true: Target binary images
            y_pred: Predicted binary images
            
        Returns:
            False positive rate
        """
        return 1.0 - AblationMetrics.specificity_score(y_true, y_pred)
    
    @staticmethod
    def ablation_efficiency(y_true: tf.Tensor, y_pred: tf.Tensor, 
                           y_noisy: tf.Tensor) -> tf.Tensor:
        """
        Ablation Efficiency: How much noise was successfully removed?
        
        Args:
            y_true: Target binary images (clean)
            y_pred: Predicted binary images (denoised)
            y_noisy: Input noisy binary images
            
        Returns:
            Ablation efficiency score
        """
        # Convert predictions to binary
        y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        
        # Noise pixels: pixels that are on in noisy but off in clean
        noise_pixels = tf.maximum(y_noisy - y_true, 0.0)
        total_noise = tf.reduce_sum(noise_pixels)
        
        # Removed noise: noise pixels that were successfully turned off
        removed_noise = tf.reduce_sum(noise_pixels * (1.0 - y_pred_binary))
        
        # Efficiency = Removed Noise / Total Noise
        efficiency = tf.cond(
            tf.greater(total_noise, 0),
            lambda: removed_noise / total_noise,
            lambda: tf.constant(1.0)  # Perfect efficiency if no noise
        )
        
        return efficiency
    
    @staticmethod
    def pixel_conservation_rate(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Pixel Conservation Rate: How well did we preserve the original signal pixels?
        
        Args:
            y_true: Target binary images (clean)
            y_pred: Predicted binary images (denoised)
            
        Returns:
            Pixel conservation rate
        """
        # Convert predictions to binary
        y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        
        # Signal pixels: pixels that should be on
        signal_pixels = y_true
        total_signal = tf.reduce_sum(signal_pixels)
        
        # Conserved signal: signal pixels that were kept on
        conserved_signal = tf.reduce_sum(signal_pixels * y_pred_binary)
        
        # Conservation rate = Conserved Signal / Total Signal
        conservation_rate = tf.cond(
            tf.greater(total_signal, 0),
            lambda: conserved_signal / total_signal,
            lambda: tf.constant(1.0)  # Perfect conservation if no signal
        )
        
        return conservation_rate


def get_ablation_loss(loss_type: str = "precision_bce", **kwargs) -> Callable:
    """
    Get an ablation loss function by name.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function
    """
    if loss_type == "precision_bce":
        return AblationLoss.precision_focused_bce
    elif loss_type == "weighted_bce":
        fp_weight = kwargs.get('false_positive_weight', 2.0)
        return lambda y_true, y_pred: AblationLoss.weighted_precision_bce(
            y_true, y_pred, fp_weight
        )
    elif loss_type == "dice":
        smooth = kwargs.get('smooth', 1e-6)
        return lambda y_true, y_pred: AblationLoss.ablation_dice_loss(
            y_true, y_pred, smooth
        )
    elif loss_type == "fp_penalty":
        penalty_weight = kwargs.get('penalty_weight', 1.0)
        return lambda y_true, y_pred: AblationLoss.false_positive_penalty(
            y_true, y_pred, penalty_weight
        )
    elif loss_type == "combined":
        bce_weight = kwargs.get('bce_weight', 1.0)
        dice_weight = kwargs.get('dice_weight', 0.5)
        fp_weight = kwargs.get('fp_weight', 0.3)
        return lambda y_true, y_pred: AblationLoss.combined_ablation_loss(
            y_true, y_pred, bce_weight, dice_weight, fp_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_ablation_metrics() -> list:
    """
    Get list of ablation-specific metrics.
    
    Returns:
        List of metric functions
    """
    return [
        AblationMetrics.precision_score,
        AblationMetrics.recall_score,
        AblationMetrics.f1_score,
        AblationMetrics.specificity_score,
        AblationMetrics.false_positive_rate,
        AblationMetrics.pixel_conservation_rate
    ]


def compute_comprehensive_ablation_metrics(y_true: np.ndarray, 
                                         y_pred: np.ndarray,
                                         y_noisy: np.ndarray = None,
                                         threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute comprehensive ablation metrics for evaluation.
    
    Args:
        y_true: Target binary images (clean)
        y_pred: Predicted binary images (denoised)
        y_noisy: Input noisy binary images (optional)
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dictionary of metric values
    """
    # Convert to TensorFlow tensors
    y_true_tf = tf.constant(y_true, dtype=tf.float32)
    y_pred_tf = tf.constant(y_pred, dtype=tf.float32)
    
    metrics = {}
    
    # Basic metrics
    metrics['precision'] = float(AblationMetrics.precision_score(y_true_tf, y_pred_tf))
    metrics['recall'] = float(AblationMetrics.recall_score(y_true_tf, y_pred_tf))
    metrics['f1_score'] = float(AblationMetrics.f1_score(y_true_tf, y_pred_tf))
    metrics['specificity'] = float(AblationMetrics.specificity_score(y_true_tf, y_pred_tf))
    metrics['false_positive_rate'] = float(AblationMetrics.false_positive_rate(y_true_tf, y_pred_tf))
    metrics['pixel_conservation_rate'] = float(AblationMetrics.pixel_conservation_rate(y_true_tf, y_pred_tf))
    
    # Advanced metrics if noisy images provided
    if y_noisy is not None:
        y_noisy_tf = tf.constant(y_noisy, dtype=tf.float32)
        metrics['ablation_efficiency'] = float(
            AblationMetrics.ablation_efficiency(y_true_tf, y_pred_tf, y_noisy_tf)
        )
    
    # Additional statistics
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    
    # Pixel counts
    metrics['true_positives'] = float(np.sum(y_true * y_pred_binary))
    metrics['false_positives'] = float(np.sum((1 - y_true) * y_pred_binary))
    metrics['false_negatives'] = float(np.sum(y_true * (1 - y_pred_binary)))
    metrics['true_negatives'] = float(np.sum((1 - y_true) * (1 - y_pred_binary)))
    
    # Pixel density changes
    metrics['original_pixel_density'] = float(np.mean(y_true))
    metrics['predicted_pixel_density'] = float(np.mean(y_pred_binary))
    
    if y_noisy is not None:
        metrics['noisy_pixel_density'] = float(np.mean(y_noisy))
        metrics['noise_reduction'] = float(np.mean(y_noisy) - np.mean(y_pred_binary))
    
    return metrics


if __name__ == "__main__":
    # Demo usage
    import numpy as np
    
    # Create sample data
    batch_size, height, width = 4, 28, 28
    
    # Clean images (sparse)
    y_true = np.random.choice([0, 1], size=(batch_size, height, width, 1), p=[0.8, 0.2])
    
    # Noisy images (more dense - additive noise)
    y_noisy = y_true.copy()
    noise_mask = np.random.choice([0, 1], size=y_noisy.shape, p=[0.9, 0.1])
    y_noisy = np.maximum(y_noisy, noise_mask)  # Only add pixels
    
    # Predicted images (somewhere in between)
    y_pred = y_noisy * np.random.uniform(0.3, 1.0, size=y_noisy.shape)
    
    print("Ablation Loss and Metrics Demo")
    print("=" * 40)
    
    # Convert to tensors
    y_true_tf = tf.constant(y_true, dtype=tf.float32)
    y_pred_tf = tf.constant(y_pred, dtype=tf.float32)
    y_noisy_tf = tf.constant(y_noisy, dtype=tf.float32)
    
    # Test loss functions
    print("\nLoss Functions:")
    precision_bce = AblationLoss.precision_focused_bce(y_true_tf, y_pred_tf)
    dice_loss = AblationLoss.ablation_dice_loss(y_true_tf, y_pred_tf)
    fp_penalty = AblationLoss.false_positive_penalty(y_true_tf, y_pred_tf)
    combined_loss = AblationLoss.combined_ablation_loss(y_true_tf, y_pred_tf)
    
    print(f"Precision-focused BCE: {precision_bce:.4f}")
    print(f"Ablation Dice Loss: {dice_loss:.4f}")
    print(f"False Positive Penalty: {fp_penalty:.4f}")
    print(f"Combined Loss: {combined_loss:.4f}")
    
    # Test metrics
    print("\nMetrics:")
    precision = AblationMetrics.precision_score(y_true_tf, y_pred_tf)
    recall = AblationMetrics.recall_score(y_true_tf, y_pred_tf)
    f1 = AblationMetrics.f1_score(y_true_tf, y_pred_tf)
    specificity = AblationMetrics.specificity_score(y_true_tf, y_pred_tf)
    efficiency = AblationMetrics.ablation_efficiency(y_true_tf, y_pred_tf, y_noisy_tf)
    conservation = AblationMetrics.pixel_conservation_rate(y_true_tf, y_pred_tf)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Ablation Efficiency: {efficiency:.4f}")
    print(f"Pixel Conservation: {conservation:.4f}")
    
    # Comprehensive metrics
    print("\nComprehensive Metrics:")
    comprehensive = compute_comprehensive_ablation_metrics(
        y_true.squeeze(), y_pred.squeeze(), y_noisy.squeeze()
    )
    
    for key, value in comprehensive.items():
        print(f"{key}: {value:.4f}")
