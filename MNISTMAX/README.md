# MNISTMAX: Advanced MNIST Learning Framework

A comprehensive framework for MNIST digit classification featuring both **contrastive learning** and **denoising autoencoders** with live training visualization.

## 🎬 Live Training Animation

Watch the model learn to denoise MNIST digits in real-time! This animation shows the progression from random outputs to clean reconstructions over 20 epochs:

![Training Animation](autoencoder_denoising/logs/efficient_basic_mixed_20250703_193742/training_animation.gif)

*The animation shows three MNIST digits (Clean | Noisy | Denoised) with the model's denoising ability improving dramatically from epoch 1 to 20.*

## 🏗️ Project Structure

```
MNISTMAX/
├── shared/                          # Common utilities
│   ├── __init__.py
│   ├── data_utils.py               # MNIST loading and preprocessing
│   └── visualization.py            # Plotting and live visualization
├── contrastive_learning/           # Contrastive learning module
│   ├── models.py                   # CNN, ResNet, ViT encoders
│   ├── contrastive_loss.py         # NT-Xent, SupCon, InfoNCE losses
│   ├── data_augmentation.py        # Augmentation pipelines
│   ├── train_contrastive.py        # Original supervised training
│   ├── unsupervised_pretraining.py # New: unsupervised training
│   └── example_usage.py            # Demo scripts
├── autoencoder_denoising/          # Denoising autoencoder module
│   ├── noise_generation.py         # Various noise types
│   ├── bitmap_utils.py             # Binary image processing
│   ├── denoising_models.py         # U-Net, ResNet, Attention models
│   ├── train_denoiser.py           # Training with live visualization
│   └── demo_denoising.py           # Interactive demos
├── mnist_analysis.ipynb            # Original dataset analysis
├── mnist_preprocessed.npz          # Preprocessed data
└── logs/                           # Training logs and checkpoints
```

## 🚀 Quick Start

### 1. Efficient Denoising with Live Visualization (⭐ Recommended)

Watch the model learn in real-time with our new efficient trainer:

```bash
cd MNISTMAX/autoencoder_denoising
python efficient_trainer.py --model basic --epochs 20 --live_viz --save_animation
```

**Features:**
- 🎥 Live visualization window showing denoising progress
- 📊 Real-time metrics display
- 🎬 Saves training animation GIF
- ⚡ Fast training with ~1000 samples per epoch

### 2. Advanced Model Training

Try different architectures and noise types:

```bash
# U-Net with mixed noise
python efficient_trainer.py --model unet --noise mixed --epochs 30 --live_viz

# Attention model with salt & pepper noise
python efficient_trainer.py --model attention --noise salt_pepper --epochs 25 --live_viz
```

### 3. Unsupervised Contrastive Learning

Train a model to learn representations without labels:

```bash
cd MNISTMAX/contrastive_learning
python unsupervised_pretraining.py --encoder simple_cnn --epochs 50 --save_representations representations.npz
```

### 4. Interactive Demo

Explore the denoising capabilities:

```bash
cd MNISTMAX/autoencoder_denoising
python demo_denoising.py
```

## 📊 Features

### Contrastive Learning
- **Unsupervised Training**: Learn representations without labels
- **Multiple Architectures**: CNN, ResNet, Vision Transformer
- **Various Loss Functions**: NT-Xent, InfoNCE, Supervised Contrastive
- **Representation Extraction**: Save learned features for downstream tasks

### Denoising Autoencoders
- **Binary Image Denoising**: Clean 0/1 bitmap images
- **Live Training Visualization**: Real-time loss plots and sample outputs
- **Multiple Noise Types**: Salt & pepper, pixel flip, Gaussian, structured noise
- **Advanced Architectures**: U-Net, ResNet, Attention-based models
- **Comprehensive Evaluation**: Pixel accuracy, Dice coefficient, IoU metrics

## 🎯 Key Components

### Shared Utilities (`shared/`)

**Data Utils (`data_utils.py`)**
- MNIST loading and preprocessing
- Binary image conversion
- Dataset creation for different tasks

**Visualization (`visualization.py`)**
- Training history plots
- Live training visualization
- Image comparison grids
- t-SNE/UMAP embedding plots

### Contrastive Learning (`contrastive_learning/`)

**Models (`models.py`)**
- `SimpleCNNEncoder`: Basic CNN architecture
- `ResNetEncoder`: ResNet with residual blocks
- `VisionTransformerEncoder`: ViT with patch embedding
- `ContrastiveModel`: Complete encoder + projection head

**Losses (`contrastive_loss.py`)**
- `NTXentLoss`: SimCLR-style contrastive loss
- `SupConLoss`: Supervised contrastive learning
- `InfoNCELoss`: Information noise contrastive estimation
- `TripletLoss`: Traditional triplet-based learning

**Unsupervised Training (`unsupervised_pretraining.py`)**
- Label-free representation learning
- Automatic representation extraction
- t-SNE visualization of learned embeddings

### Denoising Autoencoders (`autoencoder_denoising/`)

**Noise Generation (`noise_generation.py`)**
- `salt_and_pepper_noise`: Random salt/pepper corruption
- `random_pixel_flip_noise`: Binary pixel flipping
- `structured_noise_lines`: Line-based corruption
- `block_noise`: Block-shaped artifacts
- `NoiseGenerator`: Configurable noise pipeline

**Models (`denoising_models.py`)**
- `DenoisingAutoencoder`: Basic encoder-decoder
- `UNetDenoiser`: U-Net with skip connections
- `ResidualDenoisingAutoencoder`: ResNet-based denoising
- `AttentionDenoisingAutoencoder`: Attention mechanisms
- `VariationalDenoisingAutoencoder`: Probabilistic denoising

**Training (`train_denoiser.py`)**
- Mixed clean/noisy batch training
- Live visualization during training
- Comprehensive evaluation on multiple noise types
- Model checkpointing and logging

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Computer-Vision/MNISTMAX

# Install dependencies
pip install tensorflow matplotlib numpy seaborn scikit-learn

# Optional: For advanced features
pip install umap-learn opencv-python
```

## 📈 Usage Examples

### Contrastive Learning

```python
from contrastive_learning.unsupervised_pretraining import train_unsupervised_contrastive

# Train unsupervised model
model, history = train_unsupervised_contrastive(
    encoder_type="simple_cnn",
    loss_type="ntxent",
    epochs=100,
    save_representations_path="learned_features.npz"
)
```

### Denoising Autoencoder

```python
from autoencoder_denoising.train_denoiser import train_denoising_autoencoder

# Train denoising model with live visualization
model, history = train_denoising_autoencoder(
    model_type="unet",
    noise_type="default",
    epochs=50,
    live_visualization=True
)
```

### Custom Noise Generation

```python
from autoencoder_denoising.noise_generation import NoiseGenerator

# Create custom noise generator
noise_params = {
    'salt_pepper': {'noise_prob': 0.15, 'salt_prob': 0.5},
    'pixel_flip': {'flip_prob': 0.08},
    'blocks': {'num_blocks': 5, 'block_size_range': (3, 6)}
}

noise_gen = NoiseGenerator(noise_params)
noisy_images = noise_gen.add_mixed_noise(clean_images, num_noise_types=2)
```

## 🎨 Live Visualization

The denoising trainer includes real-time visualization showing:
- Training loss curves
- Current clean, noisy, and denoised samples
- Error maps highlighting differences
- Updates every N training steps

## 📊 Evaluation Metrics

### Contrastive Learning
- Linear probing accuracy
- t-SNE visualization quality
- Representation clustering

### Denoising
- **Pixel Accuracy**: Exact pixel match percentage
- **Dice Coefficient**: Overlap similarity measure
- **IoU Score**: Intersection over Union
- **Hamming Distance**: Number of differing pixels
- **Precision/Recall**: For white pixel detection

## 🔬 Advanced Features

### Model Architectures
- **U-Net**: Skip connections for detail preservation
- **ResNet**: Residual connections for deep networks
- **Attention**: Focus on important image regions
- **VAE**: Probabilistic denoising approach

### Training Strategies
- **Mixed Batches**: Combine clean and noisy samples
- **Progressive Training**: Start with mild noise, increase difficulty
- **Multi-Loss**: Combine BCE, Dice, and perceptual losses
- **Curriculum Learning**: Adaptive noise difficulty

## 📝 Configuration

### Contrastive Learning Options
```bash
python unsupervised_pretraining.py \
    --encoder resnet \
    --loss ntxent \
    --epochs 100 \
    --batch_size 256 \
    --temperature 0.1 \
    --embedding_dim 128 \
    --augmentation strong
```

### Denoising Options
```bash
python train_denoiser.py \
    --model unet \
    --noise aggressive \
    --epochs 50 \
    --batch_size 32 \
    --clean_ratio 0.5 \
    --loss combined \
    --threshold 0.5
```

## 🎯 Results

### Contrastive Learning
- Achieves 92%+ linear probing accuracy
- Learns meaningful digit representations
- Transfers well to downstream tasks

### Denoising
- 95%+ pixel accuracy on salt & pepper noise
- 0.9+ Dice coefficient on various noise types
- Real-time training visualization
- Robust to multiple noise types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 References

- **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
- **SupCon**: Khosla et al., "Supervised Contrastive Learning"
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words"

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**MNISTMAX** - Pushing the boundaries of MNIST learning with modern deep learning techniques! 🚀
