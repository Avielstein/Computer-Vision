#!/usr/bin/env python3
"""
Data Setup Script for Computer Vision Repository

This script helps users set up the required data directories and provides
instructions for downloading the necessary datasets.
"""

import os
import sys

def create_directory(path, description):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✓ Created directory: {path}")
    else:
        print(f"✓ Directory already exists: {path}")
    print(f"  Purpose: {description}")

def main():
    print("=" * 60)
    print("Computer Vision Repository - Data Setup")
    print("=" * 60)
    
    # Create necessary directories
    directories = [
        ("image_segmentation_images", "Store test images for segmentation experiments"),
        ("GradCamAgriculture/AgriClassificationData", "Agricultural dataset (download separately)"),
        ("grad_cam_images", "Generated GradCAM visualization outputs"),
    ]
    
    print("\n1. Creating required directories...")
    for dir_path, description in directories:
        create_directory(dir_path, description)
    
    print("\n2. Data Download Instructions:")
    print("-" * 40)
    
    print("\n📁 GradCAM Agriculture Project:")
    print("   Dataset: Fruit and Vegetable Image Recognition")
    print("   Source: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")
    print("   Size: ~2GB")
    print("   Instructions:")
    print("   1. Create a Kaggle account if you don't have one")
    print("   2. Download the dataset from the link above")
    print("   3. Extract the contents to: GradCamAgriculture/AgriClassificationData/")
    print("   4. Verify the structure includes: train/, test/, validation/ folders")
    
    print("\n📁 Image Segmentation Project:")
    print("   Requirements: Any test images (JPEG/PNG)")
    print("   Instructions:")
    print("   1. Add your own images to: image_segmentation_images/")
    print("   2. Recommended: Images with clear objects/regions")
    print("   3. Update the image_path variable in the notebook accordingly")
    
    print("\n📁 Face Blur Project:")
    print("   Requirements: Portrait images for testing")
    print("   Instructions:")
    print("   1. Add test images to: faceblur/ directory")
    print("   2. Use images you have permission to process")
    print("   3. Clear frontal face images work best")
    
    print("\n3. Next Steps:")
    print("-" * 40)
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Download the datasets as instructed above")
    print("   3. Open the Jupyter notebooks and start experimenting!")
    
    print("\n✨ Setup complete! Happy coding!")

if __name__ == "__main__":
    main()
