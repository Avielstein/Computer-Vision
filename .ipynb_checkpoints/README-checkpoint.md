Computer Vision Repository
==========================

Welcome to the Computer Vision Repository! This repository features a collection of comprehensive guides and notebooks demonstrating the application of advanced computer vision techniques. Below is an overview of the projects included in this repository, along with direct links for immediate access and use.

Projects
--------

### 1\. Grad-CAM: Unifying Visualization, Agriculture, and Learning Dynamics

**Description:** This project consolidates three applications of Gradient-weighted Class Activation Mapping (Grad-CAM) into a comprehensive exploration of visualization techniques within deep learning. It demonstrates Grad-CAM's versatility in interpreting CNN decisions across varied contexts: standard image classification, agricultural produce classification, and visualizing model learning evolution.

**Highlights:**

1.  **Core Grad-CAM Techniques**: Introduces Grad-CAM with a practical TensorFlow tutorial, focusing on visualizing CNN focus areas through heatmaps.
2.  **Agricultural Application**: Extends Grad-CAM to classify fruits and vegetables, illustrating its utility in specialized data analysis.
3.  **Learning Visualization**: Transforms static Grad-CAM outputs into dynamic videos, showcasing model training progress and decision refinement.

The project offers a holistic view of Grad-CAM's utility, combining foundational knowledge, sector-specific application, and innovative learning visualization to enhance model interpretability and application insight.

* [Basic Gradcam Usage](GradCam.ipynb)
* [Custom CNN with GradCam](GradCamAgriculture/Veggi_Classification.ipynb)
* [Generate Videos](GradCamAgriculture/VisualizeGradcamLearning.ipynb)

![gradCam](GradCamAgriculture/gradCamApplied.png "gradCam")


### 2\. Advanced Image Segmentation Techniques

**Description:** Explores various image segmentation methods to partition images into distinct segments or regions. Techniques covered include SLIC, Felzenszwalb's method, Quickshift, Watershed, and Mean Shift segmentation, with comparisons on effectiveness across different image types.

[View Project](image_segmentation.ipynb)

![segmented](Felzenszwalb_seg.png "segmented")


### 3\. Feature Extraction from Floral and Fungi Footage

**Description:** Features lectures and timelapse footage of flora and fungi, demonstrating feature extraction techniques from video data. Includes detailed analysis of time-lapse videos of tomatoes and mushrooms, offering practical insights into video-based data analysis.

*   **Lecture on Feature Extraction:** [Watch Lecture](https://www.youtube.com/watch?v=7TCIeCOCHMc)
[![Lecture](https://img.youtube.com/vi/7TCIeCOCHMc/0.jpg)](https://www.youtube.com/watch?v=7TCIeCOCHMc)

*   **Tomato Time-lapse Analysis:** [Watch Tomato Timelapse](https://www.youtube.com/watch?v=Y8SaA25KlVk)
[![Tomato timelapse](https://img.youtube.com/vi/Y8SaA25KlVk/0.jpg)](https://www.youtube.com/watch?v=Y8SaA25KlVk)

*   **Mushroom Detection from Footage:** [Watch Mushroom Detection](https://www.youtube.com/watch?v=zauNC9Wd6cg)
[![mushrooms detected](https://img.youtube.com/vi/zauNC9Wd6cg/0.jpg)](https://www.youtube.com/watch?v=zauNC9Wd6cg)



### 4\. Anonymized Facial Expression: Dynamic Blurring with Superpixel Segmentation

**Description:** This project demonstrates a novel approach to facial anonymization, integrating superpixel segmentation with dynamic blurring to preserve facial expressions. Through practical Python implementations, it achieves real-time anonymization of faces in static images or video streams without sacrificing the expressiveness of facial features. Leveraging OpenCV for robust face detection and skimage for advanced segmentation, the project illustrates a sophisticated method to balance privacy and emotional conveyance.

[View Project](faceblur/faceblur.ipynb)

![Blurred](faceblur/blurred_zuck.png "blured")



