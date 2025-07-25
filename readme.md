## PolSAR vs Mamba: Solving Polarimetric SAR Image Classification Task with Novel Deep Learning Method Designed for Hyperspectral Remote Sensing Data

## Abstract

Polarimetric Synthetic Aperture Radar (PolSAR) image processing is not only a fundamental approach in remote sensing but also a critical technology for multimodal and multisource remote sensing. However, the inherently complex-valued nature of PolSAR data's polarimetric features necessitates deep learning methods that often involve complex operators (e.g., complex convolution and complex activation functions). This requirement conflicts with mainstream, state-of-the-art, fully real-valued deep learning architectures, making it difficult to apply novel architectures like Mamba and Vision Transformers (ViT) directly to PolSAR. In this paper, we address this challenge by employing a straightforward and effective method, the 9-Dimensional Real-Valued Transformation (9DRVT), to convert raw PolSAR data into purely real-valued representations. Furthermore, we propose a corresponding novel Mamba-based architecture that integrates attention mechanisms and U-Net techniques. Specifically, we introduce two key modules: (1) a Dual-Scanning Mamba (DSM) module that models bidirectional context (forward-backward) with linear complexity, eliminating unidirectional bias while enabling efficient long-range dependency capture; and (2) a Crossing-Attention module that jointly attends to horizontal and vertical spatial orientations, extending receptive fields and enabling adaptive feature refinement crucial for complex boundaries. The proposed architecture, named CADSM (Crossing-Attention Dual-Scanning Mamba), was evaluated on the Flevoland, San Francisco, and Oberpfaffenhofen datasets, achieving state-of-the-art (SOTA) accuracies of 99.79%, 99.38%, and 99.43%, respectively. Additionally, tests on the Indian Pines hyperspectral dataset and the Augsburg multimodal dataset demonstrated that CADSM significantly outperforms baseline models, highlighting its remarkable generalization capability. This research introduces a novel approach to address the efficiency limitations in feature modeling for PolSAR image classification, with its methodological framework providing broadly applicable insights across various remote sensing applications.

## Requirements:

- Python 3.7
- PyTorch >= 1.12.1

## Usage:

python main.py

