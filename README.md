# CAPTCHA Classification Project

This project implements a CAPTCHA classification system using deep learning models. The system consists of two primary components: a model for predicting the length of the CAPTCHA and a character recognition model that decodes the CAPTCHA based on its predicted length. The models are built using PyTorch and utilize pre-trained ResNet architectures for character classification.

## Features

- **Length Classification**: A CNN-based model that predicts the length of the CAPTCHA (from 2 to 6 characters).
- **Character Recognition**: A modular character recognition system that uses ResNet architectures (e.g., ResNet18, ResNet34, ResNet50) to decode the CAPTCHA characters.
- **Batch Processing**: Efficient handling of multiple images through batch processing.
- **Customizable**: Supports customization of image dimensions and model parameters.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- Pillow
- logging

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>

