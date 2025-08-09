# Backend README (FastAPI + PyTorch)

## Overview

This backend serves as the API for the CNN Visualizer project. It loads pretrained ResNet models, accepts images for inference, extracts intermediate convolutional layer activations, and returns prediction results along with detailed layer information.

## Features (So Far)

- Load pretrained ResNet18 and ResNet50 models with default weights.
- Register hooks to capture intermediate activations from all convolutional layers.
- Preprocess input images using torchvision’s recommended transforms.
- Run inference and return:
  - Top-5 predicted classes with probabilities.
  - Activation statistics (mean, std, max, min) for each conv layer.
  - Metadata about convolutional layers.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- FastAPI (for upcoming API endpoints)
- PIL (Pillow)

### Installation

```bash
pip install torch torchvision fastapi pillow
```

### Running Tests
Use the provided run_test.py script to test model loading and prediction on a sample image:

```bash
python backend/app/run_test.py
```

### Project Structure
```bash
backend/
├── app/
│   ├── main.py
│   ├── model.py        # CNNVisualizer class implementation
│   └── run_test.py     # Test script for model inference
├── venv/               
└── README.md
```