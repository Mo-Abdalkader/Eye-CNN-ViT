# Diabetic Retinopathy Classification: CNN vs ViT

A deep learning project for diabetic retinopathy classification using both Convolutional Neural Networks (ResNet50) and Vision Transformers (ViT). The project implements dual-output classification for both retinopathy grading and macular edema risk assessment.

## Project Structure

```
CNN-Vs-ViT/
├── data/
│   ├── __init__.py
│   └── dataset.py               # Dataset classes and data preparation (with preprocessing)
├── models/
│   ├── __init__.py
│   └── architecture.py          # Model architectures (ResNet50, ViT)
├── outputs/                     # Model checkpoints and results
├── training/
│   ├── __init__.py
│   ├── early_stopping.py        # Early stopping utility
│   ├── trainer.py               # Training and validation functions
│   ├── evaluation.py            # Evaluation and visualization
│   └── pipeline.py              # Main training pipeline
├── TrainCnn.py                  # Train ResNet50 only
├── TrainViT.py                  # Train ViT only
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Features

- **Dual-Output Classification**: Simultaneously predicts retinopathy grade (5 classes) and macular edema risk (3 classes)
- **Multiple Architectures**: Supports both ResNet50 and Vision Transformer (ViT) models
- **Fine-tuning**: Two-phase training with initial training followed by fine-tuning
- **Early Stopping**: Prevents overfitting with configurable patience
- **Comprehensive Evaluation**: Training history plots and confusion matrices
- **Simple, Flat Structure**: Minimal abstraction for easy maintenance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CNN-Vs-ViT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

To train the ResNet50 model:
```bash
python TrainCnn.py
```

To train the ViT model:
```bash
python TrainViT.py
```

Each script will run the full training and fine-tuning pipeline for the selected model. Results and model checkpoints will be saved in the `outputs/` directory.

### Custom Training

For advanced usage, you can import and use the training pipeline directly:
```python
from training.pipeline import run_training_pipeline

# Train ResNet50
run_training_pipeline(model_type='ResNet50', fine_tune=True)

# Train ViT
run_training_pipeline(model_type='ViT', fine_tune=True)
```

### Configuration

All configuration values (batch size, learning rate, image size, etc.) are now inlined in the code for simplicity. To change any parameter, edit the relevant value in the code (e.g., `data/dataset.py`, `training/pipeline.py`).

## Dataset Format

The project expects the following dataset structure:
```
dataset/
├── train/
│   ├── images/
│   │   ├── IDRiD_001.jpg
│   │   ├── IDRiD_002.jpg
│   │   └── ...
│   └── annotations.csv
└── test/
    ├── images/
    │   ├── IDRiD_001.jpg
    │   ├── IDRiD_002.jpg
    │   └── ...
    └── annotations.csv
```

The annotations CSV should contain:
- `Image name`: Image filename without extension
- `Retinopathy grade`: Integer (0-4) for retinopathy severity
- `Risk of macular edema`: Integer (0-2) for macular edema risk

## Model Architectures

### ResNet50
- Pre-trained ResNet50 backbone
- Custom dual-output heads for retinopathy and macular edema classification
- Batch normalization and dropout for regularization

### Vision Transformer (ViT)
- Pre-trained ViT-Base-Patch16-224 backbone
- Custom dual-output heads
- Same regularization techniques as ResNet50

## Training Process

1. **Data Preparation**: Load and preprocess images
2. **Model Initialization**: Create model with pre-trained weights
3. **Initial Training**: Train all layers with full learning rate
4. **Fine-tuning**: Freeze early layers, train later layers with reduced learning rate
5. **Evaluation**: Generate confusion matrices and training history plots

## Outputs

Trained models and results are saved in the `outputs/` directory:
- `resnet50_early_stopper_best_weights.pth`: Best ResNet50 model checkpoint
- `vit_early_stopper_best_weights.pth`: Best ViT model checkpoint
- Other files for fine-tuned models and results

## Performance Metrics

The models are evaluated on:
- **Retinopathy Classification**: 5-class accuracy (No DR, Mild, Moderate, Severe, Proliferative DR)
- **Macular Edema Classification**: 3-class accuracy (No DME, Grade 1, Grade 2)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IDRiD dataset for diabetic retinopathy images
- PyTorch and torchvision for deep learning framework
- timm library for Vision Transformer implementation 