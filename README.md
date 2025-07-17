# Diabetic Retinopathy Classification: CNN vs ViT

A comprehensive deep learning project for diabetic retinopathy classification using both Convolutional Neural Networks (ResNet50) and Vision Transformers (ViT). The project implements dual-output classification for both retinopathy grading and macular edema risk assessment.

## Project Structure

```
CNN-Vs-ViT/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration parameters
├── data/
│   ├── __init__.py
│   └── dataset.py               # Dataset classes and data preparation
├── models/
│   ├── __init__.py
│   └── neural_networks.py       # Neural network architectures
├── utils/
│   ├── __init__.py
│   ├── image_processing.py      # Image loading and preprocessing
│   └── augmentation.py          # Data augmentation utilities
├── training/
│   ├── __init__.py
│   ├── early_stopping.py        # Early stopping utility
│   ├── trainer.py               # Training and validation functions
│   ├── evaluation.py            # Evaluation and visualization
│   └── pipeline.py              # Main training pipeline
├── scripts/
│   ├── __init__.py
│   └── main.py                  # Main entry point
├── outputs/                     # Model checkpoints and results
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Features

- **Dual-Output Classification**: Simultaneously predicts retinopathy grade (5 classes) and macular edema risk (3 classes)
- **Multiple Architectures**: Supports both ResNet50 and Vision Transformer (ViT) models
- **Advanced Data Augmentation**: Comprehensive augmentation pipeline including brightness, contrast, noise, blur, cutout, and CLAHE
- **Class Balancing**: Automatic dataset balancing through augmentation
- **Fine-tuning**: Two-phase training with initial training followed by fine-tuning
- **Early Stopping**: Prevents overfitting with configurable patience
- **Comprehensive Evaluation**: Training history plots and confusion matrices
- **Modular Design**: Clean, maintainable code structure following software engineering best practices

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Eye-CNN-ViT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete training pipeline for all model configurations:

```bash
python scripts/main.py
```

This will train:
- ResNet50
- ViT

### Custom Training

For custom training configurations, you can import and use the training pipeline:

```python
from training.pipeline import run_training_pipeline

# Train ResNet50 with augmentation and fine-tuning
run_training_pipeline(
    model_type='ResNet50',
    augment=False,
    fine_tune=True
)
```

### Configuration

Modify training parameters in `config/settings.py`:

```python
class Config:
    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 40
    EPOCHS_TUNE = 10
    LEARNING_RATE = 0.001
    
    # Data paths (update these for your dataset)
    TRAIN_IMAGES_PATH = "path/to/train/images"
    TRAIN_ANNOTATIONS_PATH = "path/to/train/annotations.csv"
    TEST_IMAGES_PATH = "path/to/test/images"
    TEST_ANNOTATIONS_PATH = "path/to/test/annotations.csv"
```

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

1. **Data Preparation**: Load and preprocess images, apply augmentation if enabled
2. **Model Initialization**: Create model with pre-trained weights
3. **Initial Training**: Train all layers with full learning rate
4. **Fine-tuning**: Freeze early layers, train later layers with reduced learning rate
5. **Evaluation**: Generate confusion matrices and training history plots

## Outputs

Trained models and results are saved in the `outputs/` directory:

- `{model_type}_{augment}_best_model.pth`: Best model from initial training
- `{model_type}_{augment}_best_model_finetuned.pth`: Best model after fine-tuning
- `{model_type}_{augment}_early_stopper_best_weights.pth`: Early stopping checkpoints

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
