"""
Main entry point for the diabetic retinopathy classification project.
"""

import warnings
import random
import numpy as np
import torch
from training.pipeline import run_training_pipeline
from config.settings import Config

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds():
    """Set random seeds for reproducibility."""
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main function to run the complete training pipeline."""
    print("🚀 Starting Diabetic Retinopathy Classification Training Pipeline")
    print("=" * 60)
    
    # Set random seeds
    set_seeds()
    
    # Run training for all model configurations
    configurations = [
        ('ResNet50', False, True),  # ResNet50 without augmentation
        # ('ResNet50', True, True),   # ResNet50 with augmentation
        ('ViT', False, True),       # ViT without augmentation
        # ('ViT', True, True),        # ViT with augmentation
    ]
    
    for model_type, augment, fine_tune in configurations:
        try:
            print(f"\n{'='*60}")
            print(f"Training Configuration:")
            print(f"  Model: {model_type}")
            print(f"  Augmentation: {augment}")
            print(f"  Fine-tuning: {fine_tune}")
            print(f"{'='*60}")
            
            run_training_pipeline(
                model_type=model_type,
                augment=augment,
                fine_tune=fine_tune
            )
            
            print(f"Completed training for {model_type} (augment={augment})")
            
        except Exception as e:
            print(f"Error during training {model_type} (augment={augment}): {e}")
            continue
    
    print("\nAll training configurations completed!")
    print("Check the 'outputs' directory for saved models and results.")


if __name__ == "__main__":
    main() 