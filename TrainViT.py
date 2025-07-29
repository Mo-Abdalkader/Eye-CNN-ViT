import warnings
import random
import numpy as np
import torch
from training.pipeline import run_training_pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42


def set_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("🚀 Starting ViT Training Pipeline")
    print("=" * 60)
    set_seeds()
    try:
        print(f"\n{'=' * 60}")
        run_training_pipeline(model_type='ViT', fine_tune=True)
        print(f"Completed training for ViT")
    except Exception as e:
        print(f"Error during training ViT: {e}")
    print("\nTraining completed!")
    print("Check the 'outputs' directory for saved models and results.")


if __name__ == "__main__":
    main()
