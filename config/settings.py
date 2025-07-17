"""
Configuration settings for the diabetic retinopathy classification project.
"""

class Config:
    # Data paths
    BASE_PATH = "D:\ENG ZAIAN's DATA\DATA\Augmented\Rotation 8, 16, 22 , -6, -14, -20, Flipping\IDRiD"
    TRAIN_IMAGES_PATH = f"{BASE_PATH}/train/images"
    TRAIN_ANNOTATIONS_PATH = f"{BASE_PATH}/train/annotations.csv"
    TEST_IMAGES_PATH = f"{BASE_PATH}/test/images"
    TEST_ANNOTATIONS_PATH = f"{BASE_PATH}/test/annotations.csv"

    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 40
    EPOCHS_TUNE = 10
    UNFREEZE_LAYERS_ViT = 3
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5
    RANDOM_SEED = 42

    # Augmentation parameters
    BRIGHTNESS_RANGE = (0.85, 1.15)
    CONTRAST_RANGE = (0.85, 1.15)
    GAMMA_RANGE = (0.9, 1.1)
    GAUSSIAN_NOISE_STD = 0.01
    GAUSSIAN_BLUR_SIGMA = (0.3, 0.8)
    CUTOUT_SIZE_RANGE = (5, 15)
    CUTOUT_NUM_HOLES = (1, 2)
    CLAHE_CLIP_LIMIT = (1.0, 2.0)
    ZOOM_RANGE = (0.95, 1.05)
    SHEAR_RANGE = (-3, 3)
    ROTATION_RANGE = (-5, 5)

    # Augmentation probabilities
    AUG_PROBABILITIES = {
        'brightness': 0.4, 'contrast': 0.4, 'gamma_correction': 0.3,
        'gaussian_noise': 0.2, 'gaussian_blur': 0.2, 'cutout': 0.3,
        'clahe': 0.4, 'zoom': 0.3, 'shear': 0.2, 'rotation': 0.3,
        'channel_shift': 0.2, 'histogram_equalization': 0.2
    }

    # Class labels
    RETINOPATHY_GRADES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
    MACULAR_EDEMA_RISK = {0: "No DME", 1: "Grade 1", 2: "Grade 2 (Clinically Significant)"} 