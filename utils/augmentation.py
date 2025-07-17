"""
Data augmentation utilities for diabetic retinopathy classification.
"""

import random
import numpy as np
from config.settings import Config
from utils.image_processing import (
    adjust_brightness, adjust_contrast, gamma_correction,
    add_gaussian_noise, gaussian_blur, cutout_augmentation, clahe_enhancement
)


def apply_random_augmentation(image, num_augmentations=None):
    """
    Apply random augmentations to an image.
    
    Args:
        image (numpy.ndarray): Input image
        num_augmentations (int): Number of augmentations to apply (random if None)
    
    Returns:
        tuple: (augmented_image, list_of_applied_augmentations)
    """
    augmented_image = image.copy()
    
    if num_augmentations is None:
        num_augmentations = np.random.randint(1, 4)
    
    augmentation_functions = {
        'brightness': lambda img: adjust_brightness(img, np.random.uniform(*Config.BRIGHTNESS_RANGE)),
        'contrast': lambda img: adjust_contrast(img, np.random.uniform(*Config.CONTRAST_RANGE)),
        'gamma_correction': lambda img: gamma_correction(img, np.random.uniform(*Config.GAMMA_RANGE)),
        'gaussian_noise': lambda img: add_gaussian_noise(img, np.random.uniform(0, Config.GAUSSIAN_NOISE_STD)),
        'gaussian_blur': lambda img: gaussian_blur(img, np.random.uniform(*Config.GAUSSIAN_BLUR_SIGMA)),
        'cutout': lambda img: cutout_augmentation(
            img, 
            np.random.randint(*Config.CUTOUT_NUM_HOLES), 
            np.random.randint(*Config.CUTOUT_SIZE_RANGE)
        ),
        'clahe': lambda img: clahe_enhancement(img, np.random.uniform(*Config.CLAHE_CLIP_LIMIT)),
    }
    
    available_augs = [
        name for name, prob in Config.AUG_PROBABILITIES.items() 
        if name in augmentation_functions and np.random.random() < prob
    ]
    
    selected_augs = random.sample(available_augs, min(len(available_augs), num_augmentations))
    
    for aug_name in selected_augs:
        try:
            augmented_image = augmentation_functions[aug_name](augmented_image)
        except Exception as e:
            print(f"Error applying {aug_name}: {e}")
            continue
    
    return augmented_image, selected_augs 