"""
Image processing utilities for diabetic retinopathy classification.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from config.settings import Config


def load_and_preprocess_image(image_path, target_size=None):
    """
    Load and preprocess an image from file path.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed image as float32 array
    """
    try:
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if target_size:
            image = cv2.resize(image, target_size)
        
        return image.astype(np.float32) / 255.0
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def apply_imagenet_normalization(image):
    """
    Apply ImageNet normalization to an image.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Normalized image
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (image - mean) / std


def adjust_brightness(image, factor):
    """Adjust image brightness by a factor."""
    return np.clip(image * factor, 0.0, 1.0)


def adjust_contrast(image, factor):
    """Adjust image contrast by a factor."""
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0.0, 1.0)


def gamma_correction(image, gamma):
    """Apply gamma correction to an image."""
    return np.clip(np.power(image, gamma), 0.0, 1.0)


def add_gaussian_noise(image, noise_std):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, noise_std, image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def gaussian_blur(image, sigma):
    """Apply Gaussian blur to an image."""
    return gaussian_filter(image, sigma=sigma)


def cutout_augmentation(image, num_holes, hole_size):
    """
    Apply cutout augmentation to an image.
    
    Args:
        image (numpy.ndarray): Input image
        num_holes (int): Number of holes to create
        hole_size (int): Size of each hole
    
    Returns:
        numpy.ndarray: Augmented image
    """
    augmented_image = image.copy()
    h, w = image.shape[:2]
    
    for _ in range(num_holes):
        x = np.random.randint(0, max(1, w - hole_size))
        y = np.random.randint(0, max(1, h - hole_size))
        
        actual_hole_size = np.random.randint(max(1, hole_size//2), hole_size)
        patch_mean = np.mean(image[y:y+actual_hole_size, x:x+actual_hole_size], axis=(0, 1))
        augmented_image[y:y+actual_hole_size, x:x+actual_hole_size] = patch_mean
    
    return augmented_image


def clahe_enhancement(image, clip_limit):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Args:
        image (numpy.ndarray): Input image
        clip_limit (float): CLAHE clip limit
    
    Returns:
        numpy.ndarray: Enhanced image
    """
    image_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_channels = [clahe.apply(image_uint8[:, :, i]) for i in range(image_uint8.shape[2])]
    enhanced_image = np.stack(enhanced_channels, axis=2)
    return enhanced_image.astype(np.float32) / 255.0 