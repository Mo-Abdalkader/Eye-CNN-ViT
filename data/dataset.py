"""
Dataset classes and data preparation utilities for diabetic retinopathy classification.
"""

import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

BASE_PATH = "D:\ENG ZAIAN's DATA\DATA\Augmented\Rotation 8, 16, 22 , -6, -14, -20, Flipping\IDRiD"
TRAIN_IMAGES_PATH = f"{BASE_PATH}/train/images"
TRAIN_ANNOTATIONS_PATH = f"{BASE_PATH}/train/annotations.csv"
TEST_IMAGES_PATH = f"{BASE_PATH}/test/images"
TEST_ANNOTATIONS_PATH = f"{BASE_PATH}/test/annotations.csv"


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


class CustomDataset(Dataset):
    """
    Custom dataset for diabetic retinopathy classification.
    """

    def __init__(self, df, image_dir, augment=False):
        """
        Initialize the dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame containing image metadata and labels
            image_dir (str): Directory containing the images
            augment (bool): Whether to apply data augmentation
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['Image name']}.jpg")
        image = load_and_preprocess_image(image_path, target_size=(224, 224))

        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.float32)

        image = apply_imagenet_normalization(image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        ret_label = torch.tensor(row['Retinopathy grade'], dtype=torch.long)
        mac_label = torch.tensor(row['Risk of macular edema'], dtype=torch.long)

        return image, ret_label, mac_label


def _create_balanced_dataset_with_augmentation(train_df):
    """
    Create a balanced dataset by augmenting minority classes.
    
    Args:
        train_df (pandas.DataFrame): Training DataFrame
    
    Returns:
        pandas.DataFrame: Balanced DataFrame
    """
    print("Creating balanced dataset with augmentation...")

    original_ret_counts = train_df['Retinopathy grade'].value_counts().sort_index()
    target_count_ret = original_ret_counts.max()
    all_rows = []

    for grade in range(5):
        class_subset = train_df[train_df['Retinopathy grade'] == grade]
        original_count = len(class_subset)

        if original_count == 0:
            continue

        num_to_generate = target_count_ret - original_count
        all_rows.append(class_subset)

        for _ in tqdm(range(num_to_generate), desc=f"Augmenting Grade {grade}"):
            sample_row = class_subset.sample(1).iloc[0]
            all_rows.append(sample_row.to_frame().T)

    balanced_df = pd.concat(all_rows, ignore_index=True)
    return balanced_df


def calculate_class_weights(y_ret, y_mac):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_ret (numpy.ndarray): Retinopathy labels
        y_mac (numpy.ndarray): Macular edema labels
    
    Returns:
        dict: Dictionary containing class weights for both tasks
    """
    ret_classes = np.unique(y_ret)
    ret_weights = compute_class_weight('balanced', classes=ret_classes, y=y_ret)
    ret_class_weights = dict(zip(ret_classes, ret_weights))

    mac_classes = np.unique(y_mac)
    mac_weights = compute_class_weight('balanced', classes=mac_classes, y=y_mac)
    mac_class_weights = dict(zip(mac_classes, mac_weights))

    return {
        'retinopathy_output': ret_class_weights,
        'macular_edema_output': mac_class_weights
    }


def prepare_datasets():
    """
    Prepare training and validation datasets.
    
    Args:
        augment (bool): Whether to apply data augmentation for balancing
    
    Returns:
        tuple: (train_dataset, val_dataset, class_weights)
    """
    train_df = pd.read_csv(TRAIN_ANNOTATIONS_PATH)
    val_df = pd.read_csv(TEST_ANNOTATIONS_PATH)

    # Clean column names
    train_df.columns = train_df.columns.str.strip()
    val_df.columns = val_df.columns.str.strip()

    y_train_ret = train_df['Retinopathy grade'].values
    y_train_mac = train_df['Risk of macular edema'].values

    class_weights = calculate_class_weights(y_train_ret, y_train_mac)

    train_dataset = CustomDataset(train_df, TRAIN_IMAGES_PATH)
    val_dataset = CustomDataset(val_df, TEST_IMAGES_PATH)

    return train_dataset, val_dataset, class_weights
