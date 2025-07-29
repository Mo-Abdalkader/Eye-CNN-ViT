"""
Evaluation and visualization utilities for diabetic retinopathy classification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

RETINOPATHY_GRADES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
MACULAR_EDEMA_RISK = {0: "No DME", 1: "Grade 1", 2: "Grade 2 (Clinically Significant)"}


def plot_training_history(history):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history (dict): Dictionary containing training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Training History', fontsize=16)

    # Overall loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Overall Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Retinopathy accuracy
    axes[0, 1].plot(history['train_acc_ret'], label='Training Accuracy')
    axes[0, 1].plot(history['val_acc_ret'], label='Validation Accuracy')
    axes[0, 1].set_title('Retinopathy Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Macular edema accuracy
    axes[1, 0].plot(history['train_acc_mac'], label='Training Accuracy')
    axes[1, 0].plot(history['val_acc_mac'], label='Validation Accuracy')
    axes[1, 0].set_title('Macular Edema Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_confusion_matrices(model, dataloader, device):
    """
    Plot confusion matrices for both retinopathy and macular edema predictions.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (DataLoader): Data loader for evaluation
        device (torch.device): Device to run evaluation on
    """
    model.eval()

    y_true_ret, y_true_mac, y_pred_ret, y_pred_mac = [], [], [], []

    for images, ret_labels, mac_labels in tqdm(dataloader, desc="Confusion", leave=False):
        images = images.to(device)
        ret_out, mac_out = model(images)

        y_true_ret.extend(ret_labels.numpy())
        y_true_mac.extend(mac_labels.numpy())
        y_pred_ret.extend(ret_out.argmax(1).cpu().numpy())
        y_pred_mac.extend(mac_out.argmax(1).cpu().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Retinopathy confusion matrix
    cm_ret = confusion_matrix(y_true_ret, y_pred_ret)
    sns.heatmap(
        cm_ret, annot=True, fmt='d', cmap='Blues', ax=ax1,
        xticklabels=RETINOPATHY_GRADES.values(),
        yticklabels=RETINOPATHY_GRADES.values()
    )
    ax1.set_title('Retinopathy Grade Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # Macular edema confusion matrix
    cm_mac = confusion_matrix(y_true_mac, y_pred_mac)
    sns.heatmap(
        cm_mac, annot=True, fmt='d', cmap='Oranges', ax=ax2,
        xticklabels=MACULAR_EDEMA_RISK.values(),
        yticklabels=MACULAR_EDEMA_RISK.values()
    )
    ax2.set_title('Macular Edema Risk Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()


def save_weights(model, path):
    """
    Save model weights to file.
    
    Args:
        model (torch.nn.Module): Model to save
        path (str): Path to save the weights
    """
    torch.save(model.state_dict(), path)
    print(f"Weights saved to {path}")


def save_model(model, path):
    """
    Save entire model to file.
    
    Args:
        model (torch.nn.Module): Model to save
        path (str): Path to save the model
    """
    torch.save(model, path)
    print(f"Model saved to {path}")


def load_model(path, device):
    """
    Load model from file.
    
    Args:
        path (str): Path to the saved model
        device (torch.device): Device to load the model on
    
    Returns:
        torch.nn.Module: Loaded model
    """
    model = torch.load(path, map_location=device)
    model.eval()
    return model
