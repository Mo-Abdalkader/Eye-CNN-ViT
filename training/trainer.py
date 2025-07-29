"""
Training and validation functions for diabetic retinopathy classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from training.early_stopping import EarlyStopping


def train_one_epoch(model, dataloader, optimizer, criterion_ret, criterion_mac, device):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion_ret (torch.nn.Module): Loss function for retinopathy
        criterion_mac (torch.nn.Module): Loss function for macular edema
        device (torch.device): Device to train on
    
    Returns:
        tuple: (average_loss, retinopathy_accuracy, macular_edema_accuracy)
    """
    model.train()
    running_loss, correct_ret, correct_mac, total = 0.0, 0, 0, 0

    for images, ret_labels, mac_labels in tqdm(dataloader, desc="Train", leave=False):
        images, ret_labels, mac_labels = images.to(device), ret_labels.to(device), mac_labels.to(device)

        optimizer.zero_grad()
        ret_out, mac_out = model(images)

        loss_ret = criterion_ret(ret_out, ret_labels)
        loss_mac = criterion_mac(mac_out, mac_labels)
        loss = loss_ret + loss_mac
        loss.backward()

        optimizer.step()
        running_loss += loss.item() * images.size(0)

        correct_ret += (ret_out.argmax(1) == ret_labels).sum().item()
        correct_mac += (mac_out.argmax(1) == mac_labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    acc_ret = correct_ret / total
    acc_mac = correct_mac / total

    return avg_loss, acc_ret, acc_mac


@torch.no_grad()
def validate(model, dataloader, criterion_ret, criterion_mac, device):
    """
    Validate the model.
    
    Args:
        model (torch.nn.Module): The model to validate
        dataloader (DataLoader): Validation data loader
        criterion_ret (torch.nn.Module): Loss function for retinopathy
        criterion_mac (torch.nn.Module): Loss function for macular edema
        device (torch.device): Device to validate on
    
    Returns:
        tuple: (average_loss, retinopathy_accuracy, macular_edema_accuracy)
    """
    model.eval()
    running_loss, correct_ret, correct_mac, total = 0.0, 0, 0, 0

    for images, ret_labels, mac_labels in tqdm(dataloader, desc="Val", leave=False):
        images, ret_labels, mac_labels = images.to(device), ret_labels.to(device), mac_labels.to(device)
        ret_out, mac_out = model(images)
        loss_ret = criterion_ret(ret_out, ret_labels)
        loss_mac = criterion_mac(mac_out, mac_labels)
        loss = loss_ret + loss_mac

        running_loss += loss.item() * images.size(0)

        correct_ret += (ret_out.argmax(1) == ret_labels).sum().item()
        correct_mac += (mac_out.argmax(1) == mac_labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    acc_ret = correct_ret / total
    acc_mac = correct_mac / total

    return avg_loss, acc_ret, acc_mac


def setup_training_components(model, class_weights, train_dataset):
    """
    Setup training components (optimizer, loss functions, sampler).
    
    Args:
        model (torch.nn.Module): The model
        class_weights (dict): Class weights for balancing
        train_dataset (Dataset): Training dataset
    
    Returns:
        tuple: (optimizer, criterion_ret, criterion_mac, sampler)
    """
    criterion_ret = nn.CrossEntropyLoss()
    criterion_mac = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setup weighted sampling for imbalanced classes
    y_train_ret = [y for _, y, _ in train_dataset]
    sample_weights = [class_weights['retinopathy_output'][int(y)] for y in y_train_ret]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return optimizer, criterion_ret, criterion_mac, sampler


def setup_dataloaders(train_dataset, val_dataset, sampler):
    """
    Setup data loaders for training and validation.
    
    Args:
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        sampler (Sampler): Training sampler
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader
