"""
Early stopping utility for training neural networks.
"""

import torch
import numpy as np


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting during training.
    """
    
    def __init__(self, patience=5, mode='min', delta=0.0, verbose=False, path='checkpoint.pth'):
        """
        Initialize early stopping.
        
        Args:
            patience (int): How long to wait after last time validation loss improved
            mode (str): One of {'min', 'max'} — whether to look for decreasing or increasing metric
            delta (float): Minimum change in the monitored quantity to qualify as an improvement
            verbose (bool): If True, prints messages for each improvement
            path (str): Path to save the best model
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = np.Inf if mode == 'min' else -np.Inf
        self.mode = mode
        self.delta = delta
        self.path = path

    def __call__(self, score, model):
        """
        Check if training should stop early.
        
        Args:
            score (float): Current validation score
            model (torch.nn.Module): Model to save if improved
        """
        if self.mode == 'min':
            improvement = score < self.val_score - self.delta
        else:
            improvement = score > self.val_score + self.delta

        if self.best_score is None or improvement:
            self.best_score = score
            self.val_score = score
            self._save_checkpoint(model)
            self.counter = 0
            if self.verbose:
                print(f"✅ Validation {self.mode} improved to {score:.4f}. Model saved.")
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("🛑 Early stopping triggered.")

    def _save_checkpoint(self, model):
        """Save model checkpoint."""
        torch.save(model.state_dict(), self.path) 