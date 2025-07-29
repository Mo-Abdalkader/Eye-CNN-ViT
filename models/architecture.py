"""
Neural network models for diabetic retinopathy classification.
"""

import torch
import torch.nn as nn
from torchvision import models
import timm

class DualOutputResNet50(nn.Module):
    """
    ResNet50-based model with dual outputs for retinopathy and macular edema classification.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        self.bn = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.5)
        self.shared_dense = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        self.ret_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
        self.mac_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        x = self.dropout(x)
        shared = self.shared_dense(x)
        ret_out = self.ret_head(shared)
        mac_out = self.mac_head(shared)
        return ret_out, mac_out

class DualOutputViT(nn.Module):
    """
    Vision Transformer-based model with dual outputs for retinopathy and macular edema classification.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Identity()
        self.bn = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout(0.5)
        self.shared_dense = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        self.ret_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
        self.mac_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        x = self.vit(x)
        x = self.bn(x)
        x = self.dropout(x)
        shared = self.shared_dense(x)
        ret_out = self.ret_head(shared)
        mac_out = self.mac_head(shared)
        return ret_out, mac_out

def create_dual_output_model(model_type, device):
    if model_type == 'ResNet50':
        model = DualOutputResNet50()
    elif model_type == 'ViT':
        model = DualOutputViT()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device) 