"""
Main training pipeline for diabetic retinopathy classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from data.dataset import prepare_datasets
from models.architecture import create_dual_output_model
from training.trainer import (
    train_one_epoch, validate, setup_training_components, setup_dataloaders
)
from training.early_stopping import EarlyStopping
from training.evaluation import plot_training_history, plot_confusion_matrices, save_model, save_weights


def run_training_pipeline(model_type='ResNet50', augment=False, fine_tune=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, class_weights = prepare_datasets()
    y_train_ret = [y for _, y, _ in train_dataset]

    sample_weights = [class_weights['retinopathy_output'][int(y)] for y in y_train_ret]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = create_dual_output_model(model_type, device)

    criterion_ret = nn.CrossEntropyLoss()
    criterion_mac = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc_ret': [], 'val_acc_ret': [], 'train_acc_mac': [],
               'val_acc_mac': []}

    early_stopper = EarlyStopping(patience=5, mode='min', verbose=True,
                                  path=f"outputs/{model_type.lower()}_early_stopper_best_weights.pth")

    for epoch in range(40):
        print(f"\nEpoch {epoch + 1}/{40}")

        train_loss, train_acc_ret, train_acc_mac = train_one_epoch(model, train_loader, optimizer, criterion_ret,
                                                                   criterion_mac, device)
        val_loss, val_acc_ret, val_acc_mac = validate(model, val_loader, criterion_ret, criterion_mac, device)

        print(f"Train Loss        : {train_loss:.4f}         | Val Loss: {val_loss:.4f}")
        print(f"Retinopathy Acc   : {train_acc_ret:.4f} (train) | {val_acc_ret:.4f} (val)")
        print(f"Macular Edema Acc : {train_acc_mac:.4f} (train) | {val_acc_mac:.4f} (val)")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc_ret'].append(train_acc_ret)
        history['val_acc_ret'].append(val_acc_ret)
        history['train_acc_mac'].append(train_acc_mac)
        history['val_acc_mac'].append(val_acc_mac)

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print()
            # save_model(model, f"outputs/{model_type.lower()}_best_model.pth")
            save_weights(model, f"outputs/{model_type.lower()}_best_weights.pth")

    plot_training_history(history)
    plot_confusion_matrices(model, val_loader, device)

    if fine_tune:
        print("\nStarting fine-tuning...")

        if model_type == 'ResNet50':
            # Freeze all:
            for param in model.backbone.parameters():
                param.requires_grad = False

            # Unfreeze layer3 and layer4:
            for block in [model.backbone.layer3, model.backbone.layer4]:
                for param in block.parameters():
                    param.requires_grad = True

        elif model_type == 'ViT':
            # Freeze all ViT parameters first
            for param in model.vit.parameters():
                param.requires_grad = False

            # Get the number of layers to unfreeze (default to 3 if not specified)
            layers_to_unfreeze = 3
            total_blocks = len(model.vit.blocks)

            print(f"Total ViT blocks: {total_blocks}")
            print(f"Unfreezing last {layers_to_unfreeze} blocks...")

            # Unfreeze the last N transformer blocks
            for i, block in enumerate(model.vit.blocks[-layers_to_unfreeze:], start=total_blocks - layers_to_unfreeze):
                for param in block.parameters():
                    param.requires_grad = True
                print(f"Unfroze block {i}")

            # Unfreeze the final layer norm
            if hasattr(model.vit, 'norm') and model.vit.norm is not None:
                for param in model.vit.norm.parameters():
                    param.requires_grad = True
                print("Unfroze final layer norm")

            # Unfreeze the fc_norm if it exists and is not Identity
            if hasattr(model.vit, 'fc_norm') and not isinstance(model.vit.fc_norm, nn.Identity):
                for param in model.vit.fc_norm.parameters():
                    param.requires_grad = True
                print("Unfroze fc_norm")

            # Always unfreeze the custom classifier heads
            if hasattr(model, 'ret_head'):
                for param in model.ret_head.parameters():
                    param.requires_grad = True
                print("Unfroze retinopathy head")

            if hasattr(model, 'mac_head'):
                for param in model.mac_head.parameters():
                    param.requires_grad = True
                print("Unfroze macular edema head")

            # Unfreeze shared dense layers if they exist
            if hasattr(model, 'shared_dense'):
                for param in model.shared_dense.parameters():
                    param.requires_grad = True
                print("Unfroze shared dense layers")

            # Unfreeze batch norm and dropout layers
            if hasattr(model, 'bn'):
                for param in model.bn.parameters():
                    param.requires_grad = True
                print("Unfroze batch norm layer")

            if hasattr(model, 'dropout'):
                for param in model.dropout.parameters():
                    param.requires_grad = True
                print("Unfroze dropout layer")

        # Use a lower learning rate for fine-tuning
        optimizer = optim.Adam(model.parameters(), lr=0.001 / 10)
        early_stopper = EarlyStopping(patience=5, mode='min', verbose=True,
                                      path=f"outputs/{model_type.lower()}_early_stopper_best_weights_finetuned.pth")

        for epoch in range(10):
            print(f"\nFine-tune Epoch {epoch + 1}/{10}")

            train_loss, train_acc_ret, train_acc_mac = train_one_epoch(model, train_loader, optimizer, criterion_ret,
                                                                       criterion_mac, device)
            val_loss, val_acc_ret, val_acc_mac = validate(model, val_loader, criterion_ret, criterion_mac, device)

            print(f"Train Loss        : {train_loss:.4f}            | Val Loss: {val_loss:.4f}")
            print(f"Retinopathy Acc   : {train_acc_ret:.4f} (train) | {val_acc_ret:.4f} (val)")
            print(f"Macular Edema Acc : {train_acc_mac:.4f} (train) | {val_acc_mac:.4f} (val)")

            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save_model(model, f"outputs/{model_type.lower()}_best_model_finetuned.pth")
                save_weights(model, f"outputs/{model_type.lower()}_best_weights_finetuned.pth")

        print("Fine-tuning complete.")
        plot_confusion_matrices(model, val_loader, device)
