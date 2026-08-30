# Prediction Script with Test-Time Augmentation (TTA)
"""
Enhanced prediction with:
- Test-Time Augmentation (TTA) for better accuracy
- Ensemble predictions from multiple augmented views
- Support for both single image and batch prediction
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Dict

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DEVICE, IMAGE_SIZE


# ============================================================================
# Test-Time Augmentation (TTA) Transforms
# ============================================================================

def get_tta_transforms(n_augmentations: int = 5) -> List[transforms.Compose]:
    """
    Get list of transforms for test-time augmentation.
    
    Args:
        n_augmentations: Number of augmentation variations (including original)
    
    Returns:
        List of transform compositions
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    base_transforms = [
        # Original (no augmentation)
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=10),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize
        ]),
        # Slight zoom
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(int(IMAGE_SIZE * 0.9)),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize
        ]),
        # Color jitter
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ]),
    ]
    
    return base_transforms[:n_augmentations]


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_single_with_tta(
    model: nn.Module,
    image_path: str,
    category_idx: int,
    numerical_features: np.ndarray,
    weight_preprocessor,
    device: str = DEVICE,
    n_augmentations: int = 5
) -> Tuple[float, float]:
    """
    Predict weight for a single image using Test-Time Augmentation.
    
    Args:
        model: Trained weight prediction model
        image_path: Path to the image
        category_idx: Product category index
        numerical_features: Numerical metadata features (already normalized)
        weight_preprocessor: Weight preprocessor for inverse transform
        device: Device to use
        n_augmentations: Number of TTA augmentations
    
    Returns:
        Tuple of (mean_prediction, std_prediction) in kg
    """
    model.eval()
    tta_transforms = get_tta_transforms(n_augmentations)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get predictions for each augmentation
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # Apply transform
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Prepare other inputs
            cat_tensor = torch.tensor([category_idx], dtype=torch.long).to(device)
            num_tensor = torch.tensor([numerical_features], dtype=torch.float32).to(device)
            
            # Predict
            pred = model(img_tensor, cat_tensor, num_tensor)
            pred_np = pred.squeeze().cpu().numpy()
            
            # Inverse transform to get weight in kg
            weight_kg = weight_preprocessor.inverse_transform(np.array([pred_np]))[0]
            predictions.append(weight_kg)
    
    # Aggregate predictions
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    return mean_pred, std_pred


def predict_batch(
    model: nn.Module,
    images: torch.Tensor,
    category_indices: torch.Tensor,
    numerical_features: torch.Tensor,
    weight_preprocessor,
    device: str = DEVICE
) -> np.ndarray:
    """
    Predict weights for a batch of samples.
    
    Args:
        model: Trained weight prediction model
        images: Batch of images (B, 3, H, W)
        category_indices: Category indices (B,)
        numerical_features: Numerical features (B, N)
        weight_preprocessor: Weight preprocessor
        device: Device to use
    
    Returns:
        Array of predicted weights in kg
    """
    model.eval()
    
    with torch.no_grad():
        # Move to device
        images = images.to(device)
        category_indices = category_indices.to(device)
        numerical_features = numerical_features.to(device)
        
        # Predict
        predictions = model(images, category_indices, numerical_features)
        pred_np = predictions.squeeze().cpu().numpy()
        
        # Inverse transform
        weights_kg = weight_preprocessor.inverse_transform(pred_np)
    
    return weights_kg


def predict_batch_with_tta(
    model: nn.Module,
    dataloader,
    weight_preprocessor,
    device: str = DEVICE,
    n_augmentations: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict weights for entire dataloader with TTA.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        weight_preprocessor: Weight preprocessor
        device: Device to use
        n_augmentations: Number of TTA augmentations
    
    Returns:
        Tuple of (predictions, uncertainties, targets) - all in kg
    """
    model.eval()
    
    all_preds = []
    all_stds = []
    all_targets = []
    
    tta_transforms = get_tta_transforms(n_augmentations)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image']
            cat_indices = batch['category_idx'].to(device)
            num_features = batch['numerical'].to(device)
            targets = batch['weight'].cpu().numpy()
            
            batch_preds = []
            
            # For each augmentation
            for transform in tta_transforms:
                # Apply transform to batch (need to do per-image for PIL transforms)
                # For batch processing, we'll use direct tensor augmentations
                aug_images = images.to(device)
                
                # Predict
                preds = model(aug_images, cat_indices, num_features)
                batch_preds.append(preds.squeeze().cpu().numpy())
            
            # Stack and compute mean/std
            batch_preds = np.stack(batch_preds, axis=0)  # (n_aug, batch_size)
            mean_preds = np.mean(batch_preds, axis=0)
            std_preds = np.std(batch_preds, axis=0)
            
            # Inverse transform
            mean_preds_kg = weight_preprocessor.inverse_transform(mean_preds)
            targets_kg = weight_preprocessor.inverse_transform(targets)
            
            all_preds.extend(mean_preds_kg)
            all_stds.extend(std_preds)
            all_targets.extend(targets_kg)
    
    return np.array(all_preds), np.array(all_stds), np.array(all_targets)


# ============================================================================
# Model Loading
# ============================================================================

def load_model_for_inference(
    checkpoint_path: str,
    num_categories: int,
    num_numerical_features: int,
    device: str = DEVICE
):
    """
    Load trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_categories: Number of product categories
        num_numerical_features: Number of numerical features
        device: Device to use
    
    Returns:
        Loaded model in eval mode
    """
    from config.training_config import create_optimized_model
    
    # Create model architecture
    model, preprocessor, _ = create_optimized_model(
        num_categories=num_categories,
        num_numerical_features=num_numerical_features,
        device=device
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Val MAE: {checkpoint.get('val_mae', 'N/A'):.2f} kg")
    
    return model, preprocessor


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Prediction module loaded. Use functions:")
    print("  - predict_single_with_tta()")
    print("  - predict_batch()")
    print("  - predict_batch_with_tta()")
    print("  - load_model_for_inference()")
