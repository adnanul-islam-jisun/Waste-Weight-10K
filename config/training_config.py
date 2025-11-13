"""
Training Configuration and Model Creation
Uses settings from config.py for unified configuration.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import unified config
from config.config import *

from models.image_encoder import create_default_image_encoder
from models.metadata_encoder import MetadataEncoder
from models.multimodal_fusion import MultimodalWeightPredictor, MultimodalTrainer
from models.loss_functions import recommend_loss_function


# ============================================================================
# WEIGHT PREPROCESSOR
# ============================================================================

class WeightPreprocessor:
    """
    Weight preprocessing for wide-range data (3.5kg - 3450kg).
    Uses LOG transformation (log1p/expm1) for better distribution.
    """
    
    def __init__(self, use_log_transform: bool = USE_LOG_TRANSFORM):
        self.use_log_transform = use_log_transform
        
    def transform(self, weights: np.ndarray) -> np.ndarray:
        """Transform weights for training."""
        if self.use_log_transform:
            return np.log1p(weights)
        else:
            return weights
    
    def inverse_transform(self, transformed_weights: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale."""
        if self.use_log_transform:
            return np.expm1(transformed_weights)
        else:
            return transformed_weights
    
    def transform_torch(self, weights: torch.Tensor) -> torch.Tensor:
        """Transform weights (PyTorch version)."""
        if self.use_log_transform:
            return torch.log1p(weights)
        else:
            return weights
    
    def inverse_transform_torch(self, transformed_weights: torch.Tensor) -> torch.Tensor:
        """Convert predictions back to original scale (PyTorch version)."""
        if self.use_log_transform:
            return torch.expm1(transformed_weights)
        else:
            return transformed_weights


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_optimized_model(
    num_categories: int,
    num_numerical_features: int,
    scaler=None,
    device: str = DEVICE
):
    """
    Create optimized model for weight prediction.
    
    Args:
        num_categories: Number of product types
        num_numerical_features: Number of numerical features
        scaler: Optional StandardScaler for numerical features
        device: Device to use (from config.DEVICE)
    
    Returns:
        model: Configured multimodal model
        preprocessor: Weight preprocessor
        loss_fn: Recommended loss function
    """
    
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    # 1. Image Encoder
    print(f"\n1. Image Encoder: {IMAGE_MODEL}")
    image_encoder = create_default_image_encoder(
        output_dim=IMAGE_OUTPUT_DIM,
        pretrained=True,
        freeze_backbone=False,
        dropout=DROPOUT_RATE
    )
    
    # 2. Metadata Encoder
    print("\n2. Metadata Encoder")
    metadata_encoder = MetadataEncoder(
        num_categories=num_categories,
        category_embedding_dim=CATEGORY_EMBEDDING_DIM,
        num_numerical_features=num_numerical_features,
        numerical_hidden_dims=[64, 32],
        output_dim=METADATA_OUTPUT_DIM,
        dropout=DROPOUT_RATE,
        scaler=scaler
    )
    
    # 3. Fusion Model
    print("\n3. Multimodal Fusion")
    model = MultimodalWeightPredictor(
        image_encoder=image_encoder,
        metadata_encoder=metadata_encoder,
        fusion_hidden_dims=FUSION_HIDDEN_DIMS,
        dropout=DROPOUT_RATE,
        use_residual=USE_RESIDUAL
    ).to(device)
    
    # 4. Weight Preprocessor
    print("\n4. Weight Preprocessor")
    preprocessor = WeightPreprocessor(use_log_transform=USE_LOG_TRANSFORM)
    print(f"   Transform: {'LOG (log1p/expm1)' if USE_LOG_TRANSFORM else 'None'}")
    
    # 5. Loss Function
    print("\n5. Loss Function")
    loss_fn = recommend_loss_function(
        weight_min=3.5,
        weight_max=3450.0,
        has_outliers=True,
        outlier_percentage=2.0
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*80}")
    print(f"✓ Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"{'='*80}\n")
    
    return model, preprocessor, loss_fn


# ============================================================================
# TRAINER CREATION
# ============================================================================

def create_trainer_for_your_data(model, preprocessor, loss_fn):
    """
    Create trainer with configuration from config.py
    
    Args:
        model: MultimodalWeightPredictor
        preprocessor: WeightPreprocessor
        loss_fn: Loss function
    
    Returns:
        trainer: Configured MultimodalTrainer
    """
    
    trainer = MultimodalTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        loss_fn=loss_fn
    )
    
    return trainer


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "TRAINING CONFIGURATION TEST" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    
    # Print config
    print_config()
    
    # Test model creation
    print("\nTesting model creation...")
    print("-" * 80)
    
    model, preprocessor, loss_fn = create_optimized_model(
        num_categories=10,
        num_numerical_features=11,
        scaler=None,
        device=DEVICE
    )
    
    print("✓ Model creation successful!")
    print("✓ Ready to train!")

