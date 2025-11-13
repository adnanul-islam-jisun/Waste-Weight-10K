"""
Training Configuration for Weight Prediction
Optimized for your dataset:
- Weight range: 3.5kg - 3450kg  
- Outliers: ~2% (200 samples)
- Distribution: Right-skewed, heavy-tailed, NOT normal
- Recommendation: MSLE loss with log transformation
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image_encoder import create_default_image_encoder
from models.metadata_encoder import create_metadata_encoder_from_data
from models.multimodal_fusion import create_multimodal_model, MultimodalTrainer
from models.loss_functions import (
    create_msle_loss,
    create_huber_loss,
    recommend_loss_function,
    WeightPredictionLoss
)


# ============================================================================
# DATA PREPROCESSING CONFIGURATION
# ============================================================================

class WeightPreprocessor:
    """
    Weight preprocessing for wide-range data (3.5kg - 3450kg).
    
    Based on data analysis:
    - Use LOG transformation (improves skewness from 1.08 to -0.40)
    - Better than standard scaling for this distribution
    """
    
    def __init__(self, use_log_transform: bool = True):
        """
        Args:
            use_log_transform: If True, use log(1+x) transformation
        """
        self.use_log_transform = use_log_transform
        self.mean = 751.93  # From data analysis
        self.std = 841.50   # From data analysis
        
    def transform(self, weights: np.ndarray) -> np.ndarray:
        """Transform weights for training."""
        if self.use_log_transform:
            # Log transformation (better for your data)
            return np.log1p(weights)
        else:
            # Standard scaling (alternative)
            return (weights - self.mean) / self.std
    
    def inverse_transform(self, transformed_weights: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale."""
        if self.use_log_transform:
            return np.expm1(transformed_weights)
        else:
            return transformed_weights * self.std + self.mean
    
    def transform_torch(self, weights: torch.Tensor) -> torch.Tensor:
        """Transform weights (PyTorch version)."""
        if self.use_log_transform:
            return torch.log1p(weights)
        else:
            return (weights - self.mean) / self.std
    
    def inverse_transform_torch(self, transformed_weights: torch.Tensor) -> torch.Tensor:
        """Convert predictions back to original scale (PyTorch version)."""
        if self.use_log_transform:
            return torch.expm1(transformed_weights)
        else:
            return transformed_weights * self.std + self.mean


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

def create_optimized_model(
    num_categories: int,
    num_numerical_features: int,
    scaler=None,
    device: str = 'cuda'
):
    """
    Create model optimized for your dataset.
    
    Configuration:
    - Image Encoder: ViT-B/16 (as requested)
    - Output dim: 768
    - Pretrained: True (ImageNet)
    - Freeze backbone: False (allow fine-tuning)
    
    Args:
        num_categories: Number of product types
        num_numerical_features: Number of numerical features (V_x, V_y, V_z, D_x, D_y, angle)
        scaler: StandardScaler for numerical features
        device: Device to use ('cuda', 'mps', or 'cpu')
    
    Returns:
        model: Configured multimodal model
        preprocessor: Weight preprocessor
        loss_fn: Recommended loss function
    """
    
    print("\n" + "="*80)
    print("CREATING OPTIMIZED MODEL FOR YOUR DATASET")
    print("="*80)
    
    # 1. Create image encoder (ViT-B/16)
    print("\n1. Image Encoder: ViT-Base/16")
    image_encoder = create_default_image_encoder(
        output_dim=768,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.1
    )
    
    # 2. Create metadata encoder
    print("\n2. Metadata Encoder")
    from models.metadata_encoder import MetadataEncoder
    metadata_encoder = MetadataEncoder(
        num_categories=num_categories,
        category_embedding_dim=32,
        num_numerical_features=num_numerical_features,
        numerical_hidden_dims=[64, 32],
        output_dim=256,
        dropout=0.1,
        scaler=scaler
    )
    
    # 3. Create fusion model
    print("\n3. Multimodal Fusion")
    from models.multimodal_fusion import MultimodalWeightPredictor
    model = MultimodalWeightPredictor(
        image_encoder=image_encoder,
        metadata_encoder=metadata_encoder,
        fusion_hidden_dims=[512, 256, 128],
        dropout=0.2,
        use_residual=True
    ).to(device)
    
    # 4. Create weight preprocessor
    print("\n4. Weight Preprocessor")
    preprocessor = WeightPreprocessor(use_log_transform=True)
    print(f"   Using LOG transformation (improves distribution)")
    
    # 5. Create loss function (MSLE for wide range)
    print("\n5. Loss Function")
    loss_fn = recommend_loss_function(
        weight_min=3.5,
        weight_max=3450.0,
        has_outliers=True,
        outlier_percentage=2.0
    )
    
    print(f"\n{'='*80}")
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*80}\n")
    
    return model, preprocessor, loss_fn


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training hyperparameters optimized for your dataset."""
    
    # Model
    IMAGE_OUTPUT_DIM = 768
    METADATA_OUTPUT_DIM = 256
    CATEGORY_EMBEDDING_DIM = 32
    
    # Data
    BATCH_SIZE = 32  # Adjust based on GPU memory
    NUM_WORKERS = 4
    USE_LOG_TRANSFORM = True  # IMPORTANT: Recommended for your data
    
    # Training
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss function
    LOSS_TYPE = 'msle'  # Recommended for wide range (3.5-3450kg)
    HUBER_DELTA = 10.0  # Only used if LOSS_TYPE='huber'
    
    # Learning rate schedule
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_PATIENCE = 10
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_MIN_LR = 1e-7
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    
    # Gradient clipping
    GRADIENT_CLIP_NORM = 1.0
    
    # Progressive training (recommended)
    FREEZE_IMAGE_ENCODER_EPOCHS = 10  # Freeze for first 10 epochs
    
    # Validation
    VALIDATION_SPLIT = 0.2
    
    # Checkpointing
    SAVE_BEST_MODEL = True
    CHECKPOINT_DIR = 'checkpoints'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        """Print configuration."""
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"\nModel:")
        print(f"  Image encoder:     ViT-B/16 (768-dim)")
        print(f"  Metadata encoder:  256-dim")
        print(f"  Category embedding: {cls.CATEGORY_EMBEDDING_DIM}-dim")
        
        print(f"\nData:")
        print(f"  Batch size:        {cls.BATCH_SIZE}")
        print(f"  Weight transform:  {'LOG (log1p)' if cls.USE_LOG_TRANSFORM else 'Standard scaling'}")
        print(f"  Validation split:  {cls.VALIDATION_SPLIT}")
        
        print(f"\nTraining:")
        print(f"  Epochs:            {cls.EPOCHS}")
        print(f"  Learning rate:     {cls.LEARNING_RATE}")
        print(f"  Weight decay:      {cls.WEIGHT_DECAY}")
        print(f"  Loss function:     {cls.LOSS_TYPE.upper()}")
        print(f"  Gradient clipping: {cls.GRADIENT_CLIP_NORM}")
        print(f"  Device:            {cls.DEVICE}")
        
        print(f"\nProgressive Training:")
        print(f"  Freeze encoder:    First {cls.FREEZE_IMAGE_ENCODER_EPOCHS} epochs")
        
        print(f"\nOptimization:")
        print(f"  LR scheduler:      {'Yes' if cls.USE_LR_SCHEDULER else 'No'}")
        print(f"  Early stopping:    {cls.EARLY_STOPPING_PATIENCE} epochs")
        
        print("="*80 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_trainer_for_your_data(model, preprocessor, loss_fn):
    """
    Create trainer with optimal configuration for your dataset.
    
    Args:
        model: MultimodalWeightPredictor
        preprocessor: WeightPreprocessor
        loss_fn: Loss function (WeightPredictionLoss or string)
    
    Returns:
        trainer: Configured MultimodalTrainer
    """
    
    trainer = MultimodalTrainer(
        model=model,
        device=TrainingConfig.DEVICE,
        learning_rate=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        loss_fn=loss_fn  # Can be WeightPredictionLoss object or string
    )
    
    return trainer


# ============================================================================
# QUICK START EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "TRAINING CONFIGURATION FOR YOUR DATASET" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    
    # Print configuration
    TrainingConfig.print_config()
    
    # Example: Create model
    print("\nEXAMPLE: Creating model for your dataset")
    print("-" * 80)
    
    # Assume you have these from your data
    num_categories = 10  # Example: number of product types
    num_numerical_features = 6  # V_x, V_y, V_z, D_x, D_y, view_angle
    
    # Create model
    model, preprocessor, loss_fn = create_optimized_model(
        num_categories=num_categories,
        num_numerical_features=num_numerical_features,
        scaler=None,  # Provide your fitted scaler
        device=TrainingConfig.DEVICE
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = create_trainer_for_your_data(model, preprocessor, loss_fn)
    
    print("\n" + "="*80)
    print("READY TO TRAIN!")
    print("="*80)
    
    print("\nNext steps:")
    print("1. Load your dataset")
    print("2. Preprocess weights using: preprocessor.transform(weights)")
    print("3. Create DataLoader")
    print("4. Start training:")
    print("""
    for epoch in range(TrainingConfig.EPOCHS):
        for batch in train_loader:
            loss = trainer.train_step(batch)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    """)
    
    print("\n" + "="*80)
    print("For complete training script, see: train.py (to be updated)")
    print("="*80 + "\n")
