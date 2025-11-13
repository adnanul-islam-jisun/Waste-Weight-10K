"""
Image Encoder Module: Visual Feature Extractor
Objective: Extract rich, high-dimensional representation from product images.

This module uses pre-trained Vision Transformers (ViT) ONLY - no CNN fallbacks.
Supports multiple ViT variants from small to huge models.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Literal


class ImageEncoder(nn.Module):
    """
    Vision Transformer-based Image Encoder - ViT ONLY (No CNN Fallback).
    
    Extracts comprehensive visual features from product images using
    state-of-the-art Vision Transformer models.
    
    Supported Models (in order of capacity):
    - 'vit_b_16': ViT-Base/16 (768-dim, 86M params) - **RECOMMENDED DEFAULT**
    - 'vit_b_32': ViT-Base/32 (768-dim, 88M params) - Faster inference
    - 'vit_l_16': ViT-Large/16 (1024-dim, 304M params) - Higher capacity
    - 'vit_l_32': ViT-Large/32 (1024-dim, 306M params) - Large + faster
    - 'vit_h_14': ViT-Huge/14 (1280-dim, 632M params) - Maximum capacity
    
    Args:
        model_name: ViT variant to use. Default: 'vit_b_16'
        pretrained: Whether to use ImageNet pre-trained weights. Default: True
        freeze_backbone: Whether to freeze the ViT backbone. Default: False
        output_dim: Dimension of output feature vector. If None, uses model's 
                   native dimension. Default: None
        dropout: Dropout rate for regularization. Default: 0.1
    
    Input Shape: (batch_size, 3, 224, 224) - RGB images
    Output Shape: (batch_size, output_dim) - Visual feature vectors
    
    Example:
        >>> encoder = ImageEncoder(model_name='vit_b_16', pretrained=True)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = encoder(images)  # (4, 768)
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'vit_b_16': {
            'loader': lambda weights: models.vit_b_16(weights=weights),
            'weights': models.ViT_B_16_Weights.IMAGENET1K_V1,
            'native_dim': 768,
            'params': '86M',
            'description': 'ViT-Base/16 - Balanced performance & speed'
        },
        'vit_b_32': {
            'loader': lambda weights: models.vit_b_32(weights=weights),
            'weights': models.ViT_B_32_Weights.IMAGENET1K_V1,
            'native_dim': 768,
            'params': '88M',
            'description': 'ViT-Base/32 - Faster inference with larger patches'
        },
        'vit_l_16': {
            'loader': lambda weights: models.vit_l_16(weights=weights),
            'weights': models.ViT_L_16_Weights.IMAGENET1K_V1,
            'native_dim': 1024,
            'params': '304M',
            'description': 'ViT-Large/16 - Higher capacity for complex images'
        },
        'vit_l_32': {
            'loader': lambda weights: models.vit_l_32(weights=weights),
            'weights': models.ViT_L_32_Weights.IMAGENET1K_V1,
            'native_dim': 1024,
            'params': '306M',
            'description': 'ViT-Large/32 - Large model with faster inference'
        },
        'vit_h_14': {
            'loader': lambda weights: models.vit_h_14(weights=weights),
            'weights': models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
            'native_dim': 1280,
            'params': '632M',
            'description': 'ViT-Huge/14 - Maximum capacity (SWAG pretrained)'
        }
    }
    
    def __init__(
        self, 
        model_name: Literal['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14'] = 'vit_b_16',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super(ImageEncoder, self).__init__()
        
        # Validate model name
        if model_name not in self.MODEL_CONFIGS:
            available = ', '.join(self.MODEL_CONFIGS.keys())
            raise ValueError(
                f"Invalid model_name '{model_name}'. "
                f"Available models: {available}"
            )
        
        config = self.MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.native_dim = config['native_dim']
        
        # Use native dimension if output_dim not specified
        if output_dim is None:
            output_dim = self.native_dim
        
        self.output_dim = output_dim
        
        # Load Vision Transformer
        try:
            weights = config['weights'] if pretrained else None
            vit_full_model = config['loader'](weights)
            
            # Store the full model but we'll use it carefully
            # ViT structure: conv_proj -> encoder -> heads
            # We want the encoder output (before classification head)
            self.vit_model = vit_full_model
            
            # Remove the classification head for feature extraction
            self.vit_model.heads = nn.Identity()
            
            print(f"✓ Loaded {model_name.upper().replace('_', '-')}")
            print(f"  Description: {config['description']}")
            print(f"  Parameters: {config['params']}")
            print(f"  Native output: {self.native_dim}-dim ([CLS] token)")
            if pretrained:
                print(f"  Pre-trained: ImageNet-1K")
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Vision Transformer '{model_name}'.\n"
                f"Please ensure torchvision >= 0.13.0 is installed.\n"
                f"Install: pip install --upgrade torchvision\n"
                f"Error: {str(e)}"
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vit_model.parameters():
                param.requires_grad = False
            # Unfreeze the projection head (if it exists and not Identity)
            for param in self.projection.parameters():
                param.requires_grad = True
            print(f"  Backbone frozen: {sum(p.numel() for p in self.vit_model.parameters()):,} params")
        else:
            print(f"  Trainable params: {sum(p.numel() for p in self.vit_model.parameters() if p.requires_grad):,}")
        
        # Projection head to map ViT output to desired output_dim
        # Only needed if output_dim differs from native_dim
        if output_dim != self.native_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.native_dim, max(output_dim * 2, self.native_dim)),
                nn.GELU(),  # GELU activation (used in transformers)
                nn.Dropout(dropout),
                nn.Linear(max(output_dim * 2, self.native_dim), output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            print(f"  Projection: {self.native_dim}-dim → {output_dim}-dim")
        else:
            # Identity projection (no-op)
            self.projection = nn.Identity()
            print(f"  No projection needed (output_dim matches native)")
        
        print(f"✓ ImageEncoder ready: output_dim={output_dim}\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer encoder.
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            Visual features (batch_size, output_dim)
        """
        # Extract features from Vision Transformer
        # After replacing heads with Identity, the model returns the [CLS] token features
        vit_features = self.vit_model(x)  # (batch_size, native_dim)
        
        # Project to desired output dimension
        visual_features = self.projection(vit_features)  # (batch_size, output_dim)
        
        return visual_features
    
    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.output_dim
    
    def get_native_dim(self) -> int:
        """Return the native ViT output dimension."""
        return self.native_dim
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the encoder.
        
        Args:
            trainable_only: If True, count only trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Factory Functions
# ============================================================================

def create_default_image_encoder(
    output_dim: int = 768,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.1
) -> ImageEncoder:
    """
    Create an ImageEncoder with default ViT-Base/16 configuration.
    
    This is the recommended configuration for most use cases - balanced
    performance and speed.
    
    Args:
        output_dim: Output feature dimension. Default: 768 (native ViT-B dim)
        pretrained: Use ImageNet pre-trained weights. Default: True
        freeze_backbone: Freeze ViT backbone. Default: False
        dropout: Dropout rate. Default: 0.1
    
    Returns:
        Configured ImageEncoder instance
    """
    return ImageEncoder(
        model_name='vit_b_16',
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        output_dim=output_dim,
        dropout=dropout
    )


def create_large_image_encoder(
    output_dim: int = 1024,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.1
) -> ImageEncoder:
    """
    Create a ViT-Large/16 encoder for higher capacity.
    
    Use this when you have:
    - Complex, high-resolution product images
    - Large dataset (to avoid overfitting)
    - Sufficient compute resources
    
    Args:
        output_dim: Output feature dimension. Default: 1024 (native ViT-L dim)
        pretrained: Use ImageNet pre-trained weights. Default: True
        freeze_backbone: Freeze ViT backbone. Default: False
        dropout: Dropout rate. Default: 0.1
    
    Returns:
        Configured ImageEncoder instance
    """
    return ImageEncoder(
        model_name='vit_l_16',
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        output_dim=output_dim,
        dropout=dropout
    )


def create_huge_image_encoder(
    output_dim: int = 1280,
    pretrained: bool = True,
    freeze_backbone: bool = True,  # Recommended: freeze for huge model
    dropout: float = 0.1
) -> ImageEncoder:
    """
    Create a ViT-Huge/14 encoder for maximum capacity.
    
    **WARNING**: This is a very large model (632M parameters).
    Only use if you have:
    - Very large dataset (100K+ images)
    - High-end GPU (16GB+ VRAM)
    - Complex visual features to capture
    
    **Recommendation**: Keep freeze_backbone=True to avoid overfitting.
    
    Args:
        output_dim: Output feature dimension. Default: 1280 (native ViT-H dim)
        pretrained: Use SWAG pre-trained weights. Default: True
        freeze_backbone: Freeze ViT backbone. Default: True (recommended!)
        dropout: Dropout rate. Default: 0.1
    
    Returns:
        Configured ImageEncoder instance
    """
    if not freeze_backbone:
        print("WARNING: Training ViT-Huge/14 end-to-end requires very large datasets!")
        print("         Consider setting freeze_backbone=True to avoid overfitting.")
    
    return ImageEncoder(
        model_name='vit_h_14',
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        output_dim=output_dim,
        dropout=dropout
    )


def create_fast_image_encoder(
    output_dim: int = 768,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.1
) -> ImageEncoder:
    """
    Create a ViT-Base/32 encoder optimized for faster inference.
    
    Uses larger patches (32x32 instead of 16x16) for faster processing.
    Recommended when:
    - Inference speed is critical
    - Images are relatively simple
    - Running on limited compute resources
    
    Args:
        output_dim: Output feature dimension. Default: 768
        pretrained: Use ImageNet pre-trained weights. Default: True
        freeze_backbone: Freeze ViT backbone. Default: False
        dropout: Dropout rate. Default: 0.1
    
    Returns:
        Configured ImageEncoder instance
    """
    return ImageEncoder(
        model_name='vit_b_32',
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        output_dim=output_dim,
        dropout=dropout
    )


# ============================================================================
# Model Selection Guide
# ============================================================================

def print_model_selection_guide():
    """Print a guide to help users select the appropriate ViT model."""
    
    guide = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║              VISION TRANSFORMER MODEL SELECTION GUIDE                     ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  Model        | Params | Output | Speed    | Use Case                    ║
    ║  ─────────────┼────────┼────────┼──────────┼─────────────────────────── ║
    ║  vit_b_16 ⭐  │  86M   │ 768-d  │ Balanced │ **DEFAULT** - Most tasks    ║
    ║  vit_b_32     │  88M   │ 768-d  │ Fast     │ Speed-critical applications ║
    ║  vit_l_16     │ 304M   │ 1024-d │ Slower   │ Complex images, large data  ║
    ║  vit_l_32     │ 306M   │ 1024-d │ Medium   │ Large model + speed balance ║
    ║  vit_h_14 ⚠️  │ 632M   │ 1280-d │ Slowest  │ Huge datasets, max capacity ║
    ║                                                                           ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║  RECOMMENDATIONS:                                                         ║
    ║  ─────────────────────────────────────────────────────────────────────   ║
    ║  • Small dataset (<10K images):    vit_b_16 or vit_b_32                  ║
    ║  • Medium dataset (10K-100K):      vit_b_16 or vit_l_16                  ║
    ║  • Large dataset (>100K):          vit_l_16 or vit_h_14                  ║
    ║  • Limited GPU memory (<8GB):      vit_b_32 (freeze_backbone=True)       ║
    ║  • Fast inference needed:          vit_b_32                              ║
    ║  • Maximum accuracy:               vit_h_14 (with large dataset)         ║
    ║                                                                           ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║  TRAINING STRATEGIES:                                                     ║
    ║  ─────────────────────────────────────────────────────────────────────   ║
    ║  1. Transfer Learning (Recommended):                                      ║
    ║     - Set pretrained=True, freeze_backbone=True                           ║
    ║     - Train only projection head                                          ║
    ║     - Fast training, good for small datasets                              ║
    ║                                                                           ║
    ║  2. Fine-tuning (Advanced):                                               ║
    ║     - Set pretrained=True, freeze_backbone=False                          ║
    ║     - Train entire network with small learning rate                       ║
    ║     - Best accuracy, requires more data                                   ║
    ║                                                                           ║
    ║  3. Progressive Training (Best):                                          ║
    ║     - Stage 1: freeze_backbone=True, train 10 epochs                      ║
    ║     - Stage 2: freeze_backbone=False, fine-tune 20 epochs                 ║
    ║     - Combines benefits of both approaches                                ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    
    Quick Start Examples:
    
    # Default (most common):
    encoder = create_default_image_encoder(output_dim=768)
    
    # High performance:
    encoder = create_large_image_encoder(output_dim=1024)
    
    # Maximum capacity:
    encoder = create_huge_image_encoder(output_dim=1280, freeze_backbone=True)
    
    # Fast inference:
    encoder = create_fast_image_encoder(output_dim=768)
    """
    
    print(guide)


if __name__ == "__main__":
    print_model_selection_guide()
    
    # Test all models
    print("\n" + "="*80)
    print("TESTING ALL VISION TRANSFORMER MODELS")
    print("="*80 + "\n")
    
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)
    
    for model_name in ImageEncoder.MODEL_CONFIGS.keys():
        print(f"\n{'─'*80}")
        print(f"Testing: {model_name.upper()}")
        print(f"{'─'*80}")
        
        try:
            encoder = ImageEncoder(
                model_name=model_name,
                pretrained=False,  # Don't download weights for testing
                output_dim=512  # Custom output dimension
            )
            
            # Forward pass
            with torch.no_grad():
                features = encoder(test_images)
            
            print(f"✓ Input:  {test_images.shape}")
            print(f"✓ Output: {features.shape}")
            print(f"✓ Total params: {encoder.get_num_parameters():,}")
            print(f"✓ Trainable params: {encoder.get_num_parameters(trainable_only=True):,}")
            
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*80}\n")
