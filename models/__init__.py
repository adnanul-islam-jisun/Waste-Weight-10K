# Models Module
"""
This module contains all neural network models and encoders for weight prediction.

Available modules:
- image_encoder: Vision Transformer-based image feature extractor (ViT ONLY)
- metadata_encoder: Heterogeneous metadata processor (categorical + numerical)
- multimodal_fusion: Combined model for weight prediction
- loss_functions: Comprehensive loss function module (separate file)
"""

from .image_encoder import (
    ImageEncoder,
    create_default_image_encoder,
    create_large_image_encoder,
    create_huge_image_encoder,
    create_fast_image_encoder,
    print_model_selection_guide
)

from .metadata_encoder import (
    MetadataEncoder,
    create_metadata_encoder_from_data
)

from .multimodal_fusion import (
    MultimodalWeightPredictor,
    create_multimodal_model,
    MultimodalTrainer
)

from .loss_functions import (
    WeightPredictionLoss,
    create_msle_loss,
    create_huber_loss,
    create_mae_loss,
    create_combined_loss,
    recommend_loss_function
)

__all__ = [
    # Image Encoder
    'ImageEncoder',
    'create_default_image_encoder',
    'create_large_image_encoder',
    'create_huge_image_encoder',
    'create_fast_image_encoder',
    'print_model_selection_guide',
    
    # Metadata Encoder
    'MetadataEncoder',
    'create_metadata_encoder_from_data',
    
    # Multimodal Fusion
    'MultimodalWeightPredictor',
    'create_multimodal_model',
    'MultimodalTrainer',
    
    # Loss Functions (NEW)
    'WeightPredictionLoss',
    'create_msle_loss',
    'create_huber_loss',
    'create_mae_loss',
    'create_combined_loss',
    'recommend_loss_function',
]

