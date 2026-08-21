"""
Architecture Variants for Ablation Study
Implements Image-Only and Metadata-Only models for ablation experiments.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .image_encoder import ImageEncoder
from .metadata_encoder import MetadataEncoder


class ImageOnlyPredictor(nn.Module):
    """
    Image-only weight predictor for ablation study.
    Uses only visual features from ViT, no metadata.
    
    Architecture:
        Image → ViT → Regression Head → Weight
    
    Args:
        image_encoder: Pre-configured image encoder
        fusion_hidden_dims: Hidden dimensions for regression head
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        image_encoder: ImageEncoder,
        fusion_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3
    ):
        super(ImageOnlyPredictor, self).__init__()
        
        self.image_encoder = image_encoder
        
        # Get image feature dimension
        image_dim = image_encoder.output_dim
        
        # Regression head
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]
        
        layers = []
        in_dim = image_dim
        
        for hidden_dim in fusion_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.regression_head = nn.Sequential(*layers)
    
    def forward(self, images: torch.Tensor, category_indices=None, numerical_features=None):
        """
        Forward pass (category_indices and numerical_features are ignored).
        
        Args:
            images: Image tensor (batch_size, 3, 224, 224)
            category_indices: Ignored (for compatibility)
            numerical_features: Ignored (for compatibility)
        
        Returns:
            predictions: Weight predictions (batch_size, 1)
        """
        # Extract image features
        image_features = self.image_encoder(images)
        
        # Predict weight
        predictions = self.regression_head(image_features)
        
        return predictions
    
    def freeze_image_encoder(self):
        """Freeze image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class MetadataOnlyPredictor(nn.Module):
    """
    Metadata-only weight predictor for ablation study.
    Uses only metadata features (categories + numerical), no images.
    
    Architecture:
        Metadata → Metadata Encoder → Regression Head → Weight
    
    Args:
        metadata_encoder: Pre-configured metadata encoder
        fusion_hidden_dims: Hidden dimensions for regression head
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        metadata_encoder: MetadataEncoder,
        fusion_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3
    ):
        super(MetadataOnlyPredictor, self).__init__()
        
        self.metadata_encoder = metadata_encoder
        
        # Get metadata feature dimension
        metadata_dim = metadata_encoder.output_dim
        
        # Regression head
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]
        
        layers = []
        in_dim = metadata_dim
        
        for hidden_dim in fusion_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.regression_head = nn.Sequential(*layers)
    
    def forward(self, images=None, category_indices=None, numerical_features=None):
        """
        Forward pass (images are ignored).
        
        Args:
            images: Ignored (for compatibility)
            category_indices: Category indices (batch_size,)
            numerical_features: Numerical features (batch_size, num_features)
        
        Returns:
            predictions: Weight predictions (batch_size, 1)
        """
        # Extract metadata features
        metadata_features = self.metadata_encoder(category_indices, numerical_features)
        
        # Predict weight
        predictions = self.regression_head(metadata_features)
        
        return predictions
    
    def freeze_image_encoder(self):
        """No-op for compatibility."""
        pass
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def create_image_only_model(
    image_model: str = "vit_b_16",
    image_output_dim: Optional[int] = None,
    fusion_hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
    pretrained: bool = True,
    device: str = "cuda"
):
    """
    Factory function to create image-only model.
    
    Args:
        image_model: ViT model variant
        image_output_dim: Output dimension for image encoder
        fusion_hidden_dims: Hidden dimensions for regression head
        dropout: Dropout rate
        pretrained: Use pretrained weights
        device: Device to use
    
    Returns:
        model: ImageOnlyPredictor
    """
    from .image_encoder import ImageEncoder
    
    # Create image encoder
    image_encoder = ImageEncoder(
        model_name=image_model,
        output_dim=image_output_dim,
        pretrained=pretrained,
        freeze_backbone=False,
        dropout=dropout
    )
    
    # Create model
    model = ImageOnlyPredictor(
        image_encoder=image_encoder,
        fusion_hidden_dims=fusion_hidden_dims,
        dropout=dropout
    ).to(device)
    
    return model


def create_metadata_only_model(
    num_categories: int,
    num_numerical_features: int,
    category_embedding_dim: int = 32,
    metadata_output_dim: int = 256,
    fusion_hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
    scaler=None,
    device: str = "cuda"
):
    """
    Factory function to create metadata-only model.
    
    Args:
        num_categories: Number of product categories
        num_numerical_features: Number of numerical features
        category_embedding_dim: Embedding dimension for categories
        metadata_output_dim: Output dimension for metadata encoder
        fusion_hidden_dims: Hidden dimensions for regression head
        dropout: Dropout rate
        scaler: StandardScaler for numerical features
        device: Device to use
    
    Returns:
        model: MetadataOnlyPredictor
    """
    # Create metadata encoder
    metadata_encoder = MetadataEncoder(
        num_categories=num_categories,
        category_embedding_dim=category_embedding_dim,
        num_numerical_features=num_numerical_features,
        numerical_hidden_dims=[64, 32],
        output_dim=metadata_output_dim,
        dropout=dropout,
        scaler=scaler
    )
    
    # Create model
    model = MetadataOnlyPredictor(
        metadata_encoder=metadata_encoder,
        fusion_hidden_dims=fusion_hidden_dims,
        dropout=dropout
    ).to(device)
    
    return model


def create_ablation_model(
    exp_config: dict,
    num_categories: int,
    num_numerical_features: int,
    scaler=None,
    device: str = "cuda"
):
    """
    Create model based on ablation experiment configuration.
    
    Args:
        exp_config: Experiment configuration dict
        num_categories: Number of product categories
        num_numerical_features: Number of numerical features
        scaler: StandardScaler for numerical features
        device: Device to use
    
    Returns:
        model: Configured model for ablation experiment
    """
    use_image = exp_config.get("use_image", True)
    use_metadata = exp_config.get("use_metadata", True)
    use_attention = exp_config.get("use_attention_fusion", False)
    
    # Case 1: Image only
    if use_image and not use_metadata:
        return create_image_only_model(
            image_model=exp_config.get("image_model", "vit_b_16"),
            image_output_dim=exp_config.get("image_output_dim"),
            fusion_hidden_dims=exp_config.get("fusion_hidden_dims"),
            dropout=exp_config.get("dropout_rate", 0.3),
            pretrained=True,
            device=device
        )
    
    # Case 2: Metadata only
    if not use_image and use_metadata:
        return create_metadata_only_model(
            num_categories=num_categories,
            num_numerical_features=num_numerical_features,
            category_embedding_dim=exp_config.get("category_embedding_dim", 32),
            metadata_output_dim=exp_config.get("metadata_output_dim", 256),
            fusion_hidden_dims=exp_config.get("fusion_hidden_dims"),
            dropout=exp_config.get("dropout_rate", 0.3),
            scaler=scaler,
            device=device
        )
    
    # Case 3: Both image and metadata
    if use_image and use_metadata:
        if use_attention:
            # Use mutual attention fusion
            from .image_encoder import ImageEncoder
            from .metadata_encoder import MetadataEncoder
            from .mutual_attention_fusion import MultimodalWeightPredictor_WithAttention
            
            # Create image encoder with specified model
            image_encoder = ImageEncoder(
                model_name=exp_config.get("image_model", "vit_b_16"),
                pretrained=True,
                freeze_backbone=False,
                output_dim=exp_config.get("image_output_dim", 768),
                dropout=exp_config.get("dropout_rate", 0.3) * 0.5
            )
            
            # Create metadata encoder
            metadata_encoder = MetadataEncoder(
                num_categories=num_categories,
                category_embedding_dim=exp_config.get("category_embedding_dim", 32),
                num_numerical_features=num_numerical_features,
                numerical_hidden_dims=[64, 32],
                output_dim=exp_config.get("metadata_output_dim", 256),
                dropout=exp_config.get("dropout_rate", 0.3) * 0.5,
                scaler=scaler
            )
            
            # Create model with attention
            model = MultimodalWeightPredictor_WithAttention(
                image_encoder=image_encoder,
                metadata_encoder=metadata_encoder,
                embed_dim=exp_config.get("attention_embed_dim", 256),
                num_heads=exp_config.get("attention_num_heads", 8),
                dropout=exp_config.get("dropout_rate", 0.3),
                use_residual=True
            )
            return model.to(device)
        else:
            # Use simple late fusion
            from .image_encoder import ImageEncoder
            from .metadata_encoder import MetadataEncoder
            from .multimodal_fusion import MultimodalWeightPredictor
            
            image_encoder = ImageEncoder(
                model_name=exp_config.get("image_model", "vit_b_16"),
                output_dim=exp_config.get("image_output_dim"),
                pretrained=True,
                freeze_backbone=False,
                dropout=exp_config.get("dropout_rate", 0.3)
            )
            
            metadata_encoder = MetadataEncoder(
                num_categories=num_categories,
                category_embedding_dim=exp_config.get("category_embedding_dim", 32),
                num_numerical_features=num_numerical_features,
                numerical_hidden_dims=[64, 32],
                output_dim=exp_config.get("metadata_output_dim", 256),
                dropout=exp_config.get("dropout_rate", 0.3),
                scaler=scaler
            )
            
            return MultimodalWeightPredictor(
                image_encoder=image_encoder,
                metadata_encoder=metadata_encoder,
                fusion_hidden_dims=exp_config.get("fusion_hidden_dims"),
                dropout=exp_config.get("dropout_rate", 0.3),
                use_residual=True
            ).to(device)
    
    raise ValueError("Invalid experiment configuration: must use image, metadata, or both")
