"""
Multimodal Fusion Model: Combining Image and Metadata Encoders
Objective: Fuse visual and metadata features for weight prediction.

This model combines the ImageEncoder and MetadataEncoder to create a comprehensive
weight prediction system that leverages both visual and metadata information.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .image_encoder import ImageEncoder
from .metadata_encoder import MetadataEncoder
from .loss_functions import WeightPredictionLoss, recommend_loss_function


class MultimodalWeightPredictor(nn.Module):
    """
    Multimodal fusion model for weight prediction.
    
    Combines visual features from images and metadata features to predict
    product weight using a late fusion strategy.
    
    Architecture:
        1. Image Encoder (ViT) → V_features (768-dim)
        2. Metadata Encoder (Categorical + Numerical) → M_features (256-dim)
        3. Fusion Layer → Combined representation
        4. Regression Head → Weight prediction
    
    Args:
        image_encoder (ImageEncoder): Pre-configured image encoder
        metadata_encoder (MetadataEncoder): Pre-configured metadata encoder
        fusion_hidden_dims (List[int]): Hidden dimensions for fusion layers
        dropout (float): Dropout rate for regularization
        use_residual (bool): Whether to use residual connections in fusion
    """
    
    def __init__(
        self,
        image_encoder: ImageEncoder,
        metadata_encoder: MetadataEncoder,
        fusion_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        super(MultimodalWeightPredictor, self).__init__()
        
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.use_residual = use_residual
        
        # Get output dimensions from encoders
        image_feature_dim = image_encoder.get_output_dim()
        metadata_feature_dim = metadata_encoder.get_output_dim()
        combined_dim = image_feature_dim + metadata_feature_dim
        
        # ===== FUSION LAYER =====
        # Multi-layer fusion network with skip connections
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256, 128]
        
        fusion_layers = []
        in_dim = combined_dim
        
        for i, hidden_dim in enumerate(fusion_hidden_dims):
            fusion_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        fusion_output_dim = fusion_hidden_dims[-1]
        
        # Optional residual projection for skip connection
        if use_residual:
            self.residual_projection = nn.Linear(combined_dim, fusion_output_dim)
        
        # ===== REGRESSION HEAD =====
        # Final layers for weight prediction
        # NOTE: For direct weight prediction (no log transform), we use ReLU at the end
        #       to ensure positive outputs. ReLU allows large values (50-3450 kg range)
        #       while Softplus would compress the output range.
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Single output: predicted weight in kg
            nn.ReLU()  # Ensures positive output, allows full kg range
        )
        
        print(f"MultimodalWeightPredictor initialized:")
        print(f"  - Image features: {image_feature_dim}-dim")
        print(f"  - Metadata features: {metadata_feature_dim}-dim")
        print(f"  - Fusion dimensions: {fusion_hidden_dims}")
        print(f"  - Residual connections: {use_residual}")
        print(f"  - Output activation: Softplus (positive constraint)")
    
    def forward(
        self, 
        images: torch.Tensor,
        category_indices: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the multimodal model.
        
        Args:
            images (torch.Tensor): Batch of images, shape (batch_size, 3, 224, 224)
            category_indices (torch.Tensor): Category indices, shape (batch_size,)
            numerical_features (torch.Tensor): Numerical metadata, shape (batch_size, num_features)
        
        Returns:
            torch.Tensor: Predicted weights, shape (batch_size, 1)
        """
        # ===== FEATURE EXTRACTION =====
        # Extract visual features
        visual_features = self.image_encoder(images)  # (batch_size, image_feature_dim)
        
        # Extract metadata features
        metadata_features = self.metadata_encoder(
            category_indices, numerical_features
        )  # (batch_size, metadata_feature_dim)
        
        # ===== FEATURE FUSION =====
        # Concatenate visual and metadata features
        combined_features = torch.cat([visual_features, metadata_features], dim=1)
        
        # Pass through fusion network
        fused_features = self.fusion_network(combined_features)
        
        # Add residual connection if enabled
        if self.use_residual:
            residual = self.residual_projection(combined_features)
            fused_features = fused_features + residual
        
        # ===== WEIGHT PREDICTION =====
        # Final regression to predict weight
        weight_prediction = self.regression_head(fused_features)
        
        return weight_prediction
    
    def freeze_image_encoder(self):
        """Freeze the image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen.")
    
    def unfreeze_image_encoder(self):
        """Unfreeze the image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        print("Image encoder unfrozen.")
    
    def freeze_metadata_encoder(self):
        """Freeze the metadata encoder parameters."""
        for param in self.metadata_encoder.parameters():
            param.requires_grad = False
        print("Metadata encoder frozen.")
    
    def unfreeze_metadata_encoder(self):
        """Unfreeze the metadata encoder parameters."""
        for param in self.metadata_encoder.parameters():
            param.requires_grad = True
        print("Metadata encoder unfrozen.")
    
    def get_feature_representations(
        self,
        images: torch.Tensor,
        category_indices: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> dict:
        """
        Extract intermediate feature representations for analysis.
        
        Args:
            images (torch.Tensor): Batch of images
            category_indices (torch.Tensor): Category indices
            numerical_features (torch.Tensor): Numerical metadata
        
        Returns:
            dict: Dictionary containing:
                - 'visual_features': Features from image encoder
                - 'metadata_features': Features from metadata encoder
                - 'fused_features': Combined features after fusion
        """
        with torch.no_grad():
            visual_features = self.image_encoder(images)
            metadata_features = self.metadata_encoder(category_indices, numerical_features)
            combined_features = torch.cat([visual_features, metadata_features], dim=1)
            fused_features = self.fusion_network(combined_features)
            
            if self.use_residual:
                residual = self.residual_projection(combined_features)
                fused_features = fused_features + residual
        
        return {
            'visual_features': visual_features,
            'metadata_features': metadata_features,
            'fused_features': fused_features
        }


def create_multimodal_model(
    num_categories: int,
    num_numerical_features: int,
    image_output_dim: int = 768,
    metadata_output_dim: int = 256,
    category_embedding_dim: int = 32,
    pretrained_image_encoder: bool = True,
    scaler=None
) -> MultimodalWeightPredictor:
    """
    Create a complete multimodal weight prediction model.
    
    Args:
        num_categories (int): Number of product categories
        num_numerical_features (int): Number of numerical metadata features
        image_output_dim (int): Output dimension of image encoder
        metadata_output_dim (int): Output dimension of metadata encoder
        category_embedding_dim (int): Dimension of category embeddings
        pretrained_image_encoder (bool): Use pretrained image encoder
        scaler: StandardScaler for numerical features
    
    Returns:
        MultimodalWeightPredictor: Configured multimodal model
    """
    # Create image encoder
    image_encoder = ImageEncoder(
        pretrained=pretrained_image_encoder,
        freeze_backbone=False,
        output_dim=image_output_dim,
        dropout=0.1
    )
    
    # Create metadata encoder
    metadata_encoder = MetadataEncoder(
        num_categories=num_categories,
        category_embedding_dim=category_embedding_dim,
        num_numerical_features=num_numerical_features,
        numerical_hidden_dims=[64, 32],
        output_dim=metadata_output_dim,
        dropout=0.1,
        scaler=scaler
    )
    
    # Create fusion model
    model = MultimodalWeightPredictor(
        image_encoder=image_encoder,
        metadata_encoder=metadata_encoder,
        fusion_hidden_dims=[512, 256, 128],
        dropout=0.2,
        use_residual=True
    )
    
    return model


# Training utilities
class MultimodalTrainer:
    """
    Helper class for training the multimodal model with advanced loss functions.
    
    Supported Loss Functions:
    - 'mse': Mean Squared Error (L2) - Default, penalizes large errors heavily
    - 'mae': Mean Absolute Error (L1) - More robust to outliers
    - 'huber': Huber Loss - Combines MSE and MAE benefits
    - 'smooth_l1': Smooth L1 Loss - Similar to Huber, PyTorch's implementation
    - 'mape': Mean Absolute Percentage Error - Percentage-based errors
    - 'msle': Mean Squared Log Error - For weights spanning large ranges
    - 'quantile': Quantile Loss - For uncertainty estimation
    - 'combined': Weighted combination of MSE + MAE
    """
    
    def __init__(
        self,
        model: MultimodalWeightPredictor,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_fn: str = 'huber',  # Changed default to Huber
        huber_delta: float = 1.0,
        quantile_alpha: float = 0.5
    ):
        # Convert device string to torch.device
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn_name = loss_fn
        
        # Optimizer with different learning rates for different components
        # Handle ImageOnly, MetadataOnly, and full Multimodal models
        param_groups = []
        
        # Add image encoder (if exists)
        if hasattr(model, 'image_encoder'):
            param_groups.append({'params': model.image_encoder.parameters(), 'lr': learning_rate * 0.1})
        
        # Add metadata encoder (if exists)
        if hasattr(model, 'metadata_encoder'):
            param_groups.append({'params': model.metadata_encoder.parameters(), 'lr': learning_rate})
        
        # Always add regression head
        if hasattr(model, 'regression_head'):
            param_groups.append({'params': model.regression_head.parameters(), 'lr': learning_rate})
        elif hasattr(model, 'head'):
            param_groups.append({'params': model.head.parameters(), 'lr': learning_rate})
        
        # Add fusion layer parameters (different attribute names for different models)
        if hasattr(model, 'fusion_network'):
            # Late fusion model
            param_groups.append({'params': model.fusion_network.parameters(), 'lr': learning_rate})
        elif hasattr(model, 'attention_fusion'):
            # Attention fusion model
            param_groups.append({'params': model.attention_fusion.parameters(), 'lr': learning_rate})
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Select loss function
        if isinstance(loss_fn, str):
            # String-based loss function (backward compatibility)
            self.criterion = self._get_loss_function(loss_fn, huber_delta, quantile_alpha)
            self.loss_fn_name = loss_fn
        elif isinstance(loss_fn, WeightPredictionLoss):
            # Use the new WeightPredictionLoss object
            self.criterion = loss_fn
            self.loss_fn_name = loss_fn.loss_type
        else:
            # Assume it's a PyTorch loss function
            self.criterion = loss_fn
            self.loss_fn_name = loss_fn.__class__.__name__
        
        print(f"MultimodalTrainer initialized:")
        print(f"  Loss function: {self.loss_fn_name}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        if isinstance(loss_fn, WeightPredictionLoss):
            print(f"  Loss details: {loss_fn.info.get('name', 'Custom')}")
        elif loss_fn == 'huber':
            print(f"  Huber delta: {huber_delta}")
        elif loss_fn == 'quantile':
            print(f"  Quantile alpha: {quantile_alpha}")
    
    def _get_loss_function(self, loss_fn: str, huber_delta: float, quantile_alpha: float):
        """Select and return the appropriate loss function."""
        
        if loss_fn == 'mse':
            # Mean Squared Error (L2 Loss)
            # Best for: Gaussian-distributed errors, when you want to heavily penalize large errors
            return nn.MSELoss()
        
        elif loss_fn == 'mae' or loss_fn == 'l1':
            # Mean Absolute Error (L1 Loss)
            # Best for: Robust to outliers, when errors are not normally distributed
            return nn.L1Loss()
        
        elif loss_fn == 'huber':
            # Huber Loss (Smooth L1 variant)
            # Best for: RECOMMENDED - Combines MSE and MAE benefits
            # Uses MSE for small errors, MAE for large errors (robust to outliers)
            return nn.HuberLoss(delta=huber_delta)
        
        elif loss_fn == 'smooth_l1':
            # Smooth L1 Loss
            # Best for: Similar to Huber, PyTorch's original implementation
            return nn.SmoothL1Loss()
        
        elif loss_fn == 'mape':
            # Mean Absolute Percentage Error
            # Best for: When you care about relative errors (percentage-based)
            return self._mape_loss
        
        elif loss_fn == 'msle':
            # Mean Squared Log Error
            # Best for: Weights spanning large ranges (e.g., 1kg to 1000kg)
            return self._msle_loss
        
        elif loss_fn == 'quantile':
            # Quantile Loss
            # Best for: Uncertainty estimation, asymmetric penalties
            self.quantile_alpha = quantile_alpha
            return self._quantile_loss
        
        elif loss_fn == 'combined':
            # Combined Loss (MSE + MAE)
            # Best for: Getting benefits of both MSE and MAE
            return self._combined_loss
        
        else:
            raise ValueError(
                f"Unknown loss function: {loss_fn}. "
                f"Available: mse, mae, huber, smooth_l1, mape, msle, quantile, combined"
            )
    
    def _mape_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Percentage Error."""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        return torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100
    
    def _msle_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean Squared Log Error."""
        # Add 1 to handle zero values, assumes weights are positive
        return torch.mean((torch.log1p(predictions) - torch.log1p(targets)) ** 2)
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Quantile Loss for uncertainty estimation."""
        errors = targets - predictions
        return torch.mean(torch.maximum(
            self.quantile_alpha * errors,
            (self.quantile_alpha - 1) * errors
        ))
    
    def _combined_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined MSE + MAE loss."""
        mse = nn.MSELoss()(predictions, targets)
        mae = nn.L1Loss()(predictions, targets)
        # Weight: 0.7 MSE + 0.3 MAE (can be tuned)
        return 0.7 * mse + 0.3 * mae
    
    def train_step(self, batch: dict, scaler=None) -> float:
        """
        Single training step with Automatic Mixed Precision (AMP) support.
        
        Args:
            batch: Dictionary containing batch data
            scaler: GradScaler for AMP (if None, regular training is used)
        
        Returns:
            Loss value as float
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        images = batch['image'].to(self.device)
        category_indices = batch['category_idx'].to(self.device)
        numerical_features = batch['numerical'].to(self.device)
        targets = batch['weight'].to(self.device)
        
        # Use Automatic Mixed Precision if scaler is provided
        if scaler is not None:
            # Forward pass with autocast (updated API)
            with torch.amp.autocast('cuda'):
                predictions = self.model(images, category_indices, numerical_features)
                loss = self.criterion(predictions.squeeze(), targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first)
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # Regular training (no AMP)
            predictions = self.model(images, category_indices, numerical_features)
            loss = self.criterion(predictions.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch: dict, use_amp: bool = False) -> tuple:
        """
        Single validation step with optional AMP support.
        
        Args:
            batch: Dictionary containing batch data
            use_amp: Whether to use automatic mixed precision
        
        Returns:
            Tuple of (loss, predictions, targets)
        """
        self.model.eval()
        
        with torch.no_grad():
            images = batch['image'].to(self.device)
            category_indices = batch['category_idx'].to(self.device)
            numerical_features = batch['numerical'].to(self.device)
            targets = batch['weight'].to(self.device)
            
            # Use autocast for inference if AMP is enabled
            if use_amp and self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    predictions = self.model(images, category_indices, numerical_features)
                    loss = self.criterion(predictions.squeeze(), targets)
            else:
                predictions = self.model(images, category_indices, numerical_features)
                loss = self.criterion(predictions.squeeze(), targets)
        
        return loss.item(), predictions, targets
    
    def freeze_image_encoder(self):
        """Freeze image encoder parameters for progressive training (if it exists)."""
        if hasattr(self.model, 'image_encoder'):
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            print("✓ Image encoder frozen")
        else:
            print("⚠ Model has no image encoder to freeze")
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ All parameters unfrozen")
