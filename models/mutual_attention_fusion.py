"""
Mutual Attention Block for Multimodal Fusion
Advanced cross-attention mechanism for fusing visual and metadata features.

This module implements a symmetrical dual-branch cross-attention mechanism
that allows visual and metadata features to attend to each other, learning
which features are most relevant for weight prediction.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism.
    
    Allows one set of features (query) to attend to another set (key/value).
    
    Args:
        query_dim: Dimension of query features
        kv_dim: Dimension of key/value features
        embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(MultiHeadCrossAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query features (batch_size, query_dim)
            key: Key features (batch_size, kv_dim)
            value: Value features (batch_size, kv_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Attended features (batch_size, embed_dim)
            Optionally: attention weights (batch_size, num_heads, 1, 1)
        """
        batch_size = query.size(0)
        
        # Project to Q, K, V
        Q = self.query_proj(query)  # (batch_size, embed_dim)
        K = self.key_proj(key)      # (batch_size, embed_dim)
        V = self.value_proj(value)  # (batch_size, embed_dim)
        
        # Reshape for multi-head attention
        # (batch_size, num_heads, 1, head_dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch_size, num_heads, 1, 1)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch_size, num_heads, 1, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        # (batch_size, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attn_weights.squeeze()
        return output


class MutualAttentionBlock(nn.Module):
    """
    Mutual (Bidirectional) Cross-Attention Block.
    
    Implements symmetrical dual-branch cross-attention:
    - Branch 1: Visual features attend to metadata
    - Branch 2: Metadata features attend to visual
    
    Both branches are combined with residual connections to preserve
    original information while adding cross-modal context.
    
    Args:
        visual_dim: Dimension of visual features (e.g., 768 for ViT)
        metadata_dim: Dimension of metadata features (e.g., 256)
        embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_residual: Whether to include residual connections
    """
    
    def __init__(
        self,
        visual_dim: int,
        metadata_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super(MutualAttentionBlock, self).__init__()
        
        self.use_residual = use_residual
        
        # Branch 1: Visual → Metadata attention
        # Visual features ask: "Which metadata is relevant to me?"
        self.visual_to_metadata = MultiHeadCrossAttention(
            query_dim=visual_dim,
            kv_dim=metadata_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Branch 2: Metadata → Visual attention
        # Metadata asks: "Which visual features are relevant to me?"
        self.metadata_to_visual = MultiHeadCrossAttention(
            query_dim=metadata_dim,
            kv_dim=visual_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Residual projections (if dimensions don't match)
        if use_residual:
            self.visual_residual_proj = nn.Linear(visual_dim, embed_dim)
            self.metadata_residual_proj = nn.Linear(metadata_dim, embed_dim)
        
        # Fusion layer to combine both attended features
        fusion_input_dim = embed_dim * 2  # Two attended features
        if use_residual:
            fusion_input_dim += embed_dim * 2  # Add projected residuals (not original dims)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
        print(f"MutualAttentionBlock initialized:")
        print(f"  - Visual dim: {visual_dim}")
        print(f"  - Metadata dim: {metadata_dim}")
        print(f"  - Embed dim: {embed_dim}")
        print(f"  - Num heads: {num_heads}")
        print(f"  - Residual connections: {use_residual}")
    
    def forward(
        self,
        visual_features: torch.Tensor,
        metadata_features: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass through mutual attention.
        
        Args:
            visual_features: Visual features (batch_size, visual_dim)
            metadata_features: Metadata features (batch_size, metadata_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Fused features (batch_size, embed_dim)
            Optionally: dict of attention weights
        """
        # Branch 1: Visual → Metadata
        # Visual features attend to metadata
        v_to_m = self.visual_to_metadata(
            query=visual_features,
            key=metadata_features,
            value=metadata_features,
            return_attention=return_attention
        )
        
        if return_attention:
            v_to_m_output, v_to_m_attn = v_to_m
        else:
            v_to_m_output = v_to_m
        
        v_to_m_output = self.norm1(v_to_m_output)
        
        # Branch 2: Metadata → Visual
        # Metadata features attend to visual
        m_to_v = self.metadata_to_visual(
            query=metadata_features,
            key=visual_features,
            value=visual_features,
            return_attention=return_attention
        )
        
        if return_attention:
            m_to_v_output, m_to_v_attn = m_to_v
        else:
            m_to_v_output = m_to_v
        
        m_to_v_output = self.norm2(m_to_v_output)
        
        # Concatenate attended features
        attended_features = [v_to_m_output, m_to_v_output]
        
        # Add residual connections (original features)
        if self.use_residual:
            visual_residual = self.visual_residual_proj(visual_features)
            metadata_residual = self.metadata_residual_proj(metadata_features)
            attended_features.extend([visual_residual, metadata_residual])
        
        # Fuse all features
        combined = torch.cat(attended_features, dim=1)
        fused_output = self.fusion(combined)
        
        if return_attention:
            attention_dict = {
                'visual_to_metadata': v_to_m_attn,
                'metadata_to_visual': m_to_v_attn
            }
            return fused_output, attention_dict
        
        return fused_output


class MultimodalWeightPredictor_WithAttention(nn.Module):
    """
    Multimodal Weight Predictor using Stacked Mutual Attention Blocks.
    
    Enhanced version with:
    - Multiple stacked attention layers for hierarchical feature learning
    - Deeper regression head with skip connections
    - Improved regularization
    
    Args:
        image_encoder: Pre-configured image encoder
        metadata_encoder: Pre-configured metadata encoder
        embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
        num_attention_layers: Number of stacked attention blocks
        dropout: Dropout rate
        use_residual: Whether to use residual connections
    """
    
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_attention_layers: int = 2,  # NEW: Stacked attention layers
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        super(MultimodalWeightPredictor_WithAttention, self).__init__()
        
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        
        # Get output dimensions
        visual_dim = image_encoder.get_output_dim()
        metadata_dim = metadata_encoder.get_output_dim()
        
        # First attention block (handles dimension mismatch)
        self.attention_fusion = MutualAttentionBlock(
            visual_dim=visual_dim,
            metadata_dim=metadata_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_residual=use_residual
        )
        
        # Additional stacked attention blocks (same dimension)
        self.stacked_attention = nn.ModuleList([
            MutualAttentionBlock(
                visual_dim=embed_dim,  # After first block, dimensions match
                metadata_dim=embed_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=use_residual
            )
            for _ in range(num_attention_layers - 1)
        ])
        
        # Deeper regression head with skip connection
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),  # GELU often works better than ReLU for transformers
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(64, 1),
            nn.Softplus()  # Softplus: smooth positive output, better gradients than ReLU
        )
        
        print(f"  - Attention layers: {num_attention_layers} (stacked)")
        
        print(f"\nMultimodalWeightPredictor_WithAttention initialized:")
        print(f"  - Using Stacked Mutual Attention Fusion ({num_attention_layers} layers)")
        print(f"  - Output activation: Softplus (smooth positive constraint)")
        print(f"  - Output dim: 1 (weight prediction)")
    
    def forward(
        self,
        images: torch.Tensor,
        category_indices: torch.Tensor,
        numerical_features: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass with stacked attention-based fusion.
        
        Args:
            images: Batch of images (batch_size, 3, 224, 224)
            category_indices: Category indices (batch_size,)
            numerical_features: Numerical metadata (batch_size, num_features)
            return_attention: Whether to return attention weights
        
        Returns:
            Predicted weights (batch_size, 1)
            Optionally: attention weights dict
        """
        # Extract features
        visual_features = self.image_encoder(images)
        metadata_features = self.metadata_encoder(
            category_indices, numerical_features
        )
        
        # First attention block (handles initial dimension projection)
        if return_attention:
            fused_features, attention_weights = self.attention_fusion(
                visual_features, metadata_features, return_attention=True
            )
            all_attention = [attention_weights]
        else:
            fused_features = self.attention_fusion(
                visual_features, metadata_features, return_attention=False
            )
        
        # Apply stacked attention blocks (symmetric attention)
        for i, attn_block in enumerate(self.stacked_attention):
            # Use fused features as both visual and metadata input
            # This creates a self-refinement mechanism
            if return_attention:
                fused_features, attn_weights = attn_block(
                    fused_features, fused_features, return_attention=True
                )
                all_attention.append(attn_weights)
            else:
                fused_features = attn_block(
                    fused_features, fused_features, return_attention=False
                )
        
        # Predict weight
        weight_prediction = self.regression_head(fused_features)
        
        if return_attention:
            return weight_prediction, {'all_layers': all_attention}
        
        return weight_prediction
    
    def freeze_image_encoder(self):
        """Freeze image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("✓ Image encoder frozen")
    
    def unfreeze_image_encoder(self):
        """Unfreeze image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        print("✓ Image encoder unfrozen")
    
    def freeze_metadata_encoder(self):
        """Freeze metadata encoder parameters."""
        for param in self.metadata_encoder.parameters():
            param.requires_grad = False
        print("✓ Metadata encoder frozen")
    
    def unfreeze_metadata_encoder(self):
        """Unfreeze metadata encoder parameters."""
        for param in self.metadata_encoder.parameters():
            param.requires_grad = True
        print("✓ Metadata encoder unfrozen")
    
    def visualize_attention(
        self,
        images: torch.Tensor,
        category_indices: torch.Tensor,
        numerical_features: torch.Tensor
    ):
        """
        Visualize attention weights for interpretation.
        
        Returns attention weights showing which features the model focuses on.
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(
                images, category_indices, numerical_features, 
                return_attention=True
            )
        return attention_weights


def create_attention_model(
    num_categories: int,
    num_numerical_features: int,
    image_output_dim: int = 768,
    metadata_output_dim: int = 256,
    category_embedding_dim: int = 32,
    attention_embed_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.2,
    pretrained_image_encoder: bool = True,
    scaler=None
):
    """
    Create multimodal model with mutual attention fusion.
    
    Args:
        num_categories: Number of product categories
        num_numerical_features: Number of numerical metadata features
        image_output_dim: Output dimension of image encoder
        metadata_output_dim: Output dimension of metadata encoder
        category_embedding_dim: Dimension of category embeddings
        attention_embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
        dropout: Dropout rate
        pretrained_image_encoder: Use pretrained image encoder
        scaler: StandardScaler for numerical features
    
    Returns:
        MultimodalWeightPredictor_WithAttention instance
    """
    from .image_encoder import ImageEncoder
    from .metadata_encoder import MetadataEncoder
    
    # Create encoders
    image_encoder = ImageEncoder(
        pretrained=pretrained_image_encoder,
        freeze_backbone=False,
        output_dim=image_output_dim,
        dropout=dropout * 0.5
    )
    
    metadata_encoder = MetadataEncoder(
        num_categories=num_categories,
        category_embedding_dim=category_embedding_dim,
        num_numerical_features=num_numerical_features,
        numerical_hidden_dims=[64, 32],
        output_dim=metadata_output_dim,
        dropout=dropout * 0.5,
        scaler=scaler
    )
    
    # Create model with attention
    model = MultimodalWeightPredictor_WithAttention(
        image_encoder=image_encoder,
        metadata_encoder=metadata_encoder,
        embed_dim=attention_embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        use_residual=True
    )
    
    return model
