"""
Metadata Encoder Module: Heterogeneous Data Processor
Objective: Convert mixed-type metadata into a unified dense feature vector.

This module processes both categorical (product name) and numerical (volume, distance, 
view_angle) features through parallel branches and combines them into a single representation.
"""

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Optional, Dict, List, Tuple


class MetadataEncoder(nn.Module):
    """
    Heterogeneous Metadata Encoder.
    
    Processes categorical and numerical metadata in parallel branches:
    - Categorical Sub-Encoder: Embedding layer for product names
    - Numerical Sub-Encoder: MLP for numerical features (volume, distance, view_angle)
    
    Args:
        num_categories (int): Number of unique product categories/names
        category_embedding_dim (int): Dimension of category embeddings. Default: 32
        num_numerical_features (int): Number of numerical input features
        numerical_hidden_dims (List[int]): Hidden layer dimensions for numerical MLP
        output_dim (int): Dimension of the final output feature vector
        dropout (float): Dropout rate for regularization. Default: 0.1
        scaler (Optional[StandardScaler]): Pre-fitted scaler for numerical features
    """
    
    def __init__(
        self,
        num_categories: int,
        category_embedding_dim: int = 32,
        num_numerical_features: int = 6,  # V_x, V_y, V_z, D_x, D_y, D_z or derived features
        numerical_hidden_dims: Optional[List[int]] = None,
        output_dim: int = 256,
        dropout: float = 0.1,
        scaler: Optional[StandardScaler] = None
    ):
        super(MetadataEncoder, self).__init__()
        
        self.num_categories = num_categories
        self.category_embedding_dim = category_embedding_dim
        self.num_numerical_features = num_numerical_features
        self.output_dim = output_dim
        
        # Store scaler for numerical feature normalization
        self.scaler = scaler
        self.register_buffer('scaler_mean', None)
        self.register_buffer('scaler_scale', None)
        
        if scaler is not None:
            # Store scaler parameters as buffers for model saving/loading
            self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
            self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
        
        # ===== CATEGORICAL SUB-ENCODER =====
        # Trainable embedding layer for product names/categories
        # Maps integer indices to dense vectors
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=category_embedding_dim
        )
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.category_embedding.weight)
        
        # ===== NUMERICAL SUB-ENCODER =====
        # MLP for processing normalized numerical features
        if numerical_hidden_dims is None:
            numerical_hidden_dims = [64, 32]
        
        numerical_layers = []
        in_dim = num_numerical_features
        
        for hidden_dim in numerical_hidden_dims:
            numerical_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.numerical_mlp = nn.Sequential(*numerical_layers)
        numerical_output_dim = numerical_hidden_dims[-1]
        
        # ===== FUSION LAYER =====
        # Combine categorical and numerical embeddings
        combined_dim = category_embedding_dim + numerical_output_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, output_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        print(f"MetadataEncoder initialized:")
        print(f"  - Categories: {num_categories}, Embedding dim: {category_embedding_dim}")
        print(f"  - Numerical features: {num_numerical_features}")
        print(f"  - Output dim: {output_dim}")
    
    def normalize_numerical_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize numerical features using stored scaler parameters.
        
        Args:
            x (torch.Tensor): Raw numerical features of shape (batch_size, num_features)
        
        Returns:
            torch.Tensor: Normalized features
        """
        if self.scaler_mean is not None and self.scaler_scale is not None:
            # Apply standardization: (x - mean) / scale
            return (x - self.scaler_mean) / self.scaler_scale
        else:
            # No normalization if scaler not provided
            return x
    
    def forward(
        self, 
        category_indices: torch.Tensor, 
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the metadata encoder.
        
        Args:
            category_indices (torch.Tensor): Integer indices for categories 
                                            Shape: (batch_size,) or (batch_size, 1)
            numerical_features (torch.Tensor): Numerical metadata features
                                              Shape: (batch_size, num_numerical_features)
        
        Returns:
            torch.Tensor: Unified metadata feature vector M_features 
                         Shape: (batch_size, output_dim)
        """
        # Ensure category_indices is 1D
        if len(category_indices.shape) > 1:
            category_indices = category_indices.squeeze(-1)
        
        # ===== CATEGORICAL BRANCH =====
        # Map category indices to dense embeddings
        # Output: (batch_size, category_embedding_dim)
        category_embed = self.category_embedding(category_indices.long())
        
        # ===== NUMERICAL BRANCH =====
        # Normalize numerical features
        numerical_normalized = self.normalize_numerical_features(numerical_features)
        
        # Pass through MLP
        # Output: (batch_size, numerical_output_dim)
        numerical_embed = self.numerical_mlp(numerical_normalized)
        
        # ===== CONCATENATION =====
        # Combine categorical and numerical embeddings
        # Shape: (batch_size, combined_dim)
        combined_features = torch.cat([category_embed, numerical_embed], dim=1)
        
        # ===== FUSION =====
        # Final projection to output dimension
        # Output: (batch_size, output_dim)
        metadata_features = self.fusion_layer(combined_features)
        
        return metadata_features
    
    def get_output_dim(self) -> int:
        """Returns the dimension of the output feature vector."""
        return self.output_dim
    
    def update_scaler(self, scaler: StandardScaler):
        """
        Update the internal scaler parameters.
        
        Args:
            scaler (StandardScaler): Fitted StandardScaler instance
        """
        self.scaler = scaler
        self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
        
        # Move to same device as model
        if next(self.parameters()).is_cuda:
            self.scaler_mean = self.scaler_mean.cuda()
            self.scaler_scale = self.scaler_scale.cuda()


class MetadataEncoderConfig:
    """Configuration class for MetadataEncoder."""
    
    def __init__(
        self,
        num_categories: int,
        category_embedding_dim: int = 32,
        num_numerical_features: int = 6,
        numerical_hidden_dims: Optional[List[int]] = None,
        output_dim: int = 256,
        dropout: float = 0.1,
        scaler: Optional[StandardScaler] = None
    ):
        self.num_categories = num_categories
        self.category_embedding_dim = category_embedding_dim
        self.num_numerical_features = num_numerical_features
        self.numerical_hidden_dims = numerical_hidden_dims or [64, 32]
        self.output_dim = output_dim
        self.dropout = dropout
        self.scaler = scaler
    
    def create_encoder(self) -> MetadataEncoder:
        """Create a MetadataEncoder instance with this configuration."""
        return MetadataEncoder(
            num_categories=self.num_categories,
            category_embedding_dim=self.category_embedding_dim,
            num_numerical_features=self.num_numerical_features,
            numerical_hidden_dims=self.numerical_hidden_dims,
            output_dim=self.output_dim,
            dropout=self.dropout,
            scaler=self.scaler
        )


# Utility function to create metadata encoder from dataframe
def create_metadata_encoder_from_data(
    dataframe,
    category_column: str = 'Type',
    numerical_columns: Optional[List[str]] = None,
    category_embedding_dim: int = 32,
    output_dim: int = 256,
    fit_scaler: bool = True
) -> Tuple[MetadataEncoder, StandardScaler, Dict[str, int]]:
    """
    Create a MetadataEncoder from a pandas DataFrame.
    
    Args:
        dataframe: Pandas DataFrame containing metadata
        category_column (str): Name of the categorical column. Default: 'Type'
        numerical_columns (List[str]): Names of numerical columns
        category_embedding_dim (int): Embedding dimension for categories
        output_dim (int): Output feature dimension
        fit_scaler (bool): Whether to fit a new scaler. Default: True
    
    Returns:
        Tuple containing:
            - MetadataEncoder: Configured encoder
            - StandardScaler: Fitted scaler for numerical features
            - Dict[str, int]: Mapping from category names to indices
    """
    import pandas as pd
    
    if numerical_columns is None:
        # Default numerical columns based on common metadata
        numerical_columns = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'view_angle_rad',
                           'volume_proxy', 'apparent_Vx', 'apparent_Vy', 'apparent_Vz']
        # Filter to only include columns that exist in the dataframe
        numerical_columns = [col for col in numerical_columns if col in dataframe.columns]
    
    # Extract unique categories
    unique_categories = sorted(dataframe[category_column].unique().tolist())
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    num_categories = len(unique_categories)
    
    # Fit scaler on numerical features
    scaler = None
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(dataframe[numerical_columns].values)
    
    # Create encoder
    encoder = MetadataEncoder(
        num_categories=num_categories,
        category_embedding_dim=category_embedding_dim,
        num_numerical_features=len(numerical_columns),
        numerical_hidden_dims=[64, 32],
        output_dim=output_dim,
        dropout=0.1,
        scaler=scaler
    )
    
    print(f"Created MetadataEncoder for {num_categories} categories")
    print(f"Numerical features: {numerical_columns}")
    
    return encoder, scaler, category_to_idx
