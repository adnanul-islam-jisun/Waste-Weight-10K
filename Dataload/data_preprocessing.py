"""
Data Preprocessing Module
Handles all data preparation, normalization, and DataLoader creation
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torchvision.transforms as transforms

from config.config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS
from config.training_config import WeightPreprocessor


# ============================================================================
# Custom Dataset for Multimodal Data
# ============================================================================

class WeightPredictionDataset(Dataset):
    """Dataset for weight prediction with images and metadata"""
    
    def __init__(self, df, base_image_path, product_type_to_idx, 
                 numerical_features, transform=None, weight_preprocessor=None):
        self.df = df.reset_index(drop=True)
        self.base_image_path = base_image_path
        self.product_type_to_idx = product_type_to_idx
        self.numerical_features = numerical_features
        self.transform = transform
        self.weight_preprocessor = weight_preprocessor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and transform image
        image_path = f"{self.base_image_path}/{row['image_path']}"
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, 224, 224)
        
        # Get categorical feature
        product_type_idx = self.product_type_to_idx.get(row['Type'], 0)
        
        # Get numerical features
        numerical_data = row[self.numerical_features].values.astype(np.float32)
        
        # Get target weight (apply LOG transformation)
        weight = row['weight']
        if self.weight_preprocessor:
            weight = self.weight_preprocessor.transform(np.array([weight]))[0]
        
        return {
            'image': image,
            'category_idx': torch.tensor(product_type_idx, dtype=torch.long),
            'numerical': torch.tensor(numerical_data, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32)
        }


# ============================================================================
# Data Preparation Function
# ============================================================================

def prepare_data(df, base_image_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare train/val/test splits and create DataLoaders with normalization
    
    Args:
        df: DataFrame with features and target
        base_image_path: Path to image directory
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing DataLoaders, scalers, and metadata
    """
    
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    # Create product type mapping
    product_types = sorted(df['Type'].unique().tolist())
    product_type_to_idx = {ptype: idx for idx, ptype in enumerate(product_types)}
    print(f"✓ Found {len(product_types)} unique product types: {product_types}")
    
    # Define numerical features (from feature engineering)
    numerical_features = [
        'V_x', 'V_y', 'V_z', 'D_x', 'D_y',
        'volume_proxy', 'apparent_Vx', 'apparent_Vy', 'apparent_Vz',
        'solid_angle_proxy', 'view_angle_rad'
    ]
    print(f"✓ Using {len(numerical_features)} numerical features")
    
    # Split data: train / (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), random_state=random_state, shuffle=True
    )
    
    # Split temp into val / test
    val_ratio = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio), random_state=random_state, shuffle=True
    )
    
    print(f"\n✓ Data split:")
    print(f"  - Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Weight distribution analysis
    print(f"\n📊 Weight Distribution:")
    print(f"  - Train: {train_df['weight'].min():.1f} - {train_df['weight'].max():.1f} kg (mean: {train_df['weight'].mean():.1f})")
    print(f"  - Val:   {val_df['weight'].min():.1f} - {val_df['weight'].max():.1f} kg (mean: {val_df['weight'].mean():.1f})")
    print(f"  - Test:  {test_df['weight'].min():.1f} - {test_df['weight'].max():.1f} kg (mean: {test_df['weight'].mean():.1f})")
    
    # ========================================================================
    # IMAGE TRANSFORMS
    # ========================================================================
    # STRONGER AUGMENTATION for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize larger first
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop with zoom
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Rotate ±15 degrees
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Stronger color jitter
        transforms.RandomGrayscale(p=0.1),  # 10% chance grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Random erasing (cutout)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ========================================================================
    # WEIGHT PREPROCESSING (LOG transformation)
    # ========================================================================
    weight_preprocessor = WeightPreprocessor()
    
    # ========================================================================
    # NUMERICAL FEATURE NORMALIZATION (StandardScaler)
    # ========================================================================
    print(f"\n📊 Normalizing Numerical Features (StandardScaler):")
    
    # Fit scaler on TRAIN data only (prevent data leakage)
    numerical_scaler = StandardScaler()
    numerical_scaler.fit(train_df[numerical_features])
    
    # Transform all splits using the same scaler - COPY DataFrames to avoid corruption!
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df.loc[:, numerical_features] = numerical_scaler.transform(train_df[numerical_features])
    val_df.loc[:, numerical_features] = numerical_scaler.transform(val_df[numerical_features])
    test_df.loc[:, numerical_features] = numerical_scaler.transform(test_df[numerical_features])
    
    print(f"  ✓ Features normalized to mean=0, std=1")
    print(f"  ✓ Scaler fitted on {len(train_df)} training samples")
    print(f"  ✓ Applied to train/val/test splits (DataFrames preserved)")
    
    # Display normalization stats for verification
    print(f"\n  Normalized ranges (train set):")
    for feat in numerical_features[:3]:  # Show first 3 features
        print(f"    {feat:20s}: mean={train_df[feat].mean():6.3f}, std={train_df[feat].std():6.3f}")
    print(f"    ... ({len(numerical_features)-3} more features)")
    
    # ========================================================================
    # CREATE DATASETS
    # ========================================================================
    train_dataset = WeightPredictionDataset(
        train_df, base_image_path, product_type_to_idx, 
        numerical_features, train_transform, weight_preprocessor
    )
    val_dataset = WeightPredictionDataset(
        val_df, base_image_path, product_type_to_idx,
        numerical_features, val_transform, weight_preprocessor
    )
    test_dataset = WeightPredictionDataset(
        test_df, base_image_path, product_type_to_idx,
        numerical_features, val_transform, weight_preprocessor
    )
    
    # ========================================================================
    # CREATE DATALOADERS with GPU optimization
    # ========================================================================
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None  # Prefetch batches for GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Pin memory: {PIN_MEMORY}")
    print(f"  Persistent workers: {PERSISTENT_WORKERS if NUM_WORKERS > 0 else False}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'product_type_to_idx': product_type_to_idx,
        'numerical_features': numerical_features,
        'weight_preprocessor': weight_preprocessor,
        'numerical_scaler': numerical_scaler,  # Return scaler for model
        'num_product_types': len(product_types)
    }
