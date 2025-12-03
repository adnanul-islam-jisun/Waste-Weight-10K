"""
Data Loader Module
Handles loading data and creating PyTorch datasets
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class WeightDataset(Dataset):
    """Custom Dataset for loading images and metadata."""
    
    def __init__(self, dataframe, numerical_cols, scaler, all_object_types, 
                 transform=None, weight_scaler=None, base_path=None):
        self.df = dataframe
        self.transform = transform
        self.numerical_cols = numerical_cols
        self.scaler = scaler
        self.weight_scaler = weight_scaler
        self.base_path = base_path
        
        self.type_to_idx = {t: i for i, t in enumerate(all_object_types)}
        self.num_types = len(all_object_types)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            row = self.df.iloc[[idx]]
        else:
            row = self.df.iloc[idx]

        # Image Input - use base_path for environment flexibility
        image_path = os.path.join(self.base_path, row['image_path'].iloc[0])
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            # Create a black placeholder image for missing files
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)

        # Metadata Input
        numerical_data = self.scaler.transform(row[self.numerical_cols])
        numerical_metadata = torch.tensor(numerical_data, dtype=torch.float32).squeeze(0)
        
        type_idx = self.type_to_idx[row['Type'].iloc[0]]
        type_one_hot = torch.zeros(self.num_types, dtype=torch.float32)
        type_one_hot[type_idx] = 1.0
        
        metadata_tensor = torch.cat([numerical_metadata, type_one_hot])

        # Target Value
        weight_val = row[['weight_in_kg']].values.astype(float)
        if self.weight_scaler:
            weight_val = self.weight_scaler.transform(weight_val)
        
        weight = torch.tensor(weight_val, dtype=torch.float32).squeeze()

        return {"image": image, "metadata": metadata_tensor, "weight": weight}
