"""
Image-Only Weight Prediction Model (No Metadata)
Trains and evaluates a Vision Transformer (ViT-B/16) regression model using pure RGB images.
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from config.config import CSV_PATH, BASE_IMAGE_PATH, DEVICE, BATCH_SIZE
from models.image_encoder import ImageEncoder
from models.architecture_variants import ImageOnlyPredictor
from config.training_config import WeightPreprocessor
from Dataload.data_preprocessing import prepare_data
from evaluate import evaluate_model


class ImageOnlyDataset(Dataset):
    """Dataset for Image-Only weight prediction (ignores all metadata)."""
    
    def __init__(self, df, base_image_path, transform=None, weight_preprocessor=None):
        self.df = df.reset_index(drop=True)
        self.base_image_path = base_image_path
        self.transform = transform
        self.weight_preprocessor = weight_preprocessor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.base_image_path, str(row['image_path']))
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            image = torch.zeros(3, 224, 224)

        weight = row['weight']
        if self.weight_preprocessor:
            weight = self.weight_preprocessor.transform(np.array([weight]))[0]

        return {
            'image': image,
            'category_idx': torch.tensor(0, dtype=torch.long),  # dummy
            'numerical': torch.zeros(1, dtype=torch.float32),   # dummy
            'weight': torch.tensor(weight, dtype=torch.float32)
        }


def resolve_paths():
    """Resolves data paths automatically."""
    csv_path = CSV_PATH
    base_image_path = BASE_IMAGE_PATH
    if not os.path.exists(csv_path):
        try:
            user_name = os.getlogin()
        except Exception:
            user_name = "aislam"
        alt_csv = csv_path.replace('/home/asiful/', f'/home/{user_name}/')
        alt_base = base_image_path.replace('/home/asiful/', f'/home/{user_name}/')
        if os.path.exists(alt_csv):
            csv_path = alt_csv
            base_image_path = alt_base
        elif os.path.exists('/home/aislam/adnan_workspace/Dataset/disaster_data/waste_dataset/image.csv'):
            csv_path = '/home/aislam/adnan_workspace/Dataset/disaster_data/waste_dataset/image.csv'
            base_image_path = '/home/aislam/adnan_workspace/Dataset/disaster_data/waste_dataset'
    return csv_path, base_image_path


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Image-Only Weight Prediction Model")
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_image_only_model.pt', help='Save/Load checkpoint path')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate checkpoint without training')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("IMAGE-ONLY WEIGHT PREDICTION SYSTEM (NO METADATA)")
    print("="*80)

    csv_path, base_image_path = resolve_paths()
    print(f"\n📂 Loading dataset from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    
    cols_to_numeric = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_numeric:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
    df_raw.dropna(subset=['weight_in_kg'], inplace=True)
    df_raw.fillna(0.0, inplace=True)
    df_raw = df_raw[df_raw['weight_in_kg'] >= 50].copy()
    
    if 'weight_in_kg' in df_raw.columns and 'weight' not in df_raw.columns:
        df_raw.rename(columns={'weight_in_kg': 'weight'}, inplace=True)

    # Split train/val/test
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df_raw, test_size=0.30, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

    print(f"✓ Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    weight_preprocessor = WeightPreprocessor()

    train_dataset = ImageOnlyDataset(train_df, base_image_path, transform=train_transform, weight_preprocessor=weight_preprocessor)
    val_dataset = ImageOnlyDataset(val_df, base_image_path, transform=val_test_transform, weight_preprocessor=weight_preprocessor)
    test_dataset = ImageOnlyDataset(test_df, base_image_path, transform=val_test_transform, weight_preprocessor=weight_preprocessor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Create Image-Only Model Architecture
    device = torch.device(DEVICE)
    image_encoder = ImageEncoder(model_name='vit_b_16', pretrained=True)
    model = ImageOnlyPredictor(image_encoder=image_encoder, dropout=0.3).to(device)

    os.makedirs('checkpoints', exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if not args.eval_only:
        print(f"\n🚀 Training Image-Only Model on {DEVICE} for {args.epochs} epochs...\n")
        best_val_mae = float('inf')

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                images = batch['image'].to(device)
                targets = batch['weight'].to(device)

                optimizer.zero_grad()
                preds = model(images).squeeze()
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_dataset)

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    targets = batch['weight'].to(device)
                    preds = model(images).squeeze()

                    preds_kg = weight_preprocessor.inverse_transform(preds.cpu().numpy())
                    targets_kg = weight_preprocessor.inverse_transform(targets.cpu().numpy())

                    val_preds.extend(preds_kg)
                    val_targets.extend(targets_kg)

            val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_targets)))
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f} kg")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_mae': val_mae}, args.checkpoint)
                print(f"  ★ Best Model Saved! Val MAE: {val_mae:.2f} kg")

    # Load best checkpoint for testing
    if os.path.exists(args.checkpoint):
        print(f"\n📦 Loading best checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    print("\n" + "="*80)
    print("📊 EVALUATING IMAGE-ONLY MODEL ON TEST SET")
    print("="*80)
    
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        weight_preprocessor=weight_preprocessor,
        device=device
    )

    print("\n" + "="*80)
    print("🎉 IMAGE-ONLY EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
