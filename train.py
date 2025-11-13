# Complete End-to-End Training Pipeline
"""
Optimized training pipeline for weight prediction using:
- Vision Transformer (ViT-B/16) for image encoding
- Metadata encoder for categorical + numerical features
- MSLE loss function (optimal for wide weight range: 3.5-3450kg)
- LOG transformation for target weights
- Progressive training (freeze → fine-tune)
"""

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from datetime import datetime

from config.config import *
from features.feature_engineering import engineer_features
from config.training_config import (
    create_optimized_model,
    create_trainer_for_your_data,
    WeightPreprocessor
)
from models.loss_functions import recommend_loss_function


# ============================================================================
# Custom Dataset for Multimodal Data
# ============================================================================
class WeightPredictionDataset(Dataset):
    """Dataset for multimodal weight prediction (images + metadata)"""
    
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
        
        # Load image
        image_path = os.path.join(self.base_image_path, row['image_path'])
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
        weight = row['weight']  # Column renamed from 'weight_in_kg' to 'weight' earlier
        if self.weight_preprocessor:
            weight = self.weight_preprocessor.transform(np.array([weight]))[0]
        
        return {
            'image': image,
            'category_idx': torch.tensor(product_type_idx, dtype=torch.long),
            'numerical': torch.tensor(numerical_data, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32)
        }


# ============================================================================
# Training Functions
# ============================================================================
def prepare_data(df, base_image_path, test_size=0.2, val_size=0.1, random_state=42):
    """Prepare train/val/test splits and create DataLoaders"""
    
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
    
    # Image transforms - STRONGER AUGMENTATION for small dataset
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
    
    # Create weight preprocessor (LOG transformation)
    weight_preprocessor = WeightPreprocessor()
    
    # Create datasets
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
    
    # Create DataLoaders with GPU optimization
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
    
    print(f"✓ DataLoaders created:")
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
        'num_product_types': len(product_types)
    }


def train_model(model, train_loader, val_loader, weight_preprocessor, loss_fn,
                device, num_epochs=EPOCHS, save_dir='./checkpoints'):
    """Complete training pipeline with progressive training and GPU optimization"""
    
    print("\n" + "="*80)
    print("STARTING TRAINING (GPU OPTIMIZED)")
    print("="*80)
    
    # Create trainer
    trainer = create_trainer_for_your_data(
        model=model,
        preprocessor=weight_preprocessor,
        loss_fn=loss_fn
    )
    
    # GradScaler for Automatic Mixed Precision (GPU speedup)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # CSV logging setup
    csv_path = os.path.join(save_dir, f'training_log_{timestamp}.csv')
    print(f"\n💾 Training metrics will be saved to: {csv_path}")
    
    print(f"\n📊 Training Configuration:")
    print(f"  - Model: ViT-B/16 + Multimodal Fusion")
    print(f"  - Loss: {trainer.loss_fn_name.upper()}")
    print(f"  - Preprocessing: LOG transformation")
    print(f"  - Device: {device}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Mixed Precision (AMP): {USE_AMP}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Progressive training: Freeze (10 epochs) → Fine-tune")
    print(f"  - Models save path: {save_dir}/")
    print(f"  - Best model will be: best_model_phase2_{timestamp}.pt")
    
    # Helper function for epoch training with AMP support
    def run_epoch(loader, is_training=True):
        if is_training:
            model.train()
        else:
            model.eval()
            
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.set_grad_enabled(is_training):
            for batch in tqdm(loader, desc="Training" if is_training else "Validation", leave=False):
                if is_training:
                    # Training with AMP support
                    loss = trainer.train_step(batch, scaler=scaler)
                    total_loss += loss
                else:
                    # Validation with AMP support
                    loss, preds, targets = trainer.validate_step(batch, use_amp=USE_AMP)
                    total_loss += loss
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        if is_training:
            return avg_loss
        else:
            # Convert back to original scale
            all_preds = weight_preprocessor.inverse_transform(np.array(all_preds))
            all_targets = weight_preprocessor.inverse_transform(np.array(all_targets))
            
            mae = np.mean(np.abs(all_preds - all_targets))
            rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
            
            return avg_loss, mae, rmse
    
    # PHASE 1: Train with frozen image encoder (10 epochs)
    print("\n" + "-"*80)
    print("PHASE 1: Training with FROZEN image encoder (10 epochs)")
    print("-"*80)
    trainer.freeze_image_encoder()
    
    # Track all epochs for CSV
    epoch_records = []
    
    for epoch in range(10):
        train_loss = run_epoch(train_loader, is_training=True)
        val_loss, val_mae, val_rmse = run_epoch(val_loader, is_training=False)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        
        # Record for CSV
        epoch_records.append({
            'epoch': epoch + 1,
            'phase': 'Phase1_Frozen',
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae_kg': val_mae,
            'val_rmse_kg': val_rmse,
            'is_best': False
        })
        
        # GPU memory monitoring
        gpu_mem_str = ""
        if DEVICE == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem_str = f" | GPU Mem: {allocated:.2f}/{reserved:.2f}GB"
        
        print(f"Epoch [{epoch+1}/10] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.2f}kg | "
              f"Val RMSE: {val_rmse:.2f}kg{gpu_mem_str}")
        
        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
            epoch_records[-1]['is_best'] = True
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
            }, os.path.join(save_dir, f'best_model_phase1_{timestamp}.pt'))
            print(f"  ✓ Saved best model (Phase 1) to: {save_dir}/best_model_phase1_{timestamp}.pt")
        
        # Save CSV after every epoch (instant save)
        metrics_df = pd.DataFrame(epoch_records)
        metrics_df.to_csv(csv_path, index=False)
        print(f"   💾 Metrics saved to CSV (Epoch {epoch+1}/10)")
    
    # PHASE 2: Fine-tune entire model (remaining epochs)
    remaining_epochs = num_epochs - 10
    if remaining_epochs > 0:
        print("\n" + "-"*80)
        print(f"PHASE 2: Fine-tuning ENTIRE model ({remaining_epochs} epochs)")
        print("-"*80)
        trainer.unfreeze_all()
        
        # Reduce learning rate for fine-tuning
        
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        print(f"✓ Reduced learning rate by 10x for fine-tuning")
        
        for epoch in range(remaining_epochs):
            train_loss = run_epoch(train_loader, is_training=True)
            val_loss, val_mae, val_rmse = run_epoch(val_loader, is_training=False)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            
            total_epoch = 10 + epoch + 1
            
            # Record for CSV
            epoch_records.append({
                'epoch': total_epoch,
                'phase': 'Phase2_FineTune',
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae_kg': val_mae,
                'val_rmse_kg': val_rmse,
                'is_best': False
            })
            
            # GPU memory monitoring
            gpu_mem_str = ""
            if DEVICE == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_mem_str = f" | GPU Mem: {allocated:.2f}/{reserved:.2f}GB"
            
            print(f"Epoch [{total_epoch}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f}kg | "
                  f"Val RMSE: {val_rmse:.2f}kg{gpu_mem_str}")
            
            # Save best model
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = total_epoch
                epoch_records[-1]['is_best'] = True
                torch.save({
                    'epoch': total_epoch,
                    'model_state_dict': model.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                }, os.path.join(save_dir, f'best_model_phase2_{timestamp}.pt'))
                print(f"  ✓ Saved best model (Phase 2) to: {save_dir}/best_model_phase2_{timestamp}.pt")
            
            # Save CSV after every epoch (instant save)
            metrics_df = pd.DataFrame(epoch_records)
            metrics_df.to_csv(csv_path, index=False)
            print(f"   💾 Metrics saved to CSV (Epoch {total_epoch}/{num_epochs})")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(save_dir, f'final_model_{timestamp}.pt'))
    
    # Save training history as JSON
    with open(os.path.join(save_dir, f'history_{timestamp}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final CSV save (already saved every epoch, this is redundant but ensures final state)
    metrics_df = pd.DataFrame(epoch_records)
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n💾 Final training metrics saved to CSV: {csv_path}")
    print(f"   Note: CSV was updated after every epoch automatically")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Best validation loss: {history['best_val_loss']:.4f} (epoch {history['best_epoch']})")
    print(f"✓ Models saved to: {save_dir}/")
    print(f"  - Best model (Phase 1): best_model_phase1_{timestamp}.pt")
    print(f"  - Best model (Phase 2): best_model_phase2_{timestamp}.pt ⭐ USE THIS")
    print(f"  - Final model: final_model_{timestamp}.pt")
    print(f"✓ Training log CSV: {csv_path}")
    print(f"✓ History JSON: {save_dir}/history_{timestamp}.json")
    
    return history


def evaluate_model(model, test_loader, weight_preprocessor, device):
    """Evaluate model on test set"""
    
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            category_indices = batch['category_idx'].to(device)
            numerical = batch['numerical'].to(device)
            targets = batch['weight'].to(device)
            
            # Forward pass
            predictions = model(images, category_indices, numerical).squeeze()
            
            # Inverse transform to original scale
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            
            preds_original = weight_preprocessor.inverse_transform(
                predictions.cpu().numpy()
            )
            targets_original = weight_preprocessor.inverse_transform(
                targets.cpu().numpy()
            )
            
            all_preds.extend(preds_original)
            all_targets.extend(targets_original)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mape = np.mean(np.abs((all_preds - all_targets) / all_targets)) * 100
    r2 = 1 - (np.sum((all_targets - all_preds) ** 2) / 
              np.sum((all_targets - np.mean(all_targets)) ** 2))
    
    print(f"\n📈 Test Set Results:")
    print(f"  - MAE:  {mae:.2f} kg")
    print(f"  - RMSE: {rmse:.2f} kg")
    print(f"  - MAPE: {mape:.2f}%")
    print(f"  - R²:   {r2:.4f}")
    
    # Show some predictions
    print(f"\n🎯 Sample Predictions:")
    indices = np.random.choice(len(all_preds), size=min(10, len(all_preds)), replace=False)
    for idx in indices:
        print(f"  Predicted: {all_preds[idx]:7.2f}kg | "
              f"Actual: {all_targets[idx]:7.2f}kg | "
              f"Error: {all_preds[idx] - all_targets[idx]:+7.2f}kg")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }


# ============================================================================
# Main Training Pipeline
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("WEIGHT PREDICTION - END-TO-END TRAINING PIPELINE")
    print("="*80)
    
    # 1. Load and preprocess data
    print(f"\n📂 Loading metadata from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: The file '{CSV_PATH}' was not found.")
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"✓ Successfully loaded {len(df)} records.")

    # Data cleaning
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['weight_in_kg'], inplace=True)
    df.fillna(0.0, inplace=True)
    print(f"✓ Data cleaned: {len(df)} valid records")

    # Feature engineering
    df_featured = engineer_features(df)
    print(f"✓ Feature engineering complete")
    
    # Rename weight_in_kg to weight for consistency
    if 'weight_in_kg' in df_featured.columns and 'weight' not in df_featured.columns:
        df_featured.rename(columns={'weight_in_kg': 'weight'}, inplace=True)
        print(f"✓ Renamed 'weight_in_kg' to 'weight'")

    # 2. Prepare data loaders
    data_dict = prepare_data(df_featured, BASE_IMAGE_PATH)
    
    # 3. Create optimized model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    # Use DEVICE from config.py (already optimized and detected)
    device = torch.device(DEVICE)
    print(f"✓ Using device: {device}")
    
    model, preprocessor, loss_fn = create_optimized_model(
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(data_dict['numerical_features']),
        scaler=None,  # We'll handle scaling in the model
        device=DEVICE
    )
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 4. Train model
    history = train_model(
        model=model,
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        weight_preprocessor=data_dict['weight_preprocessor'],
        loss_fn=loss_fn,
        device=device,
        num_epochs=EPOCHS,  # Use EPOCHS from config.py
        save_dir='./checkpoints'
    )
    
    # 5. Evaluate on test set
    test_results = evaluate_model(
        model=model,
        test_loader=data_dict['test_loader'],
        weight_preprocessor=data_dict['weight_preprocessor'],
        device=device
    )
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print("\n💡 Next steps:")
    print("  1. Check training curves in history JSON file")
    print("  2. Load best model for inference")
    print("  3. Increase epochs if validation loss is still decreasing")
    print("  4. Adjust hyperparameters in config/training_config.py if needed")
    print("="*80 + "\n")

