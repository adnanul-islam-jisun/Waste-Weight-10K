# Complete End-to-End Training Pipeline
"""
Optimized training pipeline for weight prediction using:
- Vision Transformer (ViT-B/16) for image encoding
- Metadata encoder for categorical + numerical features
- MSLE loss function (optimal for wide weight range: 3.5-3450kg)
- LOG transformation for target weights
- Progressive training (freeze → fine-tune)
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *
from features.feature_engineering import engineer_features
from config.training_config import (
    create_optimized_model,
    create_trainer_for_your_data,
    WeightPreprocessor
)
from models.loss_functions import recommend_loss_function
from dataload.data_preprocessing import prepare_data


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, train_loader, val_loader, weight_preprocessor, loss_fn,
                device, num_epochs=EPOCHS, save_dir='./checkpoints', resume=True):
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
    
    # Learning Rate Scheduler - support multiple types
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler_type = getattr(globals().get('config', type('', (), {})()), 'LR_SCHEDULER_TYPE', 'plateau')
        try:
            from config.config import LR_SCHEDULER_TYPE, COSINE_T_0, COSINE_T_MULT
            scheduler_type = LR_SCHEDULER_TYPE
        except ImportError:
            scheduler_type = 'plateau'
            COSINE_T_0 = 20
            COSINE_T_MULT = 2
        
        if scheduler_type == 'cosine_warm':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                trainer.optimizer,
                T_0=COSINE_T_0,
                T_mult=COSINE_T_MULT,
                eta_min=LR_SCHEDULER_MIN_LR
            )
            print(f"✓ Learning Rate Scheduler enabled:")
            print(f"   - Type: CosineAnnealingWarmRestarts")
            print(f"   - T_0: {COSINE_T_0} epochs (first restart)")
            print(f"   - T_mult: {COSINE_T_MULT} (period multiplier)")
            print(f"   - Min LR: {LR_SCHEDULER_MIN_LR}")
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=num_epochs,
                eta_min=LR_SCHEDULER_MIN_LR
            )
            print(f"✓ Learning Rate Scheduler enabled:")
            print(f"   - Type: CosineAnnealingLR")
            print(f"   - T_max: {num_epochs} epochs")
            print(f"   - Min LR: {LR_SCHEDULER_MIN_LR}")
        else:  # 'plateau' (default)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode='min',
                factor=LR_SCHEDULER_FACTOR,
                patience=LR_SCHEDULER_PATIENCE,
                min_lr=LR_SCHEDULER_MIN_LR
            )
            print(f"✓ Learning Rate Scheduler enabled:")
            print(f"   - Type: ReduceLROnPlateau")
            print(f"   - Factor: {LR_SCHEDULER_FACTOR} (multiply LR by this when plateau)")
            print(f"   - Patience: {LR_SCHEDULER_PATIENCE} epochs")
            print(f"   - Min LR: {LR_SCHEDULER_MIN_LR}")
    
    # Exponential Moving Average (EMA) for stable predictions
    ema_model = None
    try:
        from config.config import USE_EMA, EMA_DECAY
        if USE_EMA:
            from copy import deepcopy
            ema_model = deepcopy(model)
            ema_model.eval()
            for param in ema_model.parameters():
                param.requires_grad = False
            print(f"✓ EMA enabled with decay: {EMA_DECAY}")
    except ImportError:
        pass
    
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
    
    start_epoch = 0
    
    # Resume from checkpoint if exists
    latest_checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pt')
    if resume and os.path.exists(latest_checkpoint_path):
        print(f"\n🔄 Found checkpoint at {latest_checkpoint_path}. Resuming...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler state if exists
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load scheduler state if exists
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"   ✓ Scheduler state restored")
            
        # Load history and epoch
        if 'history' in checkpoint:
            history = checkpoint['history']
        start_epoch = checkpoint['epoch']
        
        print(f"   ✓ Resumed from epoch {start_epoch}")
        print(f"   ✓ Best validation loss so far: {history['best_val_loss']:.4f}")
    else:
        print(f"\n🆕 Starting new training session")
    
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
                    # FIX: Squeeze predictions from (batch_size, 1) to (batch_size,) to match targets shape
                    # This prevents numpy broadcasting bug: (N,1) - (N,) → (N,N) instead of (N,)
                    all_preds.extend(preds.squeeze().cpu().numpy())
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
    
    for epoch in range(start_epoch, 10):
        # Skip if already done (when resuming from > 10 epochs)
        if epoch >= 10: 
            break
            
        train_loss = run_epoch(train_loader, is_training=True)
        
        # Update EMA model after each epoch
        if ema_model is not None:
            try:
                from config.config import EMA_DECAY
            except ImportError:
                EMA_DECAY = 0.999
            with torch.no_grad():
                for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(EMA_DECAY).add_(model_param.data, alpha=1 - EMA_DECAY)
        
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
        
        # Save LATEST checkpoint (for resuming)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(save_dir, 'latest_checkpoint.pt'))
        print(f"   💾 Checkpoint saved: {save_dir}/latest_checkpoint.pt")
        
        # Step the learning rate scheduler based on validation loss
        if scheduler is not None:
            old_lr = trainer.optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = trainer.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"   📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
    
    # PHASE 2: Fine-tune entire model (remaining epochs)
    # If we resumed from > 10 epochs, we need to adjust the range
    current_epoch = max(10, start_epoch)
    remaining_epochs = num_epochs - current_epoch
    
    if remaining_epochs > 0:
        print("\n" + "-"*80)
        print(f"PHASE 2: Fine-tuning ENTIRE model ({remaining_epochs} epochs)")
        print("-"*80)
        trainer.unfreeze_all()
        
        # Reduce learning rate for fine-tuning with different scales for different components
        # Image encoder needs smaller LR (pretrained), other layers can use higher LR
        print(f"✓ Adjusting learning rates for fine-tuning:")
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            old_lr = param_group['lr']
            if i == 0:  # Image encoder (first group)
                # Image encoder: reduce by 5x (not 10x) to allow some adaptation
                param_group['lr'] = old_lr * 0.2  # 1e-4 * 0.1 * 0.2 = 2e-6
                print(f"   - Image encoder: {old_lr:.2e} → {param_group['lr']:.2e} (×0.2)")
            else:
                # Other layers: reduce by 2x to maintain learning capacity
                param_group['lr'] = old_lr * 0.5
                print(f"   - Layer group {i}: {old_lr:.2e} → {param_group['lr']:.2e} (×0.5)")
        
        for epoch in range(remaining_epochs):
            # Calculate actual epoch number
            total_epoch = current_epoch + epoch + 1
            
            # Run training and validation for this epoch
            train_loss = run_epoch(train_loader, is_training=True)
            val_loss, val_mae, val_rmse = run_epoch(val_loader, is_training=False)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            
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
            
            # Save LATEST checkpoint (for resuming)
            torch.save({
                'epoch': total_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'latest_checkpoint.pt'))
            print(f"   💾 Checkpoint saved: {save_dir}/latest_checkpoint.pt")
            
            # Step the learning rate scheduler based on validation loss
            if scheduler is not None:
                old_lr = trainer.optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = trainer.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"   📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(save_dir, f'final_model_{timestamp}.pt'))
    
    # Save training history as JSON (convert numpy types to Python native types)
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, list):
            history_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        elif hasattr(value, 'item'):
            history_serializable[key] = float(value)
        else:
            history_serializable[key] = value
    
    with open(os.path.join(save_dir, f'history_{timestamp}.json'), 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
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
    
    # Filter out samples with weight < 50 kg (minimum weight threshold)
    MIN_WEIGHT_KG = 50
    original_count = len(df)
    df = df[df['weight_in_kg'] >= MIN_WEIGHT_KG]
    removed_count = original_count - len(df)
    print(f"✓ Filtered weights < {MIN_WEIGHT_KG}kg: removed {removed_count} samples, {len(df)} remaining")

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
    
    # Pass numerical scaler to model for normalization
    model, preprocessor, loss_fn = create_optimized_model(
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(data_dict['numerical_features']),
        scaler=data_dict['numerical_scaler'],  # Pass fitted StandardScaler
        device=DEVICE
    )
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Numerical features will be normalized inside model using StandardScaler")
    
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

