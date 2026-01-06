"""
Model Architecture Ablation Study - Main Orchestration Script
Runs all ablation experiments automatically and saves results.

Usage:
    # Run all experiments
    python run_ablation_study.py --all
    
    # Run specific experiments
    python run_ablation_study.py --experiments 1,2,4
    
    # Resume from checkpoint
    python run_ablation_study.py --resume
    
    # Debug mode (single batch per epoch)
    python run_ablation_study.py --debug --experiments 1
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import *
from ablation_study_config import *
from ablation_utils import *
from models.architecture_variants import create_ablation_model
from models.loss_functions import recommend_loss_function
from config.training_config import WeightPreprocessor, create_trainer_for_your_data
from Dataload.data_preprocessing import prepare_data
from features.feature_engineering import engineer_features


# ============================================================================
# TRAINING FUNCTION FOR ABLATION
# ============================================================================

def train_ablation_model(
    model,
    train_loader,
    val_loader,
    weight_preprocessor,
    loss_fn,
    device,
    num_epochs,
    exp_dir,
    exp_config,
    debug_mode=False
):
    """
    Train model for ablation study (simplified version of main training).
    
    Returns:
        history: Training history dictionary
        training_time: Total training time in seconds
    """
    from models.multimodal_fusion import MultimodalTrainer
    
    print("\n" + "=" * 80)
    print("TRAINING ABLATION MODEL")
    print("=" * 80)
    
    # Create trainer
    trainer = create_trainer_for_your_data(
        model=model,
        preprocessor=weight_preprocessor,
        loss_fn=loss_fn
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
        'epoch_times': [],
    }
    
    # Start timing
    train_start_time = time.time()
    
    # Progressive training settings
    freeze_epochs = min(10, num_epochs // 3)  # First 1/3 or 10 epochs, whichever is smaller
    
    # Helper function for epoch training
    def run_epoch(loader, is_training=True):
        if is_training:
            model.train()
        else:
            model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        # Debug mode: only one batch
        if debug_mode:
            loader = [next(iter(loader))]
        
        with torch.set_grad_enabled(is_training):
            for batch in tqdm(loader, desc="Training" if is_training else "Validation", leave=False):
                if is_training:
                    loss = trainer.train_step(batch)
                    total_loss += loss
                else:
                    loss, preds, targets = trainer.validate_step(batch, use_amp=USE_AMP)
                    total_loss += loss
                    all_preds.extend(preds.squeeze().cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(loader) if not debug_mode else total_loss
        
        if is_training:
            return avg_loss
        else:
            # Convert back to original scale
            all_preds = weight_preprocessor.inverse_transform(np.array(all_preds))
            all_targets = weight_preprocessor.inverse_transform(np.array(all_targets))
            
            mae = np.mean(np.abs(all_preds - all_targets))
            rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
            
            return avg_loss, mae, rmse
    
    print(f"\n📊 Training Configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Freeze Phase: {freeze_epochs} epochs")
    print(f"  - Fine-tune Phase: {num_epochs - freeze_epochs} epochs")
    print(f"  - Debug Mode: {debug_mode}")
    
    # PHASE 1: Train with frozen image encoder (if applicable)
    if freeze_epochs > 0 and exp_config.get('use_image', False):
        print(f"\n🔒 PHASE 1: Frozen Image Encoder ({freeze_epochs} epochs)")
        trainer.freeze_image_encoder()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Switch to fine-tuning after freeze_epochs
        if epoch == freeze_epochs and exp_config.get('use_image', False):
            print(f"\n🔓 PHASE 2: Fine-tuning All Layers ({num_epochs - freeze_epochs} epochs)")
            trainer.unfreeze_all()
            # Reduce learning rate
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= 0.2
        
        # Run training and validation
        train_loss = run_epoch(train_loader, is_training=True)
        val_loss, val_mae, val_rmse = run_epoch(val_loader, is_training=False)
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_rmse'].append(float(val_rmse))
        history['epoch_times'].append(float(epoch_time))
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.2f}kg | "
              f"Val RMSE: {val_rmse:.2f}kg | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = float(val_loss)
            history['best_epoch'] = epoch + 1
            
            if SAVE_CHECKPOINTS:
                checkpoint_dir = get_checkpoint_dir(os.path.basename(exp_dir))
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
        
        # Save training log CSV after EVERY epoch (real-time tracking)
        training_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_mae': history['val_mae'],
            'val_rmse': history['val_rmse'],
            'epoch_time_s': history['epoch_times'],
        })
        training_df.to_csv(os.path.join(exp_dir, 'training_log.csv'), index=False)
        
        # Early stopping check
        if epoch - history['best_epoch'] >= ABLATION_EARLY_STOPPING_PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch+1} (no improvement for {ABLATION_EARLY_STOPPING_PATIENCE} epochs)")
            break
    
    total_train_time = time.time() - train_start_time
    
    print(f"\n✓ Training complete in {total_train_time:.0f}s ({total_train_time/60:.1f} min)")
    print(f"✓ Best epoch: {history['best_epoch']} with val_loss: {history['best_val_loss']:.4f}")
    
    # Save training log CSV
    training_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_mae': history['val_mae'],
        'val_rmse': history['val_rmse'],
        'epoch_time_s': history['epoch_times'],
    })
    training_df.to_csv(os.path.join(exp_dir, 'training_log.csv'), index=False)
    
    return history, total_train_time


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_ablation_model(model, test_loader, weight_preprocessor, device, weight_ranges):
    """Evaluate model on test set and return comprehensive metrics."""
    
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    # Measure inference time
    inference_start = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch.get('image')
            category_indices = batch.get('category_idx')
            numerical = batch.get('numerical')
            targets = batch['weight']
            
            # Move to device
            if images is not None:
                images = images.to(device)
            if category_indices is not None:
                category_indices = category_indices.to(device)
            if numerical is not None:
                numerical = numerical.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(images, category_indices, numerical).squeeze()
            
            # Handle single sample case
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            
            # Inverse transform
            preds_original = weight_preprocessor.inverse_transform(predictions.cpu().numpy())
            targets_original = weight_preprocessor.inverse_transform(targets.cpu().numpy())
            
            all_preds.extend(preds_original)
            all_targets.extend(targets_original)
    
    inference_time_total = time.time() - inference_start
    inference_time_per_sample = (inference_time_total / len(all_preds)) * 1000  # ms
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = calculate_metrics(all_preds, all_targets, weight_ranges)
    metrics['inference_time'] = inference_time_per_sample
    
    print(f"\n📈 Test Set Results:")
    print(f"  - MAE:  {metrics['mae']:.2f} kg")
    print(f"  - RMSE: {metrics['rmse']:.2f} kg")
    print(f"  - MAPE: {metrics['mape']:.2f}%")
    print(f"  - R²:   {metrics['r2']:.4f}")
    print(f"  - Inference: {metrics['inference_time']:.2f} ms/sample")
    
    return metrics, all_preds, all_targets


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    exp_key: str,
    exp_info: dict,
    data_dict: dict,
    device: str,
    debug_mode: bool = False
):
    """Run a single ablation experiment."""
    
    print("\n" + "=" * 100)
    print(f"🔬 EXPERIMENT: {exp_info['name']}")
    print(f"   {exp_info['description']}")
    print("=" * 100)
    
    exp_config = exp_info['config']
    exp_dir = get_experiment_dir(exp_key)
    
    # Save configuration
    save_experiment_config(exp_dir, exp_config, exp_info)
    
    # Clear GPU memory before creating model
    clear_gpu_memory()
    
    # Create model
    print("\n🏗️ Creating Model...")
    model = create_ablation_model(
        exp_config=exp_config,
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(data_dict['numerical_features']),
        scaler=data_dict['numerical_scaler'],
        device=device
    )
    
    # Calculate model stats
    model_stats = calculate_model_stats(model)
    print(f"  ✓ Model Parameters: {model_stats['params_millions']:.1f}M")
    print(f"  ✓ Model Size: {model_stats['model_size_mb']:.1f} MB")
    
    # Create loss function and preprocessor
    # Get sample weights from dataloader to determine loss function
    sample_weights = []
    for batch in data_dict['train_loader']:
        sample_weights.extend(batch['weight'].numpy())
        if len(sample_weights) >= 1000:  # Sample 1000 weights
            break
    sample_weights = np.array(sample_weights[:1000])
    
    # Transform back to original scale for loss function recommendation
    weight_preprocessor = data_dict['weight_preprocessor']
    original_weights = weight_preprocessor.inverse_transform(sample_weights)
    
    loss_fn = recommend_loss_function(
        weight_min=float(original_weights.min()),
        weight_max=float(original_weights.max()),
        has_outliers=True,
        outlier_percentage=5.0
    )
    
    # Train model
    history, train_time = train_ablation_model(
        model=model,
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        weight_preprocessor=weight_preprocessor,
        loss_fn=loss_fn,
        device=device,
        num_epochs=ABLATION_EPOCHS,
        exp_dir=exp_dir,
        exp_config=exp_config,
        debug_mode=debug_mode
    )
    
    # Measure GPU memory
    gpu_memory = measure_gpu_memory(device)
    
    # Evaluate on test set
    test_metrics, predictions, targets = evaluate_ablation_model(
        model=model,
        test_loader=data_dict['test_loader'],
        weight_preprocessor=weight_preprocessor,
        device=device,
        weight_ranges=WEIGHT_RANGES
    )
    
    # Combine all metrics
    final_metrics = {
        **test_metrics,
        **model_stats,
        'train_time': train_time,
        'train_time_per_epoch': train_time / len(history['train_loss']),
        'gpu_memory_mb': gpu_memory,
        'final_train_loss': history['train_loss'][-1],
        'best_val_loss': history['best_val_loss'],
        'best_epoch': history['best_epoch'],
    }
    
    # Save results
    save_experiment_results(
        exp_dir=exp_dir,
        metrics=final_metrics,
        predictions=predictions,
        targets=targets,
        training_history=history
    )
    
    # Print summary
    print_experiment_summary(exp_info['name'], final_metrics, exp_config)
    
    # Clear GPU memory after experiment
    del model
    clear_gpu_memory()
    
    return final_metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Model Architecture Ablation Study')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--experiments', type=str, help='Comma-separated experiment IDs (e.g., 1,2,4)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--debug', action='store_true', help='Debug mode (single batch per epoch)')
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.all:
        exp_list = list(EXPERIMENTS.keys())
    elif args.experiments:
        exp_list = get_experiment_list(args.experiments)
    else:
        print("Please specify --all or --experiments")
        return
    
    # Load progress if resuming
    progress = load_progress() if args.resume else {"completed": [], "failed": []}
    
    # Create directories
    create_ablation_directories()
    
    print("\n" + "=" * 100)
    print("🚀 MODEL ARCHITECTURE ABLATION STUDY")
    print("=" * 100)
    print(f"\n📋 Experiments to run: {len(exp_list)}")
    for i, exp_key in enumerate(exp_list, 1):
        status = " ✓ (completed)" if exp_key in progress['completed'] else ""
        print(f"  {i}. {EXPERIMENTS[exp_key]['name']}{status}")
    
    # Load and prepare data (once for all experiments)
    print("\n" + "=" * 100)
    print("📂 LOADING AND PREPARING DATA")
    print("=" * 100)
    
    # Load CSV
    print(f"\n📂 Loading metadata from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} records")
    
    # Clean data
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['weight_in_kg'], inplace=True)
    df.fillna(0.0, inplace=True)
    
    # Filter minimum weight
    MIN_WEIGHT_KG = 50
    df = df[df['weight_in_kg'] >= MIN_WEIGHT_KG]
    print(f"✓ Filtered to {len(df)} samples (>={MIN_WEIGHT_KG}kg)")
    
    # Feature engineering
    df_featured = engineer_features(df)
    if 'weight_in_kg' in df_featured.columns and 'weight' not in df_featured.columns:
        df_featured.rename(columns={'weight_in_kg': 'weight'}, inplace=True)
    
    # Prepare data loaders with ablation batch size
    data_dict = prepare_data(df_featured, BASE_IMAGE_PATH, batch_size=ABLATION_BATCH_SIZE)
    print(f"✓ Data prepared: {len(data_dict['train_loader'].dataset)} train, "
          f"{len(data_dict['val_loader'].dataset)} val, "
          f"{len(data_dict['test_loader'].dataset)} test")
    print(f"  Batch size: {ABLATION_BATCH_SIZE} (from ablation_study_config.py)")
    
    # Run experiments
    device = torch.device(DEVICE)
    
    for i, exp_key in enumerate(exp_list, 1):
        # Skip if already completed
        if args.resume and exp_key in progress['completed']:
            print(f"\n⏭️ Skipping {exp_key} (already completed)")
            continue
        
        try:
            print(f"\n\n{'='*100}")
            print(f"EXPERIMENT {i}/{len(exp_list)}")
            print(f"{'='*100}")
            
            # Run experiment
            run_single_experiment(
                exp_key=exp_key,
                exp_info=EXPERIMENTS[exp_key],
                data_dict=data_dict,
                device=device,
                debug_mode=args.debug
            )
            
            # Mark as completed
            progress['completed'].append(exp_key)
            save_progress(progress)
            
            print(f"\n✅ Experiment {exp_key} completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error in experiment {exp_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            progress['failed'].append(exp_key)
            save_progress(progress)
            
            # Continue with next experiment
            continue
    
    # Generate summary report
    print("\n" + "=" * 100)
    print("📊 GENERATING SUMMARY REPORT")
    print("=" * 100)
    
    results_df = aggregate_all_results(ABLATION_BASE_DIR, EXPERIMENTS)
    save_summary_report(results_df, ABLATION_BASE_DIR, format='both')
    
    print("\n" + "=" * 100)
    print("🎉 ABLATION STUDY COMPLETE!")
    print("=" * 100)
    print(f"\nResults saved to: {ABLATION_BASE_DIR}/")
    print(f"  - Summary CSV: summary_report.csv")
    print(f"  - Summary JSON: summary_report.json")
    print(f"  - LaTeX Table: summary_report.tex")
    print(f"\n💡 Next steps:")
    print(f"  1. Generate visualizations: python visualize_ablation_results.py")
    print(f"  2. Review results in: {ABLATION_BASE_DIR}/")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
