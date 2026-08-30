"""
Standalone Test Evaluation Script
Loads the best model and evaluates on test set
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *
from features.feature_engineering import engineer_features
from config.training_config import create_optimized_model, WeightPreprocessor
from dataload.data_preprocessing import prepare_data


def evaluate_model(model, test_loader, weight_preprocessor, device):
    """Evaluate model on test set with detailed metrics"""
    
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
            
            # Handle single sample case
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            
            # Inverse transform to original scale (kg)
            preds_original = weight_preprocessor.inverse_transform(
                predictions.cpu().numpy()
            )
            targets_original = weight_preprocessor.inverse_transform(
                targets.cpu().numpy()
            )
            
            all_preds.extend(preds_original)
            all_targets.extend(targets_original)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # ========================================================================
    # Calculate Metrics
    # ========================================================================
    
    # Basic metrics
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mape = np.mean(np.abs((all_preds - all_targets) / (all_targets + 1e-8))) * 100
    r2 = 1 - (np.sum((all_targets - all_preds) ** 2) / 
              np.sum((all_targets - np.mean(all_targets)) ** 2))
    
    # Additional metrics
    median_ae = np.median(np.abs(all_preds - all_targets))
    max_error = np.max(np.abs(all_preds - all_targets))
    min_error = np.min(np.abs(all_preds - all_targets))
    std_error = np.std(all_preds - all_targets)
    
    # Percentage within thresholds
    errors = np.abs(all_preds - all_targets)
    within_50kg = np.mean(errors <= 50) * 100
    within_100kg = np.mean(errors <= 100) * 100
    within_200kg = np.mean(errors <= 200) * 100
    
    # Per-weight-range analysis
    weight_ranges = [
        (0, 100, "Light (0-100kg)"),
        (100, 500, "Medium (100-500kg)"),
        (500, 1000, "Heavy (500-1000kg)"),
        (1000, 2000, "Very Heavy (1000-2000kg)"),
        (2000, 5000, "Ultra Heavy (2000+kg)")
    ]
    
    # ========================================================================
    # Print Results
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 TEST SET RESULTS")
    print("="*80)
    
    print(f"\n🎯 PRIMARY METRICS:")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  MAE:   {mae:>10.2f} kg              │")
    print(f"  │  RMSE:  {rmse:>10.2f} kg              │")
    print(f"  │  MAPE:  {mape:>10.2f} %               │")
    print(f"  │  R²:    {r2:>10.4f}                  │")
    print(f"  └─────────────────────────────────────┘")
    
    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"  - Median Absolute Error: {median_ae:.2f} kg")
    print(f"  - Max Error:             {max_error:.2f} kg")
    print(f"  - Min Error:             {min_error:.2f} kg")
    print(f"  - Std of Errors:         {std_error:.2f} kg")
    
    print(f"\n✅ ACCURACY THRESHOLDS:")
    print(f"  - Within ±50 kg:  {within_50kg:5.1f}% of samples")
    print(f"  - Within ±100 kg: {within_100kg:5.1f}% of samples")
    print(f"  - Within ±200 kg: {within_200kg:5.1f}% of samples")
    
    print(f"\n📊 PER-WEIGHT-RANGE ANALYSIS:")
    print(f"  {'Range':<25} {'Count':>8} {'MAE':>10} {'MAPE':>10}")
    print(f"  {'-'*55}")
    
    for low, high, name in weight_ranges:
        mask = (all_targets >= low) & (all_targets < high)
        if np.sum(mask) > 0:
            range_mae = np.mean(np.abs(all_preds[mask] - all_targets[mask]))
            range_mape = np.mean(np.abs((all_preds[mask] - all_targets[mask]) / (all_targets[mask] + 1e-8))) * 100
            count = np.sum(mask)
            print(f"  {name:<25} {count:>8} {range_mae:>10.2f} {range_mape:>9.1f}%")
    
    # Sample predictions
    print(f"\n🎯 SAMPLE PREDICTIONS (Random 15):")
    print(f"  {'Predicted':>12} {'Actual':>12} {'Error':>12} {'Rel.Err':>10}")
    print(f"  {'-'*50}")
    
    indices = np.random.choice(len(all_preds), size=min(15, len(all_preds)), replace=False)
    for idx in sorted(indices, key=lambda i: all_targets[i]):
        pred = all_preds[idx]
        actual = all_targets[idx]
        error = pred - actual
        rel_err = (error / actual) * 100 if actual != 0 else 0
        print(f"  {pred:>10.2f}kg {actual:>10.2f}kg {error:>+10.2f}kg {rel_err:>+9.1f}%")
    
    # Worst predictions
    print(f"\n⚠️ WORST PREDICTIONS (Top 5 errors):")
    worst_indices = np.argsort(errors)[-5:][::-1]
    for idx in worst_indices:
        pred = all_preds[idx]
        actual = all_targets[idx]
        error = pred - actual
        print(f"  Predicted: {pred:>8.2f}kg | Actual: {actual:>8.2f}kg | Error: {error:>+8.2f}kg")
    
    # Best predictions
    print(f"\n✅ BEST PREDICTIONS (Top 5 closest):")
    best_indices = np.argsort(errors)[:5]
    for idx in best_indices:
        pred = all_preds[idx]
        actual = all_targets[idx]
        error = pred - actual
        print(f"  Predicted: {pred:>8.2f}kg | Actual: {actual:>8.2f}kg | Error: {error:>+8.2f}kg")
    
    print("\n" + "="*80)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'median_ae': median_ae,
        'max_error': max_error,
        'within_50kg': within_50kg,
        'within_100kg': within_100kg,
        'within_200kg': within_200kg,
        'predictions': all_preds,
        'targets': all_targets
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/final_model_20251203_230924.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to CSV')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("WEIGHT PREDICTION - TEST EVALUATION")
    print("="*80)
    
    # ========================================================================
    # 1. Load and preprocess data
    # ========================================================================
    print(f"\n📂 Loading data from: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} records")
    
    # Data cleaning
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['weight_in_kg'], inplace=True)
    df.fillna(0.0, inplace=True)
    
    # Filter weights
    MIN_WEIGHT_KG = 50
    df = df[df['weight_in_kg'] >= MIN_WEIGHT_KG]
    print(f"✓ {len(df)} samples after filtering (weight >= {MIN_WEIGHT_KG}kg)")
    
    # Feature engineering
    df_featured = engineer_features(df)
    
    # Rename weight column
    if 'weight_in_kg' in df_featured.columns and 'weight' not in df_featured.columns:
        df_featured.rename(columns={'weight_in_kg': 'weight'}, inplace=True)
    
    # ========================================================================
    # 2. Prepare data loaders
    # ========================================================================
    data_dict = prepare_data(df_featured, BASE_IMAGE_PATH)
    
    # ========================================================================
    # 3. Create model and load checkpoint
    # ========================================================================
    print(f"\n📦 Loading model from: {args.checkpoint}")
    
    device = torch.device(DEVICE)
    
    # Create model architecture
    model, preprocessor, _ = create_optimized_model(
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(data_dict['numerical_features']),
        scaler=data_dict['numerical_scaler'],
        device=DEVICE
    )
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    val_mae = checkpoint.get('val_mae', None)
    if val_mae is not None:
        print(f"✓ Checkpoint val_mae: {val_mae:.2f} kg")
    else:
        print(f"✓ Checkpoint val_mae: N/A")
    
    # ========================================================================
    # 4. Evaluate on test set
    # ========================================================================
    results = evaluate_model(
        model=model,
        test_loader=data_dict['test_loader'],
        weight_preprocessor=data_dict['weight_preprocessor'],
        device=device
    )
    
    # ========================================================================
    # 5. Save predictions (optional)
    # ========================================================================
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'actual_kg': results['targets'],
            'predicted_kg': results['predictions'],
            'error_kg': results['predictions'] - results['targets'],
            'abs_error_kg': np.abs(results['predictions'] - results['targets']),
            'rel_error_pct': ((results['predictions'] - results['targets']) / results['targets']) * 100
        })
        
        save_path = 'checkpoints/test_predictions.csv'
        predictions_df.to_csv(save_path, index=False)
        print(f"\n💾 Predictions saved to: {save_path}")
    
    print("\n" + "="*80)
    print("🎉 EVALUATION COMPLETE!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
