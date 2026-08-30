"""
Debug script to diagnose training issues.
Checks for data leakage, transformation issues, and metric calculations.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CSV_PATH, BASE_IMAGE_PATH, DEVICE
from features.feature_engineering import engineer_features
from config.training_config import WeightPreprocessor
from dataload.data_preprocessing import prepare_data


def debug_weight_preprocessing():
    """Debug the weight preprocessing pipeline."""
    print("\n" + "="*80)
    print("DEBUG: Weight Preprocessing")
    print("="*80)
    
    # Create preprocessor
    preprocessor = WeightPreprocessor()
    
    # Test with sample weights from your data range
    test_weights = np.array([3.5, 50.0, 100.0, 500.0, 1000.0, 1500.0, 3450.0])
    
    print(f"\nOriginal weights: {test_weights}")
    
    # Transform
    transformed = preprocessor.transform(test_weights)
    print(f"After log1p:      {transformed}")
    
    # Inverse transform
    recovered = preprocessor.inverse_transform(transformed)
    print(f"After expm1:      {recovered}")
    
    # Check error
    error = np.abs(test_weights - recovered)
    print(f"Recovery error:   {error}")
    print(f"Max recovery error: {error.max():.10f}")
    
    return preprocessor


def debug_data_pipeline():
    """Debug the complete data pipeline."""
    print("\n" + "="*80)
    print("DEBUG: Data Pipeline")
    print("="*80)
    
    # Load data
    df = pd.read_csv(CSV_PATH)
    print(f"\n1. Loaded {len(df)} records")
    
    # Clean data
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['weight_in_kg'], inplace=True)
    df.fillna(0.0, inplace=True)
    
    # Rename weight
    if 'weight_in_kg' in df.columns and 'weight' not in df.columns:
        df.rename(columns={'weight_in_kg': 'weight'}, inplace=True)
    
    print(f"\n2. Weight Distribution:")
    print(f"   Min:    {df['weight'].min():.2f} kg")
    print(f"   Max:    {df['weight'].max():.2f} kg")
    print(f"   Mean:   {df['weight'].mean():.2f} kg")
    print(f"   Median: {df['weight'].median():.2f} kg")
    print(f"   Std:    {df['weight'].std():.2f} kg")
    
    # Feature engineering
    df_featured = engineer_features(df)
    
    # Check for NaN/Inf in features
    print(f"\n3. Feature Quality Check:")
    for col in df_featured.select_dtypes(include=[np.number]).columns:
        nan_count = df_featured[col].isna().sum()
        inf_count = np.isinf(df_featured[col]).sum() if df_featured[col].dtype in [np.float64, np.float32] else 0
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️ {col}: {nan_count} NaN, {inf_count} Inf")
    
    # Check new features
    print(f"\n4. New Feature Ranges:")
    new_features = ['volume_proxy', 'log_volume', 'compactness', 'dominant_dim_ratio',
                    'aspect_ratio_xy', 'aspect_ratio_xz', 'aspect_ratio_yz']
    for feat in new_features:
        if feat in df_featured.columns:
            print(f"   {feat:25s}: min={df_featured[feat].min():12.4f}, max={df_featured[feat].max():12.4f}")
    
    return df_featured


def debug_validation_metrics():
    """Debug the validation metric calculation."""
    print("\n" + "="*80)
    print("DEBUG: Validation Metric Calculation")
    print("="*80)
    
    preprocessor = WeightPreprocessor()
    
    # Simulate what happens during validation
    # Ground truth weights (original scale)
    true_weights = np.array([100.0, 500.0, 1000.0, 1500.0, 2000.0])
    
    # These get transformed when stored in dataset
    transformed_targets = preprocessor.transform(true_weights)
    
    print(f"\nSimulated Validation:")
    print(f"True weights (kg):      {true_weights}")
    print(f"Transformed targets:    {transformed_targets}")
    
    # Simulated model predictions (in log space)
    # If model predicts perfectly:
    perfect_predictions = transformed_targets.copy()
    
    # If model predicts with 10% error in log space:
    noisy_predictions = transformed_targets * 1.1
    
    print(f"\nPerfect predictions (log): {perfect_predictions}")
    print(f"Noisy predictions (log):   {noisy_predictions}")
    
    # CRITICAL: This is how train.py computes MAE
    # It inverse transforms BOTH predictions and targets
    
    # For perfect predictions:
    inv_pred_perfect = preprocessor.inverse_transform(perfect_predictions)
    inv_target = preprocessor.inverse_transform(transformed_targets)
    mae_perfect = np.mean(np.abs(inv_pred_perfect - inv_target))
    
    print(f"\n--- Perfect Prediction Case ---")
    print(f"Inverse predictions: {inv_pred_perfect}")
    print(f"Inverse targets:     {inv_target}")
    print(f"MAE (should be ~0): {mae_perfect:.4f} kg")
    
    # For noisy predictions (10% error in log space):
    inv_pred_noisy = preprocessor.inverse_transform(noisy_predictions)
    mae_noisy = np.mean(np.abs(inv_pred_noisy - inv_target))
    
    print(f"\n--- 10% Log-Space Error Case ---")
    print(f"Inverse predictions: {inv_pred_noisy}")
    print(f"Inverse targets:     {inv_target}")
    print(f"MAE: {mae_noisy:.4f} kg")
    
    # Check: what happens if predictions are in WRONG scale?
    print(f"\n--- BUG SCENARIO: Double Transform ---")
    # If model outputs in original scale but we inverse-transform anyway:
    wrong_pred = preprocessor.inverse_transform(true_weights)  # WRONG: double transform
    wrong_target = preprocessor.inverse_transform(transformed_targets)
    mae_wrong = np.mean(np.abs(wrong_pred - wrong_target))
    print(f"Wrong inverse predictions: {wrong_pred}")
    print(f"MAE (BUG - very high): {mae_wrong:.4f} kg")
    
    # Check: What about if predictions come out in wrong range?
    print(f"\n--- BUG SCENARIO: Model outputs raw weights ---")
    # If model somehow outputs raw weights (not log-transformed)
    raw_predictions = true_weights * 1.1  # 10% error
    inv_raw = preprocessor.inverse_transform(raw_predictions)  # This would be HUGE
    print(f"Raw predictions (kg):  {raw_predictions}")
    print(f"After inverse_transform (BUG): {inv_raw}")  # exp(100) is astronomical
    

def debug_model_output_scale():
    """Check what scale the model outputs are in."""
    print("\n" + "="*80)
    print("DEBUG: Model Output Scale")
    print("="*80)
    
    # Load a checkpoint and check the predictions
    checkpoint_path = "checkpoints/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Train loss: {checkpoint.get('train_loss', 'N/A')}")
        print(f"Val loss: {checkpoint.get('val_loss', 'N/A')}")
        
        # The key insight: if val_loss is ~0.025 and loss is MSLE/Huber in log space,
        # then the model is predicting in log space, which is correct.
        # MAE being 900+ kg means something is wrong with inverse_transform
        
        val_loss = checkpoint.get('val_loss', 0.025)
        print(f"\nAnalysis:")
        print(f"If val_loss = {val_loss:.4f} and using Huber loss in log space:")
        print(f"  - This corresponds to log-space error ≈ sqrt({val_loss}) = {np.sqrt(val_loss):.4f}")
        print(f"  - A log-error of {np.sqrt(val_loss):.4f} means:")
        
        log_error = np.sqrt(val_loss)
        for weight in [100, 500, 1000, 1500]:
            log_weight = np.log1p(weight)
            pred_log = log_weight + log_error
            pred_weight = np.expm1(pred_log)
            print(f"    Weight {weight} kg → prediction error ≈ {abs(pred_weight - weight):.1f} kg")
    else:
        print("No checkpoint found at", checkpoint_path)


if __name__ == "__main__":
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "TRAINING DEBUG SCRIPT" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run all debug functions
    preprocessor = debug_weight_preprocessing()
    df = debug_data_pipeline()
    debug_validation_metrics()
    debug_model_output_scale()
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("\n💡 Key Questions to Answer:")
    print("1. Is the model outputting in LOG space? (It should be)")
    print("2. Are targets transformed correctly before training?")
    print("3. Is inverse_transform called correctly for both pred and target?")
    print("4. Are there any NaN/Inf values in features?")
