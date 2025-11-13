"""
Data Analysis Script: Check Weight Distribution and Outliers
This script analyzes your weight dataset to understand:
- Weight range and distribution
- Outliers detection
- Data quality issues
- Recommended loss function and preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CSV_PATH


def analyze_weight_distribution(df):
    """Analyze weight distribution and detect outliers."""
    
    print("\n" + "="*80)
    print("WEIGHT DATA ANALYSIS")
    print("="*80)
    
    weights = df['weight_in_kg'].dropna()
    
    # Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 80)
    print(f"   Count:       {len(weights):,} samples")
    print(f"   Min weight:  {weights.min():.2f} kg")
    print(f"   Max weight:  {weights.max():.2f} kg")
    print(f"   Mean:        {weights.mean():.2f} kg")
    print(f"   Median:      {weights.median():.2f} kg")
    print(f"   Std Dev:     {weights.std():.2f} kg")
    print(f"   Range:       {weights.max() - weights.min():.2f} kg")
    
    # Percentiles
    print("\n2. PERCENTILES")
    print("-" * 80)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(weights, p)
        print(f"   {p:2d}th percentile: {val:.2f} kg")
    
    # Outlier detection using IQR method
    print("\n3. OUTLIER DETECTION (IQR Method)")
    print("-" * 80)
    Q1 = weights.quantile(0.25)
    Q3 = weights.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = weights[(weights < lower_bound) | (weights > upper_bound)]
    outlier_percentage_iqr = (len(outliers_iqr) / len(weights)) * 100
    
    print(f"   Q1 (25th):       {Q1:.2f} kg")
    print(f"   Q3 (75th):       {Q3:.2f} kg")
    print(f"   IQR:             {IQR:.2f} kg")
    print(f"   Lower bound:     {lower_bound:.2f} kg")
    print(f"   Upper bound:     {upper_bound:.2f} kg")
    print(f"   Outliers (IQR):  {len(outliers_iqr):,} ({outlier_percentage_iqr:.2f}%)")
    
    # Z-score method
    print("\n4. OUTLIER DETECTION (Z-Score Method, threshold=3)")
    print("-" * 80)
    z_scores = np.abs(stats.zscore(weights))
    outliers_z = weights[z_scores > 3]
    outlier_percentage_z = (len(outliers_z) / len(weights)) * 100
    
    print(f"   Outliers (Z>3):  {len(outliers_z):,} ({outlier_percentage_z:.2f}%)")
    
    # Distribution shape
    print("\n5. DISTRIBUTION ANALYSIS")
    print("-" * 80)
    skewness = stats.skew(weights)
    kurtosis = stats.kurtosis(weights)
    
    print(f"   Skewness:        {skewness:.4f}", end="")
    if abs(skewness) < 0.5:
        print(" (Fairly symmetric)")
    elif skewness > 0:
        print(" (Right-skewed/long tail on right)")
    else:
        print(" (Left-skewed/long tail on left)")
    
    print(f"   Kurtosis:        {kurtosis:.4f}", end="")
    if abs(kurtosis) < 0.5:
        print(" (Normal-like)")
    elif kurtosis > 0:
        print(" (Heavy-tailed, more outliers)")
    else:
        print(" (Light-tailed, fewer outliers)")
    
    # Normality test
    print("\n6. NORMALITY TEST (Shapiro-Wilk)")
    print("-" * 80)
    if len(weights) <= 5000:
        stat, p_value = stats.shapiro(weights)
        print(f"   Test statistic:  {stat:.6f}")
        print(f"   P-value:         {p_value:.6f}")
        if p_value > 0.05:
            print("   Result:          Data IS normally distributed (p > 0.05)")
        else:
            print("   Result:          Data is NOT normally distributed (p < 0.05)")
    else:
        # Use sample for large datasets
        sample = weights.sample(n=5000, random_state=42)
        stat, p_value = stats.shapiro(sample)
        print(f"   Test statistic:  {stat:.6f} (on 5000 sample)")
        print(f"   P-value:         {p_value:.6f}")
        if p_value > 0.05:
            print("   Result:          Data IS normally distributed (p > 0.05)")
        else:
            print("   Result:          Data is NOT normally distributed (p < 0.05)")
    
    # Log transformation test
    print("\n7. LOG TRANSFORMATION TEST")
    print("-" * 80)
    log_weights = np.log1p(weights)
    log_skewness = stats.skew(log_weights)
    log_kurtosis = stats.kurtosis(log_weights)
    
    print(f"   Original skewness: {skewness:.4f}")
    print(f"   Log skewness:      {log_skewness:.4f}")
    print(f"   Improvement:       {abs(skewness) - abs(log_skewness):.4f}")
    
    if abs(log_skewness) < abs(skewness):
        print("   ✓ Log transformation IMPROVES distribution")
    else:
        print("   ✗ Log transformation does NOT improve distribution")
    
    # Data quality checks
    print("\n8. DATA QUALITY CHECKS")
    print("-" * 80)
    
    # Check for zeros
    zeros = len(weights[weights == 0])
    print(f"   Zero weights:    {zeros} ({zeros/len(weights)*100:.2f}%)")
    
    # Check for negative (shouldn't exist but good to check)
    negatives = len(weights[weights < 0])
    print(f"   Negative weights: {negatives}")
    
    # Check for unrealistic values
    very_small = len(weights[weights < 1])
    very_large = len(weights[weights > 2000])
    print(f"   Very small (<1kg): {very_small} ({very_small/len(weights)*100:.2f}%)")
    print(f"   Very large (>2000kg): {very_large} ({very_large/len(weights)*100:.2f}%)")
    
    return {
        'weights': weights,
        'outlier_percentage_iqr': outlier_percentage_iqr,
        'outlier_percentage_z': outlier_percentage_z,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'is_normal': p_value > 0.05,
        'log_improves': abs(log_skewness) < abs(skewness),
        'min': weights.min(),
        'max': weights.max(),
        'mean': weights.mean(),
        'median': weights.median(),
        'std': weights.std()
    }


def recommend_configuration(stats):
    """Recommend model configuration based on data analysis."""
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR YOUR DATA")
    print("="*80)
    
    # Loss function recommendation
    print("\n1. RECOMMENDED LOSS FUNCTION")
    print("-" * 80)
    
    has_outliers = stats['outlier_percentage_iqr'] > 5
    wide_range = (stats['max'] - stats['min']) > 100
    is_log_normal = stats['log_improves'] and stats['skewness'] > 1
    
    if has_outliers and wide_range and is_log_normal:
        print("   PRIMARY:   MSLE (Mean Squared Log Error)")
        print("   REASON:    - Wide weight range (20-1500kg)")
        print("              - Log transformation improves distribution")
        print("              - Handles outliers well")
        print("   BACKUP:    Huber Loss (delta=10.0)")
    elif has_outliers:
        print("   PRIMARY:   Huber Loss")
        print("   REASON:    - Significant outliers detected")
        print("              - Robust to measurement errors")
        print("   BACKUP:    MAE (Mean Absolute Error)")
    elif wide_range:
        print("   PRIMARY:   MSLE (Mean Squared Log Error)")
        print("   REASON:    - Wide weight range")
        print("   BACKUP:    MAPE (Mean Absolute Percentage Error)")
    else:
        print("   PRIMARY:   MSE (Mean Squared Error)")
        print("   REASON:    - Well-behaved data")
        print("   BACKUP:    Huber Loss")
    
    # Preprocessing recommendations
    print("\n2. PREPROCESSING RECOMMENDATIONS")
    print("-" * 80)
    
    if stats['log_improves']:
        print("   ✓ Use LOG transformation before training")
        print("     - Transform: log1p(weight)")
        print("     - After prediction: expm1(prediction)")
    else:
        print("   ✓ Use STANDARD SCALING")
        print("     - Transform: (weight - mean) / std")
        print("     - After prediction: prediction * std + mean")
    
    print(f"\n   Suggested scaling parameters:")
    print(f"   - Mean:  {stats['mean']:.2f} kg")
    print(f"   - Std:   {stats['std']:.2f} kg")
    print(f"   - Min:   {stats['min']:.2f} kg")
    print(f"   - Max:   {stats['max']:.2f} kg")
    
    # Model configuration
    print("\n3. MODEL CONFIGURATION")
    print("-" * 80)
    print("   Image Encoder:   ViT-Base/16 (vit_b_16)")
    print("   Pretrained:      True (ImageNet)")
    print("   Freeze backbone: False (allow fine-tuning)")
    print("   Output dim:      768")
    
    # Loss function parameters
    print("\n4. LOSS FUNCTION PARAMETERS")
    print("-" * 80)
    
    if has_outliers and wide_range and is_log_normal:
        print("   Loss: MSLE")
        print("   No hyperparameters to tune")
    elif has_outliers:
        suggested_delta = stats['std'] * 0.3
        print(f"   Loss: Huber")
        print(f"   Suggested delta: {suggested_delta:.2f}")
        print(f"   (Based on std deviation)")
    
    # Training recommendations
    print("\n5. TRAINING RECOMMENDATIONS")
    print("-" * 80)
    print("   Learning rate:   1e-4 (start)")
    print("   Batch size:      32-64 (depending on GPU)")
    print("   Epochs:          50-100")
    print("   Optimizer:       AdamW")
    print("   Weight decay:    1e-5")
    
    if has_outliers:
        print("\n   Data Augmentation:")
        print("   - Consider outlier removal (keep 95-99th percentile)")
        print("   - Or use robust loss function (recommended)")
    
    # Expected performance
    print("\n6. EXPECTED PERFORMANCE")
    print("-" * 80)
    expected_mae_pct = 5 if stats['is_normal'] else 8
    expected_mae = stats['mean'] * (expected_mae_pct / 100)
    print(f"   Expected MAE:    {expected_mae:.2f} kg ({expected_mae_pct}% of mean)")
    print(f"   Expected MAPE:   {expected_mae_pct}-{expected_mae_pct+5}%")
    
    return {
        'loss_function': 'msle' if (has_outliers and wide_range and is_log_normal) else 'huber',
        'use_log_transform': stats['log_improves'],
        'huber_delta': stats['std'] * 0.3 if has_outliers else 1.0
    }


def main():
    """Main analysis function."""
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "WEIGHT DATA ANALYSIS & RECOMMENDATIONS" + " "*20 + "║")
    print("╚" + "="*78 + "╝")
    
    # Load data
    print(f"\nLoading data from: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: File not found: {CSV_PATH}")
        print("Please update the CSV_PATH in config/config.py")
        return
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ Loaded {len(df):,} records")
        
        # Check for weight column
        if 'weight_in_kg' not in df.columns:
            print(f"\n❌ ERROR: 'weight_in_kg' column not found")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        # Analyze data
        stats = analyze_weight_distribution(df)
        
        # Get recommendations
        config = recommend_configuration(stats)
        
        # Save configuration
        print("\n" + "="*80)
        print("CONFIGURATION TO USE")
        print("="*80)
        print(f"""
# Recommended configuration for your dataset:

from models import create_default_image_encoder, MultimodalTrainer

# Image encoder (ViT-B/16)
image_encoder = create_default_image_encoder(
    output_dim=768,
    pretrained=True,
    freeze_backbone=False
)

# Trainer with recommended loss
trainer = MultimodalTrainer(
    model=model,
    device='cuda',
    loss_fn='{config['loss_function']}',
    huber_delta={config['huber_delta']:.2f},
    learning_rate=1e-4,
    weight_decay=1e-5
)

# Preprocessing
use_log_transform = {config['use_log_transform']}
mean_weight = {stats['mean']:.2f}
std_weight = {stats['std']:.2f}
        """)
        
        print("="*80)
        print("\n✓ Analysis complete! Use the configuration above for training.")
        print("  See models/LOSS_FUNCTIONS_GUIDE.md for more details on loss functions.")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
