"""
Sensitivity Analysis Script for Weight Prediction System
Evaluates model robustness against measurement noise in manual metadata (V_x, V_y, V_z, D_x, D_y).
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from config.config import CSV_PATH, BASE_IMAGE_PATH, DEVICE, BATCH_SIZE
from features.feature_engineering import engineer_features
from config.training_config import create_optimized_model, WeightPreprocessor
from Dataload.data_preprocessing import prepare_data, WeightPredictionDataset
from evaluate import evaluate_model


def inject_metadata_noise(df: pd.DataFrame, noise_level: float = 0.05, noise_type: str = 'gaussian', random_seed: int = 42) -> pd.DataFrame:
    """
    Injects noise into raw metadata dimensions before feature engineering.
    
    Args:
        df: Input DataFrame containing raw metadata columns ('V_x', 'V_y', 'V_z', 'D_x', 'D_y')
        noise_level: Noise magnitude (e.g., 0.05 for ±5% noise)
        noise_type: 'gaussian' or 'uniform'
        random_seed: Seed for reproducibility
    """
    np.random.seed(random_seed)
    df_noisy = df.copy()
    raw_cols = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y']
    
    for col in raw_cols:
        if col in df_noisy.columns:
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_level, size=len(df_noisy))
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_level, noise_level, size=len(df_noisy))
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            
            # Apply multiplicative noise: x_noisy = x * (1 + noise)
            df_noisy[col] = df_noisy[col] * (1.0 + noise)
            # Ensure physical dimensions remain strictly positive
            df_noisy[col] = np.maximum(df_noisy[col], 1e-4)
            
    return df_noisy


def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis on Metadata Errors")
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/best_model_phase2_20260105_005726.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--noise-type', type=str, choices=['gaussian', 'uniform'], default='gaussian',
                        help='Type of measurement noise to inject (gaussian or uniform)')
    parser.add_argument('--output-csv', type=str, default='sensitivity_results.csv',
                        help='CSV file path to save sensitivity metrics')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: METADATA MEASUREMENT NOISE EVALUATION")
    print("="*80)

    # 1. Load Raw CSV Data
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

    print(f"\n📂 Loading raw dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    df_raw = pd.read_csv(csv_path)
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    df_raw.dropna(subset=['weight_in_kg'], inplace=True)
    df_raw.fillna(0.0, inplace=True)
    df_raw = df_raw[df_raw['weight_in_kg'] >= 50]
    
    if 'weight_in_kg' in df_raw.columns and 'weight' not in df_raw.columns:
        df_raw.rename(columns={'weight_in_kg': 'weight'}, inplace=True)

    # 2. Get Clean Train/Test Splits & Fitted Normalizers
    # We run prepare_data on clean engineered features to get standard test split and scalers
    df_clean_engineered = engineer_features(df_raw)
    data_dict = prepare_data(df_clean_engineered, base_image_path)
    
    numerical_scaler = data_dict['numerical_scaler']
    numerical_features = data_dict['numerical_features']
    product_type_to_idx = data_dict['product_type_to_idx']
    weight_preprocessor = data_dict['weight_preprocessor']
    
    # 3. Load Trained Model Architecture and Weights
    device = torch.device(DEVICE)
    print(f"\n📦 Loading model checkpoint from: {args.checkpoint}")
    
    model, _, _ = create_optimized_model(
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(numerical_features),
        scaler=numerical_scaler,
        device=DEVICE
    )
    
    if not os.path.exists(args.checkpoint):
        # Fallback to search any .pt file in checkpoints if default not found
        available_pts = [os.path.join('checkpoints', f) for f in os.listdir('checkpoints') if f.endswith('.pt')]
        if available_pts:
            args.checkpoint = available_pts[0]
            print(f"⚠️ Specified checkpoint not found. Using available checkpoint: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"No checkpoint files found in checkpoints/")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model successfully loaded!")

    # Standard val/test image transformation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Define Noise Levels to Evaluate (0% up to ±15%)
    noise_levels = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15]
    sensitivity_records = []

    # Get clean test raw dataframe (unnormalized)
    test_indices = data_dict['test_loader'].dataset.df.index
    # Map back to raw base measurements
    test_raw_df = df_raw.iloc[test_indices].copy().reset_index(drop=True)

    print(f"\n🔬 Starting Sensitivity Experiments on {len(test_raw_df)} test samples across {len(noise_levels)} noise levels...")

    for noise_lvl in noise_levels:
        noise_pct = noise_lvl * 100
        print(f"\n----------------------------------------------------------------")
        print(f"⚡ Testing Noise Level: ±{noise_pct:.1f}% ({args.noise_type.upper()})")
        print(f"----------------------------------------------------------------")
        
        # Inject noise into raw measurements
        if noise_lvl == 0.0:
            test_noisy_raw = test_raw_df.copy()
        else:
            test_noisy_raw = inject_metadata_noise(test_raw_df, noise_level=noise_lvl, noise_type=args.noise_type)

        # Re-compute engineered features from noisy raw dimensions
        test_noisy_engineered = engineer_features(test_noisy_raw)
        
        # Normalize numerical features using CLEAN training scaler
        test_noisy_engineered[numerical_features] = test_noisy_engineered[numerical_features].astype(float)
        test_noisy_engineered.loc[:, numerical_features] = numerical_scaler.transform(test_noisy_engineered[numerical_features])
        
        # Ensure 'Type' is cleaned identically
        test_noisy_engineered['Type'] = test_noisy_engineered['Type'].str.strip().str.lower()
        type_corrections = {
            'grash': 'grass', 'bonet': 'bonnet', 'card board': 'cardboard',
            'cylinder track': 'cylinder_track', 'car door': 'car_door'
        }
        test_noisy_engineered['Type'] = test_noisy_engineered['Type'].replace(type_corrections)

        # Create evaluation dataset & dataloader
        noisy_test_dataset = WeightPredictionDataset(
            df=test_noisy_engineered,
            base_image_path=base_image_path,
            product_type_to_idx=product_type_to_idx,
            numerical_features=numerical_features,
            transform=test_transform,
            weight_preprocessor=weight_preprocessor
        )
        
        noisy_test_loader = DataLoader(
            noisy_test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # Evaluate model
        metrics = evaluate_model(
            model=model,
            test_loader=noisy_test_loader,
            weight_preprocessor=weight_preprocessor,
            device=device
        )

        sensitivity_records.append({
            'Noise_Level_%': f"±{noise_pct:.1f}%",
            'MAE_kg': round(metrics['mae'], 2),
            'RMSE_kg': round(metrics['rmse'], 2),
            'MAPE_%': round(metrics['mape'], 2),
            'R2_Score': round(metrics['r2'], 4),
            'Within_50kg_%': round(metrics['within_50kg'], 1),
            'Within_100kg_%': round(metrics['within_100kg'], 1)
        })

    # 5. Summarize Results
    results_df = pd.DataFrame(sensitivity_records)
    print("\n" + "="*80)
    print("📊 SENSITIVITY ANALYSIS SUMMARY RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    results_df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Results successfully saved to: {args.output_csv}")
    print("="*80)


if __name__ == "__main__":
    main()
