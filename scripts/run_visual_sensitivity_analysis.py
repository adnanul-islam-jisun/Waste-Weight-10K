"""
Visual Sensitivity Analysis Script for Full Multimodal Weight Prediction Model
Evaluates full model robustness under visual perturbations (brightness, color jitter, rotation).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CSV_PATH, BASE_IMAGE_PATH, DEVICE, BATCH_SIZE
from features.feature_engineering import engineer_features
from config.training_config import create_optimized_model, WeightPreprocessor
from dataload.data_preprocessing import prepare_data, WeightPredictionDataset
from scripts.evaluate import evaluate_model


def resolve_paths():
    """Resolves data paths ensuring compatibility across user environments."""
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


def get_perturbed_transform(perturbation_type: str):
    """
    Constructs torchvision transform pipeline for specific visual perturbations.
    """
    base_size = (224, 224)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if perturbation_type == 'clean':
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.ToTensor(),
            norm
        ])
    elif perturbation_type == 'brightness_dark':
        # Darken image by 40%
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.ColorJitter(brightness=(0.6, 0.6)),
            transforms.ToTensor(),
            norm
        ])
    elif perturbation_type == 'brightness_bright':
        # Brighten image by 40%
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.ColorJitter(brightness=(1.4, 1.4)),
            transforms.ToTensor(),
            norm
        ])
    elif perturbation_type == 'color_jitter':
        # Strong color, contrast, saturation, hue change
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            norm
        ])
    elif perturbation_type == 'rotation_15deg':
        # Rotate ±15 degrees
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.RandomRotation(degrees=(15, 15)),
            transforms.ToTensor(),
            norm
        ])
    elif perturbation_type == 'rotation_30deg':
        # Rotate ±30 degrees
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.RandomRotation(degrees=(30, 30)),
            transforms.ToTensor(),
            norm
        ])
    elif perturbation_type == 'rotation_90deg':
        # Rotate 90 degrees
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            norm
        ])
    elif perturbation_type == 'combined_extreme':
        # Rotation (30deg) + Brightness (1.3x) + Color Jitter (hue 0.15)
        return transforms.Compose([
            transforms.Resize(base_size),
            transforms.RandomRotation(degrees=(30, 30)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.ToTensor(),
            norm
        ])
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def main():
    parser = argparse.ArgumentParser(description="Visual Sensitivity Evaluation for Full Multimodal Model")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_phase2_20260105_005726.pt',
                        help='Path to trained full model checkpoint')
    parser.add_argument('--output-csv', type=str, default='visual_sensitivity_results.csv',
                        help='CSV file path to save visual sensitivity metrics')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("VISUAL SENSITIVITY ANALYSIS: FULL MULTIMODAL MODEL")
    print("Evaluating Image Perturbations (Brightness, Color, Rotation)")
    print("="*80)

    # 1. Load Data
    csv_path, base_image_path = resolve_paths()
    print(f"\n📂 Loading dataset from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    df_raw.dropna(subset=['weight_in_kg'], inplace=True)
    df_raw.fillna(0.0, inplace=True)
    df_raw = df_raw[df_raw['weight_in_kg'] >= 50].copy()
    
    if 'weight_in_kg' in df_raw.columns and 'weight' not in df_raw.columns:
        df_raw.rename(columns={'weight_in_kg': 'weight'}, inplace=True)

    # Feature engineering & Data Preparation
    df_engineered = engineer_features(df_raw)
    data_dict = prepare_data(df_engineered, base_image_path)

    numerical_scaler = data_dict['numerical_scaler']
    numerical_features = data_dict['numerical_features']
    product_type_to_idx = data_dict['product_type_to_idx']
    weight_preprocessor = data_dict['weight_preprocessor']

    # Load Full Multimodal Model
    device = torch.device(DEVICE)
    print(f"\n📦 Loading full multimodal checkpoint from: {args.checkpoint}")
    
    model, _, _ = create_optimized_model(
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(numerical_features),
        scaler=numerical_scaler,
        device=DEVICE
    )

    if not os.path.exists(args.checkpoint):
        available_pts = [os.path.join('checkpoints', f) for f in os.listdir('checkpoints') if f.endswith('.pt')]
        if available_pts:
            args.checkpoint = available_pts[0]
            print(f"⚠️ Specified checkpoint not found. Using available checkpoint: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"No model checkpoints found in checkpoints/")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Full Multimodal Model loaded successfully!")

    # Define Image Perturbation Experiments
    perturbation_experiments = [
        ('Clean Baseline', 'clean'),
        ('Darkened (-40%)', 'brightness_dark'),
        ('Brightened (+40%)', 'brightness_bright'),
        ('Color & Hue Shift', 'color_jitter'),
        ('Rotation (±15°)', 'rotation_15deg'),
        ('Rotation (±30°)', 'rotation_30deg'),
        ('Rotation (90°)', 'rotation_90deg'),
        ('Combined Perturbations', 'combined_extreme'),
    ]

    test_df_normalized = data_dict['test_loader'].dataset.df

    visual_sensitivity_records = []

    print(f"\n🔬 Evaluating {len(test_df_normalized)} test samples across {len(perturbation_experiments)} image conditions...")

    for name, p_type in perturbation_experiments:
        print(f"\n----------------------------------------------------------------")
        print(f"⚡ Testing Image Condition: {name.upper()}")
        print(f"----------------------------------------------------------------")
        
        transform = get_perturbed_transform(p_type)

        # Create test dataset with specific image transform
        perturbed_test_dataset = WeightPredictionDataset(
            df=test_df_normalized,
            base_image_path=base_image_path,
            product_type_to_idx=product_type_to_idx,
            numerical_features=numerical_features,
            transform=transform,
            weight_preprocessor=weight_preprocessor
        )

        perturbed_test_loader = DataLoader(
            perturbed_test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        # Evaluate model
        metrics = evaluate_model(
            model=model,
            test_loader=perturbed_test_loader,
            weight_preprocessor=weight_preprocessor,
            device=device
        )

        visual_sensitivity_records.append({
            'Image_Condition': name,
            'MAE_kg': round(metrics['mae'], 2),
            'RMSE_kg': round(metrics['rmse'], 2),
            'MAPE_%': round(metrics['mape'], 2),
            'R2_Score': round(metrics['r2'], 4),
            'Within_50kg_%': round(metrics['within_50kg'], 1),
            'Within_100kg_%': round(metrics['within_100kg'], 1)
        })

    # Display Summary Results
    results_df = pd.DataFrame(visual_sensitivity_records)
    print("\n" + "="*80)
    print("📊 VISUAL SENSITIVITY SUMMARY RESULTS (FULL MULTIMODAL MODEL)")
    print("="*80)
    print(results_df.to_string(index=False))

    results_df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Results successfully saved to: {args.output_csv}")
    print("="*80)


if __name__ == "__main__":
    main()
