"""
Metadata-Only Traditional Machine Learning System for Weight Prediction
Trains and benchmarks multiple traditional ML algorithms (XGBoost, LightGBM, CatBoost, Random Forest, 
ExtraTrees, Gradient Boosting, MLP, Ridge) exclusively on physical metadata and engineered features.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Scikit-learn models & tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

# Gradient Boosting frameworks
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Project imports
from config.config import CSV_PATH, BASE_IMAGE_PATH
from features.feature_engineering import engineer_features


def resolve_csv_path() -> str:
    """Resolves data path ensuring compatibility across user environments."""
    csv_path = CSV_PATH
    if not os.path.exists(csv_path):
        try:
            user_name = os.getlogin()
        except Exception:
            user_name = "aislam"
        alt_csv = csv_path.replace('/home/asiful/', f'/home/{user_name}/')
        if os.path.exists(alt_csv):
            csv_path = alt_csv
        elif os.path.exists('/home/aislam/adnan_workspace/Dataset/disaster_data/waste_dataset/image.csv'):
            csv_path = '/home/aislam/adnan_workspace/Dataset/disaster_data/waste_dataset/image.csv'
        else:
            raise FileNotFoundError(f"Dataset CSV not found at {CSV_PATH}")
    return csv_path


def load_and_preprocess_data(csv_path: str):
    """Loads raw dataset, cleans product types, applies feature engineering, and splits data."""
    print(f"📂 Loading dataset from: {csv_path}")
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
        
    # Clean product types
    df_raw['Type'] = df_raw['Type'].astype(str).str.strip().str.lower()
    type_corrections = {
        'grash': 'grass', 'bonet': 'bonnet', 'card board': 'cardboard',
        'cylinder track': 'cylinder_track', 'car door': 'car_door'
    }
    df_raw['Type'] = df_raw['Type'].replace(type_corrections)
    
    # Feature engineering
    df_featured = engineer_features(df_raw)
    
    # Train (70%) / Val (15%) / Test (15%) split (matching project standard)
    train_df, temp_df = train_test_split(df_featured, test_size=0.30, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)
    
    print(f"✓ Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics on actual weight scale (kg)."""
    # Prevent negative or zero weight predictions
    y_pred = np.maximum(y_pred, 1e-3)
    
    errors = np.abs(y_pred - y_true)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mape = np.mean(errors / (y_true + 1e-8)) * 100
    r2 = 1.0 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    within_50kg = np.mean(errors <= 50) * 100
    within_100kg = np.mean(errors <= 100) * 100
    within_200kg = np.mean(errors <= 200) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'within_50kg': within_50kg,
        'within_100kg': within_100kg,
        'within_200kg': within_200kg
    }


def main():
    print("\n" + "="*80)
    print("TRADITIONAL MACHINE LEARNING METADATA-ONLY BENCHMARK")
    print("="*80)
    
    csv_path = resolve_csv_path()
    train_df, val_df, test_df = load_and_preprocess_data(csv_path)
    
    # Define feature columns
    categorical_cols = ['Type']
    numerical_cols = [
        'V_x', 'V_y', 'V_z', 'D_x', 'D_y',
        'log_volume', 'log_surface_area', 'max_dimension', 'log_max_dimension', 'log_geo_mean_dim',
        'aspect_ratio_xy', 'aspect_ratio_xz', 'aspect_ratio_yz', 'compactness',
        'flatness', 'elongation', 'sphericity', 'log_vol_surface_ratio',
        'log_distance', 'log_apparent_volume', 'view_angle_rad', 'depth_ratio',
        'volume_compactness', 'surface_sphericity', 'size_distance_interaction'
    ]
    
    feature_cols = categorical_cols + numerical_cols
    
    # Target values (Apply Log transform matching MSLE standard: log1p(weight))
    y_train = np.log1p(train_df['weight'].values)
    y_val = np.log1p(val_df['weight'].values)
    y_test = np.log1p(test_df['weight'].values)
    
    y_test_kg = test_df['weight'].values
    
    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Fit preprocessor on train data
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])
    
    print(f"✓ Transformed Feature Matrix Shape: {X_train.shape}")
    
    # Define traditional ML regressors
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
        ),
        'CatBoost': cb.CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=6, 
            random_seed=42, verbose=0
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
        'MLP (Neural Tabular)': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=300, 
            activation='relu', random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    benchmark_results = []
    trained_model_objs = {}
    
    print(f"\n🚀 Training and evaluating {len(models)} machine learning models...\n")
    
    for name, model in models.items():
        t0 = time.time()
        # Train model on log-transformed targets
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        # Predict on test set
        y_pred_log = model.predict(X_test)
        # Convert back to kilograms: expm1(log1p_pred)
        y_pred_kg = np.expm1(y_pred_log)
        
        metrics = evaluate_predictions(y_test_kg, y_pred_kg)
        metrics['model'] = name
        metrics['fit_time_sec'] = round(fit_time, 3)
        
        benchmark_results.append(metrics)
        trained_model_objs[name] = model
        
        print(f"  [{name:<20s}] MAE: {metrics['mae']:>7.2f} kg | RMSE: {metrics['rmse']:>7.2f} kg | MAPE: {metrics['mape']:>6.2f}% | R²: {metrics['r2']:>6.4f} | Within ±50kg: {metrics['within_50kg']:>5.1f}%")
        
    # Build dataframe and sort by MAE ascending
    results_df = pd.DataFrame(benchmark_results)
    results_df = results_df.sort_values(by='mae', ascending=True).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("🏆 FINAL METADATA-ONLY TRADITIONAL ML BENCHMARK RESULTS")
    print("="*80)
    
    display_cols = ['model', 'mae', 'rmse', 'mape', 'r2', 'within_50kg', 'within_100kg', 'fit_time_sec']
    display_df = results_df[display_cols].copy()
    display_df.columns = ['Model', 'MAE (kg)', 'RMSE (kg)', 'MAPE (%)', 'R² Score', '±50kg (%)', '±100kg (%)', 'Time (s)']
    print(display_df.to_string(index=False))
    
    # Save CSV benchmark output
    output_csv = "metadata_ml_benchmark_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n💾 Full benchmark results saved to: {output_csv}")
    
    # Save best model to checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    best_model_name = results_df.iloc[0]['model']
    best_model_obj = trained_model_objs[best_model_name]
    best_checkpoint_path = "checkpoints/best_metadata_ml_model.joblib"
    
    joblib.dump({
        'model': best_model_obj,
        'preprocessor': preprocessor,
        'model_name': best_model_name,
        'metrics': results_df.iloc[0].to_dict(),
        'feature_cols': feature_cols
    }, best_checkpoint_path)
    
    print(f"🌟 Best Metadata Model ({best_model_name}) saved to: {best_checkpoint_path}")
    print("="*80)


if __name__ == "__main__":
    main()
