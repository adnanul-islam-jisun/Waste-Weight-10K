# Training Pipeline
import pandas as pd
import os
from config.config import CSV_PATH, BASE_IMAGE_PATH
from features.feature_engineering import engineer_features


if __name__ == "__main__":
    # Load and preprocess data
    print(f"Loading metadata from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: The file '{CSV_PATH}' was not found.")
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded {len(df)} records.")

    # Data cleaning
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['weight_in_kg'], inplace=True)
    df.fillna(0.0, inplace=True)

    # Filter out records with weight less than 50kg
    initial_record_count = len(df)
    # df = df[df['weight_in_kg'] >= 50].copy()
    # print(f"Filtered out {initial_record_count - len(df)} records with weight < 50kg. Remaining records: {len(df)}")

    # Feature engineering
    df_featured = engineer_features(df)

    # Display sample data
    print("\n--- Sample data with features ---")
    print(df_featured.sample(n=5, random_state=42))

    # Get object types
    object_types = sorted(df_featured['Type'].unique().tolist())
    print(f"\nFound {len(object_types)} unique object types: {object_types}")

