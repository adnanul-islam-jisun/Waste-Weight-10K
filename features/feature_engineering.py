# Feature Engineering
import numpy as np
import pandas as pd


def engineer_features(df):
    """Creates advanced, physics-informed features."""
    print("Performing advanced feature engineering...")
    new_df = df.copy()
    epsilon = 1e-6

    # Simple Features
    new_df['volume_proxy'] = new_df['V_x'] * new_df['V_y'] * new_df['V_z']
    
    # Physics-Informed Features
    new_df['apparent_Vx'] = new_df['V_x'] / (new_df['D_x'] + epsilon)
    new_df['apparent_Vy'] = new_df['V_y'] / (new_df['D_x'] + epsilon)
    new_df['apparent_Vz'] = new_df['V_z'] / (new_df['D_x'] + epsilon)
    
    new_df['solid_angle_proxy'] = (new_df['V_x'] * new_df['V_y']) / (new_df['D_x']**2 + epsilon)
    new_df['view_angle_rad'] = np.arctan2(new_df['D_y'], new_df['D_x'])
    
    print("New features created: apparent dimensions, solid angle proxy, and viewing angle.")
    return new_df

