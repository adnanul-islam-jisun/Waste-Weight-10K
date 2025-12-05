# Feature Engineering
import numpy as np
import pandas as pd


def engineer_features(df):
    """
    Creates physics-informed features for weight prediction.
    
    Weight prediction requires features that capture:
    1. SIZE/VOLUME: Primary determinant of weight (Weight ∝ Density × Volume)
    2. SHAPE: Affects density distribution and structural weight
    3. PERSPECTIVE: Camera distance affects apparent size
    4. INTERACTION: Cross-feature interactions for complex relationships
    
    Feature Selection Rationale:
    - Include raw dimensions (V_x, V_y, V_z) as they directly relate to size
    - Use log transforms for wide-range values
    - Add derived shape features for density estimation
    - Add interaction features for non-linear relationships
    """
    print("Performing feature engineering...")
    new_df = df.copy()
    epsilon = 1e-6

    # ========================================================================
    # 1. SIZE FEATURES (Most important for weight prediction!)
    # ========================================================================
    
    # Raw volume (critical for weight - directly proportional to mass)
    volume = new_df['V_x'] * new_df['V_y'] * new_df['V_z']
    new_df['log_volume'] = np.log1p(volume)
    
    # Surface area proxy (correlates with shell/container weight)
    surface_area = 2 * (new_df['V_x'] * new_df['V_y'] + 
                        new_df['V_y'] * new_df['V_z'] + 
                        new_df['V_x'] * new_df['V_z'])
    new_df['log_surface_area'] = np.log1p(surface_area)
    
    # Dominant dimension (largest dimension - useful for long objects)
    new_df['max_dimension'] = new_df[['V_x', 'V_y', 'V_z']].max(axis=1)
    new_df['log_max_dimension'] = np.log1p(new_df['max_dimension'])
    
    # Geometric mean of dimensions (better central tendency for volume)
    geo_mean = (new_df['V_x'] * new_df['V_y'] * new_df['V_z']) ** (1/3)
    new_df['log_geo_mean_dim'] = np.log1p(geo_mean)
    
    # ========================================================================
    # 2. SHAPE DESCRIPTORS
    # ========================================================================
    
    # Aspect ratios (shape indicators - different shapes have different densities)
    new_df['aspect_ratio_xy'] = new_df['V_x'] / (new_df['V_y'] + epsilon)
    new_df['aspect_ratio_xz'] = new_df['V_x'] / (new_df['V_z'] + epsilon)
    new_df['aspect_ratio_yz'] = new_df['V_y'] / (new_df['V_z'] + epsilon)  # NEW
    
    # Compactness: how cube-like (0 = elongated, 1 = cube)
    # Cube-like objects often have different density patterns
    max_dim = new_df[['V_x', 'V_y', 'V_z']].max(axis=1)
    min_dim = new_df[['V_x', 'V_y', 'V_z']].min(axis=1)
    mid_dim = new_df[['V_x', 'V_y', 'V_z']].median(axis=1)  # NEW
    new_df['compactness'] = min_dim / (max_dim + epsilon)
    
    # Flatness ratio (NEW): flat objects vs 3D objects
    new_df['flatness'] = min_dim / (mid_dim + epsilon)
    
    # Elongation ratio (NEW): how stretched the object is
    new_df['elongation'] = max_dim / (mid_dim + epsilon)
    
    # Sphericity proxy: how close to a sphere (affects packing density)
    # Sphericity = (π^(1/3) * (6V)^(2/3)) / A
    new_df['sphericity'] = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / (surface_area + epsilon)
    
    # Volume-to-surface ratio (NEW): indicator of solid vs hollow
    new_df['vol_surface_ratio'] = volume / (surface_area + epsilon)
    new_df['log_vol_surface_ratio'] = np.log1p(new_df['vol_surface_ratio'])
    
    # ========================================================================
    # 3. DISTANCE/PERSPECTIVE FEATURES
    # ========================================================================
    
    # Distance to camera (affects apparent size measurement accuracy)
    distance = np.sqrt(new_df['D_x']**2 + new_df['D_y']**2)
    new_df['log_distance'] = np.log1p(distance)
    
    # Apparent volume (what camera sees, adjusted for distance)
    apparent_volume = volume / (new_df['D_x']**2 + epsilon)  # Using D_x^2 (inverse square law)
    new_df['log_apparent_volume'] = np.log1p(apparent_volume)
    
    # Viewing angle (lateral position - affects measurement distortion)
    new_df['view_angle_rad'] = np.arctan2(new_df['D_y'], new_df['D_x'])
    
    # Depth ratio (how frontal is the view)
    new_df['depth_ratio'] = new_df['D_x'] / (distance + epsilon)
    
    # ========================================================================
    # 4. INTERACTION FEATURES (NEW - for non-linear relationships)
    # ========================================================================
    
    # Volume × compactness (compact heavy objects vs elongated light ones)
    new_df['volume_compactness'] = new_df['log_volume'] * new_df['compactness']
    
    # Surface area × sphericity (shell weight indicator)
    new_df['surface_sphericity'] = new_df['log_surface_area'] * new_df['sphericity']
    
    # Size × distance interaction (perspective-corrected size)
    new_df['size_distance_interaction'] = new_df['log_volume'] / (new_df['log_distance'] + epsilon)
    
    # ========================================================================
    # Summary of engineered features
    # ========================================================================
    new_features = [
        # Size features (5)
        'log_volume', 'log_surface_area', 'max_dimension', 'log_max_dimension', 'log_geo_mean_dim',
        # Shape features (8)
        'aspect_ratio_xy', 'aspect_ratio_xz', 'aspect_ratio_yz', 'compactness', 
        'flatness', 'elongation', 'sphericity', 'log_vol_surface_ratio',
        # Perspective features (4)
        'log_distance', 'log_apparent_volume', 'view_angle_rad', 'depth_ratio',
        # Interaction features (3)
        'volume_compactness', 'surface_sphericity', 'size_distance_interaction'
    ]
    
    print(f"✓ Created {len(new_features)} physics-informed features:")
    print(f"  Size features (5):        log_volume, log_surface_area, max_dimension, log_max_dimension, log_geo_mean_dim")
    print(f"  Shape features (8):       aspect_ratio_xy/xz/yz, compactness, flatness, elongation, sphericity, log_vol_surface_ratio")
    print(f"  Perspective features (4): log_distance, log_apparent_volume, view_angle_rad, depth_ratio")
    print(f"  Interaction features (3): volume_compactness, surface_sphericity, size_distance_interaction")
    
    return new_df

