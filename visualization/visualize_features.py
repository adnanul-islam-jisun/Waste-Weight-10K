# Feature Visualization and Correlation Analysis
"""
Visualizes the engineered features and their relationships with weight.
Helps identify which features are most predictive and check for multicollinearity.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.config import CSV_PATH, BASE_IMAGE_PATH
from features.feature_engineering import engineer_features


def load_and_prepare_data():
    """Load data and apply feature engineering."""
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    # Data cleaning
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['weight_in_kg'], inplace=True)
    df.fillna(0.0, inplace=True)
    
    # Filter out very low weights
    df = df[df['weight_in_kg'] >= 50]
    
    # Apply feature engineering
    df = engineer_features(df)
    
    # Rename for consistency
    if 'weight_in_kg' in df.columns:
        df['weight'] = df['weight_in_kg']
    
    print(f"✓ Loaded {len(df)} samples")
    return df


def get_feature_groups():
    """Define feature groups for organized visualization."""
    return {
        'Size Features': [
            'log_volume', 'log_surface_area', 'max_dimension', 
            'log_max_dimension', 'log_geo_mean_dim'
        ],
        'Shape Features': [
            'aspect_ratio_xy', 'aspect_ratio_xz', 'aspect_ratio_yz',
            'compactness', 'flatness', 'elongation', 
            'sphericity', 'log_vol_surface_ratio'
        ],
        'Perspective Features': [
            'log_distance', 'log_apparent_volume', 
            'view_angle_rad', 'depth_ratio'
        ],
        'Interaction Features': [
            'volume_compactness', 'surface_sphericity', 
            'size_distance_interaction'
        ]
    }


def plot_correlation_matrix(df, save_path='attention_analysis/feature_correlation.png'):
    """Plot correlation matrix for all features."""
    feature_groups = get_feature_groups()
    all_features = []
    for features in feature_groups.values():
        all_features.extend([f for f in features if f in df.columns])
    
    # Add weight
    all_features.append('weight')
    
    # Filter to existing features
    existing_features = [f for f in all_features if f in df.columns]
    
    # Compute correlation matrix
    corr_matrix = df[existing_features].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={'size': 8},
        ax=ax
    )
    
    plt.title('Feature Correlation Matrix\n(Including Weight)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {save_path}")
    
    return corr_matrix


def plot_feature_vs_weight(df, save_path='attention_analysis/features_vs_weight.png'):
    """Plot each feature against weight to visualize relationships."""
    feature_groups = get_feature_groups()
    
    # Collect all features
    all_features = []
    for features in feature_groups.values():
        all_features.extend([f for f in features if f in df.columns])
    
    n_features = len(all_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(all_features):
        ax = axes[idx]
        
        # Scatter plot with alpha for density
        ax.scatter(df[feature], df['weight'], alpha=0.3, s=10, c='steelblue')
        
        # Add regression line
        try:
            z = np.polyfit(df[feature].dropna(), df['weight'].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')
        except:
            pass
        
        # Compute correlation
        corr, pval = stats.pearsonr(df[feature].dropna(), df['weight'].dropna())
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Weight (kg)', fontsize=10)
        ax.set_title(f'{feature}\nr = {corr:.3f} (p = {pval:.2e})', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature vs Weight Relationships', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {save_path}")


def plot_feature_distributions(df, save_path='attention_analysis/feature_distributions.png'):
    """Plot distribution of each feature."""
    feature_groups = get_feature_groups()
    
    # Collect all features
    all_features = []
    for features in feature_groups.values():
        all_features.extend([f for f in features if f in df.columns])
    
    n_features = len(all_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(all_features):
        ax = axes[idx]
        
        # Histogram with KDE
        data = df[feature].dropna()
        ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        
        try:
            # KDE overlay
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        except:
            pass
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{feature}\nμ={data.mean():.2f}, σ={data.std():.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {save_path}")


def plot_feature_importance(df, save_path='attention_analysis/feature_importance.png'):
    """Plot feature importance based on correlation with weight."""
    feature_groups = get_feature_groups()
    
    # Collect all features
    all_features = []
    group_labels = []
    for group_name, features in feature_groups.items():
        for f in features:
            if f in df.columns:
                all_features.append(f)
                group_labels.append(group_name)
    
    # Compute correlations with weight
    correlations = []
    for feature in all_features:
        corr, _ = stats.pearsonr(df[feature].dropna(), df['weight'].dropna())
        correlations.append(abs(corr))  # Use absolute correlation
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Correlation': correlations,
        'Group': group_labels
    }).sort_values('Correlation', ascending=True)
    
    # Color by group
    group_colors = {
        'Size Features': '#2ecc71',
        'Shape Features': '#3498db',
        'Perspective Features': '#e74c3c',
        'Interaction Features': '#9b59b6'
    }
    colors = [group_colors[g] for g in importance_df['Group']]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    bars = ax.barh(importance_df['Feature'], importance_df['Correlation'], color=colors)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, importance_df['Correlation']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{corr:.3f}', va='center', fontsize=9)
    
    ax.set_xlabel('|Correlation with Weight|', fontsize=12)
    ax.set_title('Feature Importance (by Correlation with Weight)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in group_colors.values()]
    ax.legend(legend_handles, group_colors.keys(), loc='lower right', title='Feature Group')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {save_path}")
    
    return importance_df


def plot_pairplot_by_group(df, save_dir='attention_analysis'):
    """Create pairplots for each feature group."""
    feature_groups = get_feature_groups()
    
    for group_name, features in feature_groups.items():
        existing_features = [f for f in features if f in df.columns]
        if len(existing_features) < 2:
            continue
        
        # Add weight for comparison
        plot_features = existing_features + ['weight']
        
        # Sample for faster plotting
        sample_df = df[plot_features].sample(min(1000, len(df)), random_state=42)
        
        # Create pairplot
        fig = sns.pairplot(
            sample_df, 
            diag_kind='kde',
            plot_kws={'alpha': 0.5, 's': 20},
            diag_kws={'fill': True}
        )
        
        fig.fig.suptitle(f'{group_name} Pairplot', y=1.02, fontsize=14, fontweight='bold')
        
        save_path = os.path.join(save_dir, f'pairplot_{group_name.lower().replace(" ", "_")}.png')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: {save_path}")


def print_correlation_summary(df):
    """Print summary of correlations with weight."""
    feature_groups = get_feature_groups()
    
    print("\n" + "="*70)
    print("CORRELATION SUMMARY WITH WEIGHT")
    print("="*70)
    
    all_correlations = []
    
    for group_name, features in feature_groups.items():
        print(f"\n{group_name}:")
        print("-" * 50)
        
        for feature in features:
            if feature in df.columns:
                corr, pval = stats.pearsonr(df[feature].dropna(), df['weight'].dropna())
                significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  {feature:30s}: r = {corr:+.4f} {significance}")
                all_correlations.append((feature, corr, pval, group_name))
    
    # Find top 5 positive and negative correlations
    all_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 5 POSITIVE CORRELATIONS:")
    print("-" * 50)
    for feat, corr, pval, group in all_correlations[:5]:
        print(f"  {feat:30s}: r = {corr:+.4f} ({group})")
    
    print("\n" + "="*70)
    print("TOP 5 NEGATIVE CORRELATIONS:")
    print("-" * 50)
    for feat, corr, pval, group in all_correlations[-5:]:
        print(f"  {feat:30s}: r = {corr:+.4f} ({group})")
    
    print("\n" + "="*70)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("="*70 + "\n")


def check_multicollinearity(df, threshold=0.8):
    """Check for highly correlated features (multicollinearity)."""
    feature_groups = get_feature_groups()
    all_features = []
    for features in feature_groups.values():
        all_features.extend([f for f in features if f in df.columns])
    
    corr_matrix = df[all_features].corr()
    
    print("\n" + "="*70)
    print(f"MULTICOLLINEARITY CHECK (|r| > {threshold})")
    print("="*70)
    
    high_corr_pairs = []
    for i in range(len(all_features)):
        for j in range(i+1, len(all_features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                high_corr_pairs.append((all_features[i], all_features[j], corr))
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs:")
        print("-" * 50)
        for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {f1} ↔ {f2}: r = {corr:.4f}")
        print(f"\nNote: Consider removing one feature from each pair to reduce multicollinearity.")
    else:
        print(f"\n✓ No feature pairs with |r| > {threshold} found.")
    
    print("="*70 + "\n")


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'attention_analysis')

def main():
    """Run all visualizations."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data()
    
    # Print statistics
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df)}")
    print(f"Weight range: {df['weight'].min():.1f} - {df['weight'].max():.1f} kg")
    print(f"Weight mean: {df['weight'].mean():.1f} kg")
    print(f"Weight std: {df['weight'].std():.1f} kg")
    print("="*70 + "\n")
    
    # 1. Correlation summary
    print_correlation_summary(df)
    
    # 2. Check multicollinearity
    check_multicollinearity(df)
    
    # 3. Plot correlation matrix
    print("\n📊 Generating correlation matrix...")
    plot_correlation_matrix(df, save_path=os.path.join(OUTPUT_DIR, 'feature_correlation.png'))
    
    # 4. Plot features vs weight
    print("\n📊 Generating feature vs weight plots...")
    plot_feature_vs_weight(df, save_path=os.path.join(OUTPUT_DIR, 'features_vs_weight.png'))
    
    # 5. Plot feature distributions
    print("\n📊 Generating feature distribution plots...")
    plot_feature_distributions(df, save_path=os.path.join(OUTPUT_DIR, 'feature_distributions.png'))
    
    # 6. Plot feature importance
    print("\n📊 Generating feature importance plot...")
    importance_df = plot_feature_importance(df, save_path=os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    
    # 7. Optional: Pairplots (can be slow for large datasets)
    print("\n📊 Generating pairplots by feature group...")
    plot_pairplot_by_group(df, save_dir=OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print(f"  Check the '{OUTPUT_DIR}' folder for saved plots.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
