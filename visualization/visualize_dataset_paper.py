import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the current directory to the path so we can import config
sys.path.append(os.getcwd())

try:
    from config.config import CSV_PATH
except ImportError:
    # Fallback if config import fails
    CSV_PATH = "/home/asiful/adnan_workspace/Dataset/disaster_data/waste_dataset/image.csv"

def set_professional_style():
    """Sets a professional plotting style suitable for academic papers."""
    # Use seaborn style as a base
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Custom rcParams for publication quality
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 6
    })

def load_and_clean_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    print(f"Initial records in CSV: {initial_count}")
    
    # Convert numerical columns
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing weight
    df_clean = df.dropna(subset=['weight_in_kg'])
    nan_dropped = len(df) - len(df_clean)
    if nan_dropped > 0:
        print(f"Dropped {nan_dropped} records with missing or invalid 'weight_in_kg'")
    df = df_clean
    
    # Filter out weights < 50kg as per training script
    # df_filtered = df[df['weight_in_kg'] >= 50]
    # low_weight_dropped = len(df) - len(df_filtered)
    # if low_weight_dropped > 0:
    #     print(f"Dropped {low_weight_dropped} records with weight < 50kg")
    # df = df_filtered
    
    # Clean 'Type' column
    if 'Type' in df.columns:
        # Strip whitespace
        df['Type'] = df['Type'].str.strip()
        
        # Standardize categories
        type_mapping = {
            'vehicle': 'Automotive Scrap',
            'car door': 'Automotive Scrap',
            'bonet': 'Automotive Scrap',
            'back': 'Automotive Scrap',
            'Tire': 'Automotive Scrap',
            'car': 'Automotive Scrap',
            'car ': 'Automotive Scrap', # Handle potential trailing space if strip didn't catch it or for safety
            'grash': 'General Trash',
            'card board': 'Cardboard',
            'cylinder track': 'Cylindrical Object',
            'metal': 'Ferrous Metal',
            'plastic': 'Rigid Plastic',
            'wood': 'Wood',
            'rubber': 'Rubber',
            'battery': 'Battery',
            'fridge': 'Appliance',
            'foam': 'Foam'
        }
        df['Type'] = df['Type'].replace(type_mapping)
        print("Cleaned 'Type' column: Renamed categories to formal names")

    print(f"Final Data loaded: {len(df)} samples")
    return df

def plot_target_distribution(df, save_dir):
    """Plots the distribution of the target variable (Weight)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original Weight Distribution
    sns.histplot(data=df, x='weight_in_kg', kde=True, ax=axes[0], color='#2c3e50', bins=30)
    axes[0].set_title('Distribution of Weight (kg)')
    axes[0].set_xlabel(r'Weight $y$ (kg)')
    axes[0].set_ylabel('Count')
    
    # Log-Transformed Weight Distribution
    log_weights = np.log1p(df['weight_in_kg'])
    sns.histplot(x=log_weights, kde=True, ax=axes[1], color='#e74c3c', bins=30)
    axes[1].set_title('Log-Transformed Weight Distribution')
    axes[1].set_xlabel(r'Log(Weight $y$ + 1)')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_distribution.png'), bbox_inches='tight')
    plt.close()
    print("Saved weight_distribution.png")

def plot_categorical_counts(df, save_dir):
    """Plots counts of categorical features."""
    categorical_cols = ['Type', 'sub_type']
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Get value counts and sort
        counts = df[col].value_counts()
        
        # Create bar plot
        sns.barplot(x=counts.index, y=counts.values, palette='viridis', hue=counts.index, legend=False)
        
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{col}_distribution.png'), bbox_inches='tight')
        plt.close()
        print(f"Saved {col}_distribution.png")

def plot_numerical_distributions(df, save_dir):
    """Plots boxplots for numerical features."""
    numerical_cols = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y']
    
    # Melt dataframe for seaborn boxplot
    df_melted = df[numerical_cols].melt(var_name='Feature', value_name='Value')
    
    # Rename features for plotting
    feature_map = {
        'V_x': r'$V_x$',
        'V_y': r'$V_y$',
        'V_z': r'$V_z$',
        'D_x': r'$D_x$',
        'D_y': r'$D_y$'
    }
    df_melted['Feature'] = df_melted['Feature'].map(feature_map)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Feature', y='Value', palette='Set2', hue='Feature', legend=False)
    
    plt.title('Distribution of Numerical Features (Dimensions)')
    plt.xlabel('Feature')
    plt.ylabel('Value (cm)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'numerical_features_boxplot.png'), bbox_inches='tight')
    plt.close()
    print("Saved numerical_features_boxplot.png")

def plot_correlation_matrix(df, save_dir):
    """Plots correlation matrix heatmap."""
    numerical_cols = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    corr = df[numerical_cols].corr()
    
    # Rename columns/index for heatmap
    label_map = {
        'V_x': r'$V_x$', 'V_y': r'$V_y$', 'V_z': r'$V_z$',
        'D_x': r'$D_x$', 'D_y': r'$D_y$',
        'weight_in_kg': r'Weight $y$'
    }
    corr.rename(columns=label_map, index=label_map, inplace=True)

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'), bbox_inches='tight')
    plt.close()
    print("Saved correlation_matrix.png")

def plot_feature_vs_weight(df, save_dir):
    """Plots scatter plots of features vs weight."""
    numerical_cols = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.scatterplot(data=df, x=col, y='weight_in_kg', ax=axes[i], alpha=0.6, hue='Type', palette='deep')
        
        # Mathematical labels
        col_label = f"${col}$"
        axes[i].set_title(f'{col_label} vs Weight')
        axes[i].set_xlabel(f'{col_label} (cm)')
        axes[i].set_ylabel(r'Weight $y$ (kg)')
    
    # Remove empty subplot if any
    if len(numerical_cols) < len(axes):
        for j in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[j])
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'features_vs_weight.png'), bbox_inches='tight')
    plt.close()
    print("Saved features_vs_weight.png")

def plot_3d_dimensions(df, save_dir):
    """Plots 3D scatter plot of dimensions."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Map types to colors
    types = df['Type'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(types)))
    type_color_map = dict(zip(types, colors))
    
    for t in types:
        subset = df[df['Type'] == t]
        ax.scatter(subset['V_x'], subset['V_y'], subset['V_z'], 
                   label=t, s=50, alpha=0.6, c=[type_color_map[t]])
    
    ax.set_xlabel(r'$V_x$')
    ax.set_ylabel(r'$V_y$')
    ax.set_zlabel(r'$V_z$')
    ax.set_title(r'3D Distribution of Dimensions ($V_x, V_y, V_z$)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3d_dimensions.png'), bbox_inches='tight')
    plt.close()
    print("Saved 3d_dimensions.png")

def main():
    save_dir = 'dataset_visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    set_professional_style()
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    df = load_and_clean_data(CSV_PATH)
    
    print("Generating visualizations...")
    plot_target_distribution(df, save_dir)
    plot_categorical_counts(df, save_dir)
    plot_numerical_distributions(df, save_dir)
    plot_correlation_matrix(df, save_dir)
    plot_feature_vs_weight(df, save_dir)
    plot_3d_dimensions(df, save_dir)
    
    print(f"\nAll visualizations saved to directory: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
