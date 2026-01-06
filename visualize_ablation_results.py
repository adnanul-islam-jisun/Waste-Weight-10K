"""
Visualization Script for Ablation Study Results
Generates publication-ready figures and comparison charts.

Usage:
    python visualize_ablation_results.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_study_config import *
from ablation_utils import aggregate_all_results, load_experiment_results


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_mae_comparison(results_df: pd.DataFrame, save_dir: str):
    """Create bar chart comparing MAE across experiments."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by MAE
    plot_df = results_df.sort_values('mae')
    
    # Create bar chart
    bars = ax.bar(range(len(plot_df)), plot_df['mae'], color='steelblue', alpha=0.8)
    
    # Highlight best model
    best_idx = plot_df['mae'].idxmin()
    bars[list(plot_df.index).index(best_idx)].set_color('darkgreen')
    bars[list(plot_df.index).index(best_idx)].set_alpha(1.0)
    
    # Customize
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['experiment_name'], rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Model Architecture Ablation: MAE Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row['mae'] + 1, f"{row['mae']:.1f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mae_comparison.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ MAE comparison chart saved")


def plot_rmse_comparison(results_df: pd.DataFrame, save_dir: str):
    """Create bar chart comparing RMSE across experiments."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by RMSE
    plot_df = results_df.sort_values('rmse')
    
    # Create bar chart
    bars = ax.bar(range(len(plot_df)), plot_df['rmse'], color='coral', alpha=0.8)
    
    # Highlight best model
    best_idx = plot_df['rmse'].idxmin()
    bars[list(plot_df.index).index(best_idx)].set_color('darkgreen')
    bars[list(plot_df.index).index(best_idx)].set_alpha(1.0)
    
    # Customize
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['experiment_name'], rotation=45, ha='right')
    ax.set_ylabel('Root Mean Squared Error (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Model Architecture Ablation: RMSE Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row['rmse'] + 1, f"{row['rmse']:.1f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rmse_comparison.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ RMSE comparison chart saved")


def plot_metrics_comparison(results_df: pd.DataFrame, save_dir: str):
    """Create grouped bar chart for multiple metrics."""
    
    metrics = ['mae', 'rmse', 'mape']
    metric_labels = ['MAE (kg)', 'RMSE (kg)', 'MAPE (%)']
    
    # Normalize metrics for comparison (0-100 scale)
    normalized_df = results_df[['experiment_name'] + metrics].copy()
    for metric in metrics:
        max_val = normalized_df[metric].max()
        normalized_df[f'{metric}_norm'] = (normalized_df[metric] / max_val) * 100
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    # Create bars
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = width * (i - 1)
        ax.bar(x + offset, normalized_df[f'{metric}_norm'], width, label=label, alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Metric Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['experiment_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Multi-metric comparison chart saved")


def plot_training_curves(ablation_base_dir: str, experiments: dict, save_dir: str):
    """Plot training and validation curves for all experiments."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for i, (exp_key, exp_info) in enumerate(experiments.items()):
        exp_dir = os.path.join(ablation_base_dir, exp_key)
        results = load_experiment_results(exp_dir)
        
        if 'history' not in results:
            continue
        
        history = results['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot losses
        ax1.plot(epochs, history['train_loss'], '--', color=colors[i], alpha=0.6, linewidth=1)
        ax1.plot(epochs, history['val_loss'], '-', color=colors[i], 
                label=exp_info['name'], linewidth=2)
        
        # Plot MAE
        ax2.plot(epochs, history['val_mae'], '-', color=colors[i], 
                label=exp_info['name'], linewidth=2)
    
    # Customize loss plot
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Customize MAE plot
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation MAE (kg)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation MAE Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Training curves saved")


def plot_error_distribution(ablation_base_dir: str, experiments: dict, save_dir: str):
    """Create box plots of prediction errors for each experiment."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    all_errors = []
    labels = []
    
    for exp_key, exp_info in experiments.items():
        exp_dir = os.path.join(ablation_base_dir, exp_key)
        results = load_experiment_results(exp_dir)
        
        if 'predictions' not in results:
            continue
        
        pred_df = results['predictions']
        errors = pred_df['error_kg'].values
        
        all_errors.append(errors)
        labels.append(exp_info['name'])
    
    # Create box plot
    bp = ax.boxplot(all_errors, labels=labels, patch_artist=True, 
                     showfliers=False, widths=0.6)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_errors)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution Across Experiments', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Error distribution plot saved")


def plot_weight_range_performance(results_df: pd.DataFrame, save_dir: str):
    """Plot performance across different weight ranges."""
    
    # Check if weight range metrics exist
    if 'light_mae' not in results_df.columns:
        print("  ⚠ Weight range metrics not found, skipping...")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    ranges = ['light', 'medium', 'heavy']
    range_labels = ['Light\n(50-100kg)', 'Medium\n(100-500kg)', 'Heavy\n(500+kg)']
    
    x = np.arange(len(results_df))
    width = 0.25
    
    # Plot MAE by range
    for i, (range_name, label) in enumerate(zip(ranges, range_labels)):
        col = f'{range_name}_mae'
        if col in results_df.columns:
            offset = width * (i - 1)
            ax1.bar(x + offset, results_df[col], width, label=label, alpha=0.8)
    
    ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (kg)', fontsize=12, fontweight='bold')
    ax1.set_title('MAE by Weight Range', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['experiment_name'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot RMSE by range
    for i, (range_name, label) in enumerate(zip(ranges, range_labels)):
        col = f'{range_name}_rmse'
        if col in results_df.columns:
            offset = width * (i - 1)
            ax2.bar(x + offset, results_df[col], width, label=label, alpha=0.8)
    
    ax2.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (kg)', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE by Weight Range', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['experiment_name'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_range_performance.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Weight range performance plot saved")


def plot_speed_vs_accuracy(results_df: pd.DataFrame, save_dir: str):
    """Create scatter plot of training time vs MAE."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(results_df['train_time'] / 60, results_df['mae'], 
                        s=200, alpha=0.7, c=range(len(results_df)), 
                        cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add labels for each point
    for idx, row in results_df.iterrows():
        ax.annotate(row['experiment_name'], 
                   (row['train_time'] / 60, row['mae']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='yellow', alpha=0.3))
    
    # Add Pareto frontier (best trade-off curve)
    sorted_df = results_df.sort_values('train_time')
    best_mae = float('inf')
    pareto_points = []
    for idx, row in sorted_df.iterrows():
        if row['mae'] < best_mae:
            best_mae = row['mae']
            pareto_points.append((row['train_time'] / 60, row['mae']))
    
    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')
        ax.legend()
    
    ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'speed_vs_accuracy.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Speed vs accuracy plot saved")


def plot_model_size_vs_accuracy(results_df: pd.DataFrame, save_dir: str):
    """Create scatter plot of model parameters vs MAE."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(results_df['params_millions'], results_df['mae'], 
                        s=200, alpha=0.7, c=range(len(results_df)), 
                        cmap='plasma', edgecolors='black', linewidth=1.5)
    
    # Add labels
    for idx, row in results_df.iterrows():
        ax.annotate(row['experiment_name'], 
                   (row['params_millions'], row['mae']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='lightblue', alpha=0.3))
    
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Model Size vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_size_vs_accuracy.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Model size vs accuracy plot saved")


def create_summary_table_figure(results_df: pd.DataFrame, save_dir: str):
    """Create a publication-ready table as an image."""
    
    fig, ax = plt.subplots(figsize=(16, len(results_df) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns for table
    table_cols = ['experiment_name', 'mae', 'rmse', 'mape', 'r2', 
                  'params_millions', 'train_time']
    table_df = results_df[table_cols].copy()
    
    # Format columns
    table_df['train_time'] = table_df['train_time'].apply(lambda x: f"{x/60:.1f} min")
    table_df['params_millions'] = table_df['params_millions'].apply(lambda x: f"{x:.1f}M")
    
    # Rename columns
    table_df.columns = ['Experiment', 'MAE (kg)', 'RMSE (kg)', 'MAPE (%)', 
                       'R²', 'Parameters', 'Train Time']
    
    # Create table
    table = ax.table(cellText=table_df.values,
                    colLabels=table_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f1f1f1' if i % 2 == 0 else 'white')
    
    # Highlight best values
    best_mae_idx = table_df['MAE (kg)'].astype(float).idxmin() + 1
    best_rmse_idx = table_df['RMSE (kg)'].astype(float).idxmin() + 1
    
    table[(best_mae_idx, 1)].set_facecolor('#90EE90')
    table[(best_rmse_idx, 2)].set_facecolor('#90EE90')
    
    plt.savefig(os.path.join(save_dir, 'summary_table.png'), 
               dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("  ✓ Summary table image saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("📊 GENERATING ABLATION STUDY VISUALIZATIONS")
    print("=" * 80)
    
    # Check if results exist
    if not os.path.exists(ABLATION_BASE_DIR):
        print(f"\n❌ Error: Results directory not found: {ABLATION_BASE_DIR}")
        print("Please run the ablation study first: python run_ablation_study.py --all")
        return
    
    # Create visualization directory
    viz_dir = get_visualization_dir()
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load results
    print("\n📂 Loading experiment results...")
    results_df = aggregate_all_results(ABLATION_BASE_DIR, EXPERIMENTS)
    
    if len(results_df) == 0:
        print("❌ No experiment results found!")
        return
    
    print(f"✓ Loaded {len(results_df)} experiment results")
    
    # Generate all visualizations
    print("\n🎨 Generating visualizations...")
    
    plot_mae_comparison(results_df, viz_dir)
    plot_rmse_comparison(results_df, viz_dir)
    plot_metrics_comparison(results_df, viz_dir)
    plot_training_curves(ABLATION_BASE_DIR, EXPERIMENTS, viz_dir)
    plot_error_distribution(ABLATION_BASE_DIR, EXPERIMENTS, viz_dir)
    plot_weight_range_performance(results_df, viz_dir)
    plot_speed_vs_accuracy(results_df, viz_dir)
    plot_model_size_vs_accuracy(results_df, viz_dir)
    create_summary_table_figure(results_df, viz_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("=" * 80)
    print(f"\nVisualizations saved to: {viz_dir}/")
    print("\nGenerated files:")
    for viz_file in sorted(os.listdir(viz_dir)):
        if viz_file.endswith('.png'):
            print(f"  - {viz_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
