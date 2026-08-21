"""
Ablation Study Utilities
Helper functions for running ablation experiments and analyzing results.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any


# ============================================================================
# EXPERIMENT MANAGEMENT
# ============================================================================

def save_experiment_config(exp_dir: str, exp_config: dict, exp_info: dict):
    """Save experiment configuration to JSON file."""
    config_path = os.path.join(exp_dir, "config.json")
    
    full_config = {
        "experiment_info": exp_info,
        "model_config": exp_config,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    
    print(f"  ✓ Saved config to: {config_path}")


def save_experiment_results(
    exp_dir: str,
    metrics: dict,
    predictions: np.ndarray = None,
    targets: np.ndarray = None,
    training_history: dict = None
):
    """Save experiment results."""
    
    # Save metrics as JSON
    metrics_path = os.path.join(exp_dir, "test_results.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved metrics to: {metrics_path}")
    
    # Save predictions CSV
    if predictions is not None and targets is not None:
        pred_df = pd.DataFrame({
            'actual_weight_kg': targets,
            'predicted_weight_kg': predictions,
            'error_kg': predictions - targets,
            'abs_error_kg': np.abs(predictions - targets),
            'error_percent': np.abs((predictions - targets) / targets) * 100
        })
        pred_path = os.path.join(exp_dir, "predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"  ✓ Saved predictions to: {pred_path}")
    
    # Save training history
    if training_history is not None:
        history_path = os.path.join(exp_dir, "training_history.json")
        # Convert numpy types to Python native types
        history_serializable = {}
        for key, value in training_history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            elif hasattr(value, 'item'):
                history_serializable[key] = float(value)
            else:
                history_serializable[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"  ✓ Saved training history to: {history_path}")


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    weight_ranges: Dict[str, Tuple[float, float]] = None
) -> dict:
    """
    Calculate comprehensive metrics for evaluation.
    
    Args:
        predictions: Predicted weights
        targets: Actual weights
        weight_ranges: Dict of weight ranges for detailed analysis
    
    Returns:
        Dictionary of metrics
    """
    # Ensure numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Overall metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((predictions - targets) / targets)) * 100
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "mean_error": float(np.mean(predictions - targets)),
        "median_error": float(np.median(predictions - targets)),
        "std_error": float(np.std(predictions - targets)),
        "num_samples": len(predictions),
    }
    
    # Per-range metrics if provided
    if weight_ranges is not None:
        for range_name, (min_weight, max_weight) in weight_ranges.items():
            mask = (targets >= min_weight) & (targets < max_weight)
            if mask.sum() > 0:
                range_preds = predictions[mask]
                range_targets = targets[mask]
                
                range_mae = np.mean(np.abs(range_preds - range_targets))
                range_rmse = np.sqrt(np.mean((range_preds - range_targets) ** 2))
                range_mape = np.mean(np.abs((range_preds - range_targets) / range_targets)) * 100
                
                metrics[f"{range_name}_mae"] = float(range_mae)
                metrics[f"{range_name}_rmse"] = float(range_rmse)
                metrics[f"{range_name}_mape"] = float(range_mape)
                metrics[f"{range_name}_samples"] = int(mask.sum())
    
    return metrics


def calculate_model_stats(model: torch.nn.Module) -> dict:
    """Calculate model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size (rough approximation)
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "model_size_mb": float(model_size_mb),
        "params_millions": float(total_params / 1e6),
    }


def measure_inference_time(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_batches: int = 10
) -> float:
    """
    Measure average inference time per sample.
    
    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to use
        num_batches: Number of batches to measure
    
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Prepare batch
            images = batch['image'].to(device) if 'image' in batch else None
            category_indices = batch['category_idx'].to(device) if 'category_idx' in batch else None
            numerical = batch['numerical'].to(device) if 'numerical' in batch else None
            
            # Synchronize before timing
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Time inference
            start = time.time()
            
            if images is not None and category_indices is not None:
                _ = model(images, category_indices, numerical)
            elif images is not None:
                _ = model(images)
            else:
                _ = model(category_indices=category_indices, numerical_features=numerical)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
    
    # Calculate average time per sample (in milliseconds)
    avg_time_per_batch = np.mean(times)
    batch_size = len(batch['image']) if 'image' in batch else len(batch['category_idx'])
    avg_time_per_sample_ms = (avg_time_per_batch / batch_size) * 1000
    
    return float(avg_time_per_sample_ms)


def measure_gpu_memory(device: str) -> float:
    """Measure peak GPU memory usage in MB."""
    if device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return float(memory_allocated)
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# RESULTS AGGREGATION
# ============================================================================

def load_experiment_results(exp_dir: str) -> dict:
    """Load results from an experiment directory."""
    results = {}
    
    # Load config
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
    
    # Load test results
    metrics_path = os.path.join(exp_dir, "test_results.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Load training history
    history_path = os.path.join(exp_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            results['history'] = json.load(f)
    
    # Load predictions
    pred_path = os.path.join(exp_dir, "predictions.csv")
    if os.path.exists(pred_path):
        results['predictions'] = pd.read_csv(pred_path)
    
    return results


def aggregate_all_results(ablation_base_dir: str, experiments: dict) -> pd.DataFrame:
    """
    Aggregate results from all experiments into a single DataFrame.
    
    Args:
        ablation_base_dir: Base directory for ablation results
        experiments: Dictionary of experiment configurations
    
    Returns:
        DataFrame with all experiment results
    """
    all_results = []
    
    for exp_key, exp_info in experiments.items():
        exp_dir = os.path.join(ablation_base_dir, exp_key)
        
        if not os.path.exists(exp_dir):
            print(f"⚠ Experiment directory not found: {exp_dir}")
            continue
        
        results = load_experiment_results(exp_dir)
        
        if 'metrics' not in results:
            print(f"⚠ No metrics found for: {exp_key}")
            continue
        
        # Combine info
        row = {
            'experiment_id': exp_key,
            'experiment_name': exp_info['name'],
            'description': exp_info['description'],
        }
        
        # Add metrics
        row.update(results['metrics'])
        
        all_results.append(row)
    
    df = pd.DataFrame(all_results)
    
    # Sort by MAE (lower is better)
    if 'mae' in df.columns:
        df = df.sort_values('mae')
    
    return df


def save_summary_report(
    df: pd.DataFrame,
    ablation_base_dir: str,
    format: str = 'both'
):
    """
    Save summary report in CSV and/or LaTeX format.
    
    Args:
        df: Results DataFrame
        ablation_base_dir: Base directory
        format: 'csv', 'latex', or 'both'
    """
    # CSV format
    if format in ['csv', 'both']:
        csv_path = os.path.join(ablation_base_dir, "summary_report.csv")
        df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"\n✓ Summary CSV saved to: {csv_path}")
    
    # LaTeX format
    if format in ['latex', 'both']:
        latex_path = os.path.join(ablation_base_dir, "summary_report.tex")
        
        # Select key columns for LaTeX table
        latex_cols = ['experiment_name', 'mae', 'rmse', 'mape', 'r2', 
                      'train_time', 'params_millions', 'gpu_memory_mb']
        latex_cols = [col for col in latex_cols if col in df.columns]
        
        latex_df = df[latex_cols].copy()
        
        # Rename for better display
        latex_df.columns = [
            'Experiment', 'MAE (kg)', 'RMSE (kg)', 'MAPE (%)', 'R²',
            'Time (s)', 'Params (M)', 'GPU Mem (MB)'
        ][:len(latex_cols)]
        
        # Generate LaTeX
        latex_str = latex_df.to_latex(
            index=False,
            float_format='%.2f',
            caption='Model Architecture Ablation Study Results',
            label='tab:ablation_results',
            position='htbp'
        )
        
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        
        print(f"✓ LaTeX table saved to: {latex_path}")
    
    # JSON format (always save for programmatic access)
    json_path = os.path.join(ablation_base_dir, "summary_report.json")
    df.to_json(json_path, orient='records', indent=2)
    print(f"✓ JSON report saved to: {json_path}")


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def calculate_statistical_significance(
    results1: np.ndarray,
    results2: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Calculate statistical significance between two sets of results.
    Uses paired t-test.
    
    Args:
        results1: First set of predictions/errors
        results2: Second set of predictions/errors
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    from scipy import stats
    
    t_stat, p_value = stats.ttest_rel(results1, results2)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
    }


def print_experiment_summary(exp_name: str, metrics: dict, config: dict = None):
    """Print a nicely formatted experiment summary."""
    print("\n" + "=" * 80)
    print(f"📊 EXPERIMENT SUMMARY: {exp_name}")
    print("=" * 80)
    
    if config:
        print("\n🔧 Configuration:")
        print(f"  - Use Image: {config.get('use_image', 'N/A')}")
        print(f"  - Use Metadata: {config.get('use_metadata', 'N/A')}")
        print(f"  - Use Attention: {config.get('use_attention_fusion', 'N/A')}")
        if 'image_model' in config:
            print(f"  - Image Model: {config['image_model']}")
    
    print("\n📈 Performance Metrics:")
    print(f"  - MAE:  {metrics.get('mae', 0):.2f} kg")
    print(f"  - RMSE: {metrics.get('rmse', 0):.2f} kg")
    print(f"  - MAPE: {metrics.get('mape', 0):.2f}%")
    print(f"  - R²:   {metrics.get('r2', 0):.4f}")
    
    print("\n⏱️ Efficiency Metrics:")
    if 'train_time' in metrics:
        print(f"  - Training Time: {metrics['train_time']:.0f} seconds ({metrics['train_time']/60:.1f} min)")
    if 'inference_time' in metrics:
        print(f"  - Inference Time: {metrics['inference_time']:.2f} ms/sample")
    if 'params_millions' in metrics:
        print(f"  - Model Parameters: {metrics['params_millions']:.1f}M")
    if 'gpu_memory_mb' in metrics:
        print(f"  - GPU Memory: {metrics['gpu_memory_mb']:.0f} MB")
    
    # Weight range performance
    if 'light_mae' in metrics:
        print("\n🎯 Per Weight Range:")
        print(f"  - Light (50-100kg):   MAE={metrics.get('light_mae', 0):.2f}kg, RMSE={metrics.get('light_rmse', 0):.2f}kg")
        print(f"  - Medium (100-500kg): MAE={metrics.get('medium_mae', 0):.2f}kg, RMSE={metrics.get('medium_rmse', 0):.2f}kg")
        print(f"  - Heavy (500+kg):     MAE={metrics.get('heavy_mae', 0):.2f}kg, RMSE={metrics.get('heavy_rmse', 0):.2f}kg")
    
    print("=" * 80)
