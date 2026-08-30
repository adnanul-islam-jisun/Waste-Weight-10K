"""
Ablation Study Configuration
Defines all experiment variants for Model Architecture Ablation Study.
"""

import os
from datetime import datetime

# ============================================================================
# ABLATION STUDY SETTINGS
# ============================================================================

# Base directory for all ablation results
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION_BASE_DIR = os.path.join(PROJECT_ROOT, "results", "ablation")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Training settings for ablation (faster than full training)
ABLATION_EPOCHS = 500  # Reduced from 50 for faster experiments
ABLATION_EARLY_STOPPING_PATIENCE = 10
ABLATION_BATCH_SIZE = 64  # Can be adjusted based on GPU memory

# Whether to save full checkpoints (can be large)
SAVE_CHECKPOINTS = True  # Set to False to save space

# Whether to run in debug mode (single batch per epoch)
DEBUG_MODE = False

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

EXPERIMENTS = {
    # Experiment 1: Full Model (Baseline) - ViT + Metadata + Mutual Attention
    "exp1_full_model": {
        "name": "Full Model (Baseline)",
        "description": "Complete architecture: ViT-B/16 + Metadata + Mutual Attention",
        "config": {
            "use_image": True,
            "use_metadata": True,
            "use_attention_fusion": True,
            "image_model": "vit_b_16",
            "image_output_dim": 768,
            "category_embedding_dim": 32,
            "metadata_output_dim": 256,
            "dropout_rate": 0.3,
            "fusion_hidden_dims": [512, 256, 128],
            "attention_embed_dim": 256,
            "attention_num_heads": 8,
        },
        "expected_params": "~98M",
        "notes": "Baseline configuration with all features enabled"
    },
    
    # Experiment 2: Image Only - No metadata
    "exp2_image_only": {
        "name": "Image Only",
        "description": "Only ViT-B/16 image encoder, no metadata features",
        "config": {
            "use_image": True,
            "use_metadata": False,
            "use_attention_fusion": False,  # Not applicable
            "image_model": "vit_b_16",
            "image_output_dim": 768,
            "dropout_rate": 0.3,
            "fusion_hidden_dims": [512, 256, 128],
        },
        "expected_params": "~86M",
        "notes": "Tests contribution of metadata features"
    },
    
    # Experiment 3: Metadata Only - No images
    "exp3_metadata_only": {
        "name": "Metadata Only",
        "description": "Only metadata encoder (categories + numerical), no images",
        "config": {
            "use_image": False,
            "use_metadata": True,
            "use_attention_fusion": False,  # Not applicable
            "category_embedding_dim": 32,
            "metadata_output_dim": 256,
            "dropout_rate": 0.3,
            "fusion_hidden_dims": [512, 256, 128],
        },
        "expected_params": "~12M",
        "notes": "Tests contribution of visual features"
    },
    
    # Experiment 4: No Mutual Attention - Simple Late Fusion
    "exp4_no_attention": {
        "name": "No Mutual Attention",
        "description": "ViT + Metadata with simple concatenation (no attention)",
        "config": {
            "use_image": True,
            "use_metadata": True,
            "use_attention_fusion": False,
            "image_model": "vit_b_16",
            "image_output_dim": 768,
            "category_embedding_dim": 32,
            "metadata_output_dim": 256,
            "dropout_rate": 0.3,
            "fusion_hidden_dims": [512, 256, 128],
        },
        "expected_params": "~98M",
        "notes": "Tests contribution of mutual attention mechanism"
    },
    
    # Experiment 5: ViT-B/32 - Faster variant
    "exp5_vit_b32": {
        "name": "ViT-B/32 (Faster)",
        "description": "Faster ViT variant with larger patches",
        "config": {
            "use_image": True,
            "use_metadata": True,
            "use_attention_fusion": True,
            "image_model": "vit_b_32",
            "image_output_dim": 768,
            "category_embedding_dim": 32,
            "metadata_output_dim": 256,
            "dropout_rate": 0.3,
            "fusion_hidden_dims": [512, 256, 128],
            "attention_embed_dim": 256,
            "attention_num_heads": 8,
        },
        "expected_params": "~88M",
        "notes": "Tests speed vs accuracy trade-off with larger patches"
    },
    
    # Experiment 6: ViT-L/16 - Larger model
    "exp6_vit_l16": {
        "name": "ViT-L/16 (Larger)",
        "description": "Higher capacity ViT variant",
        "config": {
            "use_image": True,
            "use_metadata": True,
            "use_attention_fusion": True,
            "image_model": "vit_l_16",
            "image_output_dim": 1024,  # ViT-L has 1024-dim output
            "category_embedding_dim": 32,
            "metadata_output_dim": 256,
            "dropout_rate": 0.3,
            "fusion_hidden_dims": [512, 256, 128],
            "attention_embed_dim": 256,
            "attention_num_heads": 8,
        },
        "expected_params": "~316M",
        "notes": "Tests if larger model improves accuracy"
    },
}

# ============================================================================
# METRICS TO TRACK
# ============================================================================

PRIMARY_METRICS = [
    "mae",          # Mean Absolute Error (kg)
    "rmse",         # Root Mean Squared Error (kg)
    "mape",         # Mean Absolute Percentage Error (%)
    "r2",           # R-squared Score
]

SECONDARY_METRICS = [
    "train_time",           # Total training time (seconds)
    "train_time_per_epoch", # Average time per epoch (seconds)
    "inference_time",       # Average inference time per sample (ms)
    "model_params",         # Total model parameters
    "model_size_mb",        # Model size in MB
    "gpu_memory_mb",        # Peak GPU memory usage (MB)
    "final_train_loss",     # Final training loss
    "best_val_loss",        # Best validation loss
    "best_epoch",           # Epoch with best validation loss
]

# Weight range categories for detailed analysis
WEIGHT_RANGES = {
    "light": (50, 100),      # 50-100 kg
    "medium": (100, 500),    # 100-500 kg
    "heavy": (500, 10000),   # 500+ kg
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figures to generate
VISUALIZATIONS = [
    "mae_comparison",           # Bar chart of MAE across experiments
    "rmse_comparison",          # Bar chart of RMSE across experiments
    "training_curves",          # Training/validation loss curves
    "error_distribution",       # Box plots of prediction errors
    "weight_range_performance", # Performance by weight range
    "speed_vs_accuracy",        # Training time vs MAE scatter
    "model_size_vs_accuracy",   # Model parameters vs MAE
]

# Plot style
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 300  # High resolution for publication
FIGURE_FORMAT = "png"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_experiment_dir(exp_key: str) -> str:
    """Get the directory path for an experiment."""
    return os.path.join(ABLATION_BASE_DIR, exp_key)

def get_checkpoint_dir(exp_key: str) -> str:
    """Get the checkpoint directory for an experiment."""
    return os.path.join(get_experiment_dir(exp_key), "checkpoints")

def get_visualization_dir() -> str:
    """Get the visualization directory."""
    return os.path.join(ABLATION_BASE_DIR, "visualizations")

def create_ablation_directories():
    """Create all necessary directories for ablation study."""
    os.makedirs(ABLATION_BASE_DIR, exist_ok=True)
    os.makedirs(get_visualization_dir(), exist_ok=True)
    
    for exp_key in EXPERIMENTS.keys():
        os.makedirs(get_experiment_dir(exp_key), exist_ok=True)
        if SAVE_CHECKPOINTS:
            os.makedirs(get_checkpoint_dir(exp_key), exist_ok=True)

def get_experiment_list(experiment_ids: str = None) -> list:
    """
    Get list of experiments to run.
    
    Args:
        experiment_ids: Comma-separated experiment IDs (e.g., "1,2,4") or None for all
    
    Returns:
        List of experiment keys to run
    """
    if experiment_ids is None:
        return list(EXPERIMENTS.keys())
    
    # Parse experiment IDs
    exp_nums = [int(x.strip()) for x in experiment_ids.split(",")]
    exp_keys = [f"exp{num}_{list(EXPERIMENTS.keys())[num-1].split('_', 1)[1]}" 
                for num in exp_nums]
    
    # Validate
    for key in exp_keys:
        if key not in EXPERIMENTS:
            raise ValueError(f"Invalid experiment key: {key}")
    
    return exp_keys

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def get_progress_file() -> str:
    """Get the progress tracking file path."""
    return os.path.join(ABLATION_BASE_DIR, "progress.json")

def load_progress():
    """Load progress from file."""
    import json
    progress_file = get_progress_file()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "timestamp": None}

def save_progress(progress):
    """Save progress to file."""
    import json
    progress_file = get_progress_file()
    progress["timestamp"] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
