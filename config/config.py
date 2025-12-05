"""
Unified Configuration for Weight Prediction System
Merges all configuration settings into a single, clean file.
"""

import torch
import numpy as np
import os


# ============================================================================
# DATA PATHS
# ============================================================================

# ============================================================================
# DATA PATHS
# ============================================================================

# Use Environment Variable 'DATA_PATH' if set (for Docker), else use default local path
DEFAULT_LOCAL_PATH = "/home/asiful/adnan_workspace/Dataset/disaster_data/waste_dataset"
DATA_PATH = os.getenv("DATA_PATH", DEFAULT_LOCAL_PATH)

BASE_IMAGE_PATH = DATA_PATH
CSV_PATH = os.path.join(DATA_PATH, "image.csv")

print(f"✓ Using Data Path: {DATA_PATH}")


# ============================================================================
# DEVICE CONFIGURATION & GPU OPTIMIZATION
# ============================================================================

# Auto-detect best available device
if torch.cuda.is_available():
    DEVICE = "cuda"
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune for optimal performance
    torch.backends.cudnn.deterministic = False  # Faster but non-deterministic
    torch.backends.cudnn.enabled = True
    
    # Enable TF32 on Ampere GPUs for faster matmul
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"✓ Compute Capability: {'.'.join(map(str, torch.cuda.get_device_capability()))}")
    
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✓ Using Apple Silicon GPU (MPS)")
else:
    DEVICE = "cpu"
    print("⚠ Using CPU - Training will be slow")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Image Encoder (Vision Transformer)
IMAGE_MODEL = 'vit_b_16'  # Options: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
IMAGE_OUTPUT_DIM = 768  # ViT-B/16 output dimension
IMAGE_SIZE = 224  # ViT requires 224x224

# Metadata Encoder
CATEGORY_EMBEDDING_DIM = 32
METADATA_OUTPUT_DIM = 256

# Fusion Network
FUSION_HIDDEN_DIMS = [512, 256, 128]
DROPOUT_RATE = 0.3  # FIXED: Was 0.5 (too high), now balanced
USE_RESIDUAL = True

# Mutual Attention Fusion (Advanced)
USE_ATTENTION_FUSION = True  # Set to False for simple late fusion
ATTENTION_EMBED_DIM = 256    # Embedding dimension for attention
ATTENTION_NUM_HEADS = 8      # Number of attention heads (must divide ATTENTION_EMBED_DIM)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Weight transformation
# LOG TRANSFORM: Optimal for wide weight ranges (50-3450kg)
# Transforms exponential distribution to more normal distribution
# Model predicts log1p(weight), then expm1() converts back to kg
USE_LOG_TRANSFORM = True  # Log transform for better optimization

# Data splits - ADJUSTED for 10k dataset
TRAIN_SPLIT = 0.7   # 7,000 images
VAL_SPLIT = 0.15    # 1,500 images (increased from 0.1 for more stable validation)
TEST_SPLIT = 0.15   # 1,500 images

# Data augmentation (training only)
AUGMENTATION = {
    'horizontal_flip': 0.5,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2
    }
}


# ============================================================================
# TRAINING HYPERPARAMETERS (GPU OPTIMIZED)
# ============================================================================

# Basic training settings
EPOCHS = 120

# Batch size - OPTIMIZED for 10k dataset
# Balance: Large enough for stable gradients, small enough for regularization
# NOTE: Batch size 256 is TOO LARGE for ~7k samples (only ~27 batches per epoch)
#       Smaller batch sizes provide more gradient updates and regularization
if DEVICE == "cuda":
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 24:  # A100, RTX 3090/4090
        BATCH_SIZE = 128  # CHANGED from 256: More gradient updates per epoch (~109 batches)
    elif gpu_memory_gb >= 16:  # V100, RTX 3080
        BATCH_SIZE = 48  # ~145 batches per epoch
    elif gpu_memory_gb >= 12:  # RTX 3060 Ti
        BATCH_SIZE = 32  # ~218 batches per epoch
    elif gpu_memory_gb >= 8:  # RTX 3060
        BATCH_SIZE = 24  # ~291 batches per epoch
    else:  # Lower-end GPUs
        BATCH_SIZE = 16
    print(f"✓ Auto-adjusted batch size: {BATCH_SIZE} (based on {gpu_memory_gb:.1f} GB GPU)")
    print(f"  Batches per epoch: ~{7000//BATCH_SIZE} (with 7k training samples)")
elif DEVICE == "mps":
    BATCH_SIZE = 32  # Good balance for Apple Silicon
    print(f"✓ Batch size set to {BATCH_SIZE} for MPS (Apple Silicon)")
else:
    BATCH_SIZE = 8  # CPU

# DataLoader workers - GPU optimized
if DEVICE == "cuda":
    NUM_WORKERS = min(8, os.cpu_count() or 4)  # Use multiple workers for GPU
    PIN_MEMORY = True  # Faster data transfer to GPU
    PERSISTENT_WORKERS = True  # Keep workers alive between epochs
elif DEVICE == "mps":
    # MPS Optimization
    NUM_WORKERS = min(4, os.cpu_count() or 2)  # Allow parallel data loading
    PIN_MEMORY = False  # Usually better False for MPS (Unified Memory)
    PERSISTENT_WORKERS = True  # CRITICAL: Keep workers alive to reduce overhead
else:
    NUM_WORKERS = 2
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False

# Optimizer settings
LEARNING_RATE = 1e-4  # Initial LR
WEIGHT_DECAY = 1e-4  # Reduced from 0.01 to allow more adaptation

# Loss function
# MSLE: Mean Squared Log Error - optimal for wide weight ranges
# Works in log-space, so relative errors are penalized equally across all weights
LOSS_TYPE = 'msle'  # Best for wide weight ranges (50-3450kg)

# Progressive training (freeze → fine-tune)
FREEZE_IMAGE_ENCODER_EPOCHS = 10  # Freeze ViT for first N epochs

# Gradient settings
GRADIENT_CLIP_NORM = 1.0
USE_AMP = DEVICE == "cuda"  # Automatic Mixed Precision for GPU speedup

# Learning rate scheduler - Use Cosine Annealing with Warm Restarts
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'cosine_warm'  # Options: 'plateau', 'cosine', 'cosine_warm'
LR_SCHEDULER_PATIENCE = 10  # For plateau scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-7
COSINE_T_0 = 20  # Restart every 20 epochs
COSINE_T_MULT = 2  # Double the period after each restart

# Exponential Moving Average (EMA) for stable predictions
USE_EMA = True
EMA_DECAY = 0.999  # Higher = more stable but slower adaptation

# Label Smoothing (helps with overconfidence)
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_FACTOR = 0.1  # Add noise to targets

# Early stopping
EARLY_STOPPING_PATIENCE = 20


# ============================================================================
# SAVE PATHS
# ============================================================================

CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Note: cudnn.benchmark=True makes training non-deterministic but faster


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_gpu_memory():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ GPU cache cleared")


def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        return allocated, reserved
    return 0, 0


def optimize_for_inference(model):
    """Optimize model for faster inference."""
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Use torch.jit.script for speedup (if compatible)
    # model = torch.jit.script(model)  # Uncomment if model supports
    
    return model


def print_config():
    """Print current configuration."""
    print("\n" + "="*80)
    print("CONFIGURATION (GPU OPTIMIZED)")
    print("="*80)
    print(f"\n📂 Data:")
    print(f"  CSV: {CSV_PATH}")
    print(f"  Images: {BASE_IMAGE_PATH}")
    
    print(f"\n🖥️  Device & Performance:")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  Mixed Precision (AMP): {USE_AMP}")
        print(f"  Pin Memory: {PIN_MEMORY}")
        print(f"  Persistent Workers: {PERSISTENT_WORKERS}")
    
    print(f"\n🏗️  Model:")
    print(f"  Image Encoder: {IMAGE_MODEL}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Output Dims: Image={IMAGE_OUTPUT_DIM}, Metadata={METADATA_OUTPUT_DIM}")
    
    print(f"\n📊 Training:")
    print(f"  Batch Size: {BATCH_SIZE} (auto-optimized)")
    print(f"  DataLoader Workers: {NUM_WORKERS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Loss: {LOSS_TYPE.upper()}")
    print(f"  Weight Transform: {'LOG (log1p)' if USE_LOG_TRANSFORM else 'None'}")
    print(f"  Progressive Training: Freeze {FREEZE_IMAGE_ENCODER_EPOCHS} epochs")
    print(f"  Gradient Clipping: {GRADIENT_CLIP_NORM}")
    
    print(f"\n💾 Outputs:")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print(f"  Logs: {LOGS_DIR}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_config()
    if DEVICE == "cuda":
        get_gpu_memory_usage()


