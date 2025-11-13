# Configuration Settings
import torch
import os
import numpy as np

# Data paths - Update these paths to your dataset location
CSV_PATH = "/content/drive/MyDrive/KaggleData/disaster/waste_dataset/image.csv"
BASE_IMAGE_PATH = "//content/drive/MyDrive/KaggleData/disaster/waste_dataset"

# Image and training parameters
IMG_SIZE = 224
BATCH_SIZE = 128  # Reduced for GPU memory constraints
LEARNING_RATE = 0.001
EPOCHS = 100

# GPU detection and setup
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Available: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
        
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon GPU
    print("Using Apple Silicon GPU (MPS)")
else:
    DEVICE = "cpu"
    print("Using CPU - GPU not available")

# For reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")

# Memory management function
def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Model save paths
MODEL_SAVE_DIR = "saved_models"
CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
