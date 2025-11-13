# Configuration Files - Unified & Cleaned

## ✅ Changes Made

### **1. Unified Configuration (`config/config.py`)**

**Single source of truth** for all settings:

```python
# Data paths
CSV_PATH = "..."
BASE_IMAGE_PATH = "..."

# Device (auto-detected)
DEVICE = "cuda" | "mps" | "cpu"

# Model architecture
IMAGE_MODEL = 'vit_b_16'
IMAGE_OUTPUT_DIM = 768
CATEGORY_EMBEDDING_DIM = 32
METADATA_OUTPUT_DIM = 256

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LOSS_TYPE = 'msle'

# Progressive training
FREEZE_IMAGE_ENCODER_EPOCHS = 10

# Preprocessing
USE_LOG_TRANSFORM = True
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
```

### **2. Simplified `training_config.py`**

Now imports from `config.py` and provides:
- `WeightPreprocessor` class (LOG transformation)
- `create_optimized_model()` function
- `create_trainer_for_your_data()` function

**Removed redundancy:**
- ❌ Duplicate configuration values
- ❌ `TrainingConfig` class (moved to config.py)
- ❌ Hardcoded values

### **3. Deprecated `hyperparameters.py`**

**Marked as deprecated** - now imports from `config.py` for backward compatibility.

```python
# ⚠️  WARNING: Use config.py instead
from config.config import *
```

---

## 📂 New File Structure

```
config/
├── config.py              ⭐ MAIN CONFIG - Use this!
├── training_config.py     → Model creation functions
└── hyperparameters.py     ⚠️  DEPRECATED (backward compatibility)
```

---

## 🎯 How to Use

### **Import Configuration**

```python
# In your code, simply import:
from config.config import *

# Now you have access to:
# - CSV_PATH, BASE_IMAGE_PATH
# - DEVICE, BATCH_SIZE, EPOCHS
# - LEARNING_RATE, LOSS_TYPE
# - All other settings
```

### **Adjust Settings**

Edit **`config/config.py`** only:

```python
# Change batch size
BATCH_SIZE = 16  # For less GPU memory

# Change epochs
EPOCHS = 50

# Change loss function
LOSS_TYPE = 'huber'  # Options: msle, huber, mae, mse

# Change model
IMAGE_MODEL = 'vit_l_16'  # For larger model
```

### **Print Configuration**

```python
from config.config import print_config

print_config()
```

Output:
```
================================================================================
CONFIGURATION
================================================================================

📂 Data:
  CSV: /path/to/data.csv
  Images: /path/to/images

🖥️  Device:
  Device: mps

🏗️  Model:
  Image Encoder: vit_b_16
  Image Size: 224x224
  Output Dims: Image=768, Metadata=256

📊 Training:
  Batch Size: 32
  Epochs: 100
  Learning Rate: 0.0001
  Loss: MSLE
  Weight Transform: LOG (log1p)
  Progressive Training: Freeze 10 epochs

💾 Outputs:
  Checkpoints: checkpoints/
  Logs: logs/
================================================================================
```

---

## ✨ Benefits of Unified Config

✅ **Single source of truth** - No conflicting values  
✅ **Easy to modify** - Change one file  
✅ **Clean imports** - `from config.config import *`  
✅ **No redundancy** - DRY principle  
✅ **Type safety** - All values in one place  
✅ **Clear organization** - Logical sections  

---

## 🔄 Migration Guide

### **Old Way (DEPRECATED)**

```python
# ❌ OLD - Don't use
from config.config import CSV_PATH, DEVICE
from config.training_config import TrainingConfig

batch_size = TrainingConfig.BATCH_SIZE
epochs = TrainingConfig.EPOCHS
```

### **New Way (CLEAN)**

```python
# ✅ NEW - Use this
from config.config import *

# All settings available directly
batch_size = BATCH_SIZE
epochs = EPOCHS
device = DEVICE
```

---

## 📝 Configuration Reference

### **Data Settings**
- `CSV_PATH` - Path to metadata CSV
- `BASE_IMAGE_PATH` - Path to images folder
- `IMAGE_SIZE` - 224 (ViT input size)

### **Model Settings**
- `IMAGE_MODEL` - ViT variant (vit_b_16, vit_l_16, etc.)
- `IMAGE_OUTPUT_DIM` - 768 for ViT-B/16
- `METADATA_OUTPUT_DIM` - 256
- `CATEGORY_EMBEDDING_DIM` - 32
- `FUSION_HIDDEN_DIMS` - [512, 256, 128]
- `DROPOUT_RATE` - 0.2
- `USE_RESIDUAL` - True

### **Training Settings**
- `EPOCHS` - 100
- `BATCH_SIZE` - 32
- `NUM_WORKERS` - 4
- `LEARNING_RATE` - 1e-4
- `WEIGHT_DECAY` - 1e-5
- `GRADIENT_CLIP_NORM` - 1.0

### **Loss & Optimization**
- `LOSS_TYPE` - 'msle' (msle, huber, mae, mse, smooth_l1)
- `USE_LR_SCHEDULER` - True
- `LR_SCHEDULER_PATIENCE` - 10
- `LR_SCHEDULER_FACTOR` - 0.5
- `EARLY_STOPPING_PATIENCE` - 20

### **Preprocessing**
- `USE_LOG_TRANSFORM` - True (CRITICAL!)
- `TRAIN_SPLIT` - 0.7
- `VAL_SPLIT` - 0.1
- `TEST_SPLIT` - 0.2

### **Progressive Training**
- `FREEZE_IMAGE_ENCODER_EPOCHS` - 10

### **Paths**
- `CHECKPOINT_DIR` - "checkpoints"
- `LOGS_DIR` - "logs"

### **Reproducibility**
- `RANDOM_SEED` - 42

---

## 🚀 Ready to Train!

```bash
# Check configuration
python config/config.py

# Run training
python train.py
```

**All settings are now in one clean file!** ✨
