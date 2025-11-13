# Weight Prediction System - Multimodal Deep Learning

> **Vision Transformer + Metadata Fusion for Product Weight Estimation**  
> **🚀 GPU-Optimized with Automatic Mixed Precision (AMP)**

Predicts product weights (3.5 - 3,450 kg) using RGB images and metadata features with state-of-the-art Vision Transformers and multimodal fusion architecture.

---

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training (50 epochs, progressive training, GPU-optimized)
python train.py

# 3. Check results
cat checkpoints/training_log_*.csv
```

**🚀 GPU Optimizations:**
- **2-3x faster training** with Automatic Mixed Precision (AMP)
- **50% memory reduction** - train with larger batch sizes
- **Auto batch sizing** based on GPU memory
- **Pin memory + persistent workers** for efficient data loading

See [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) for details.

---

## 🏗️ Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL WEIGHT PREDICTION SYSTEM                       │
│                   Vision Transformer + Metadata Fusion                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA LAYER                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

    CSV: product_metadata.csv (10,421 samples)
    ├── image_path          → Product images (224×224 RGB)
    ├── Type                → Categorical (metal, wood, plastic, etc.)
    ├── V_x, V_y, V_z      → Volume dimensions
    ├── D_x, D_y           → Distance/viewing parameters
    └── weight_in_kg       → Target (3.5 - 3,450 kg)
           │
           ├─► Feature Engineering (features/feature_engineering.py)
           │   ├── volume_proxy = V_x × V_y × V_z
           │   ├── apparent_Vx = V_x / (D_x + ε)
           │   ├── apparent_Vy = V_y / (D_x + ε)
           │   ├── apparent_Vz = V_z / (D_x + ε)
           │   ├── solid_angle_proxy = (V_x × V_y) / (D_x² + ε)
           │   └── view_angle_rad = arctan2(D_y, D_x)
           │   → Total: 11 numerical features
           │
           └─► Preprocessing (config/training_config.py)
               ├── LOG Transformation: log1p(weight)
               ├── Split: 70% train / 10% val / 20% test
               └── Augmentation: Resize, Flip, ColorJitter

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. MODEL ARCHITECTURE (86.8M parameters)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐       ┌──────────────────────────────────┐
│   IMAGE ENCODER             │       │   METADATA ENCODER               │
│   models/image_encoder.py   │       │   models/metadata_encoder.py     │
├─────────────────────────────┤       ├──────────────────────────────────┤
│ Vision Transformer B/16     │       │ Categorical Branch:              │
│ ├─ Input: [B, 3, 224, 224] │       │ ├─ Embedding(Type → 32-dim)     │
│ ├─ Patch: 16×16             │       │ └─ Dropout(0.1)                  │
│ ├─ 12 Transformer Layers    │       │                                  │
│ ├─ Multi-Head Attention     │       │ Numerical Branch:                │
│ ├─ Pretrained (ImageNet)    │       │ ├─ Input: 11 features           │
│ └─ Output: 768-dim          │       │ ├─ MLP [11→64→32]               │
│                             │       │ └─ Output: 256-dim               │
│ Parameters: ~86M            │       │ Parameters: ~15K                 │
└─────────────────────────────┘       └──────────────────────────────────┘
           │                                        │
           └────────────────┬───────────────────────┘
                           │
                           ▼
           ┌────────────────────────────────────────┐
           │   MULTIMODAL FUSION                    │
           │   models/multimodal_fusion.py          │
           ├────────────────────────────────────────┤
           │ Concatenate: [768 + 256] = 1024-dim    │
           │         ↓                              │
           │ Fusion MLP: [1024→512→256→128]        │
           │ ├─ BatchNorm + ReLU + Dropout(0.2)     │
           │ └─ Residual Connections                │
           │         ↓                              │
           │ Regression Head: [128 → 1]             │
           │         ↓                              │
           │ Output: log(weight)                    │
           └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TRAINING STRATEGY (Progressive Training)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    PHASE 1: Warm-up (10 epochs)
    ├─ FREEZE: Image Encoder (ViT)
    ├─ TRAIN: Metadata Encoder + Fusion Network
    └─ LR: 1e-4 (fusion layers)
              ↓
    PHASE 2: Fine-tuning (40 epochs)
    ├─ UNFREEZE: All parameters
    ├─ TRAIN: End-to-end
    └─ LR: 1e-5 (ViT), 1e-5 (fusion)

    Loss: MSLE (Mean Squared Log Error)
    Optimizer: AdamW (weight_decay=1e-5)
    Techniques: Gradient Clipping, Early Stopping

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. OUTPUTS & TRACKING                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    checkpoints/
    ├── training_log_*.csv                ⭐ Track every epoch
    ├── best_model_phase1_*.pt            (Phase 1 best)
    ├── best_model_phase2_*.pt            ⭐ BEST MODEL
    ├── final_model_*.pt                  (Last epoch)
    └── history_*.json                    (All metrics)

    CSV Columns: epoch, phase, train_loss, val_loss, val_mae_kg, 
                 val_rmse_kg, is_best
```

---

## 📂 Project Structure

```
Weight_mannagemner/
│
├── train.py                   ⭐ MAIN TRAINING SCRIPT
│   ├─ WeightPredictionDataset
│   ├─ prepare_data()
│   ├─ train_model()
│   └─ evaluate_model()
│
├── config/
│   ├── config.py              → Paths (CSV, images)
│   ├── training_config.py     ⭐ Hyperparameters & model creation
│   │   ├─ TrainingConfig (BATCH_SIZE=8, EPOCHS=100, etc.)
│   │   ├─ WeightPreprocessor (LOG transformation)
│   │   ├─ create_optimized_model()
│   │   └─ create_trainer_for_your_data()
│   └── hyperparameters.py
│
├── models/
│   ├── image_encoder.py       ⭐ ViT-B/16, ViT-L/16, ViT-H/14
│   ├── metadata_encoder.py    → Categorical + Numerical features
│   ├── multimodal_fusion.py   → Fusion + MultimodalTrainer
│   └── loss_functions.py      ⭐ 10 loss options (MSLE recommended)
│
├── features/
│   ├── feature_engineering.py → Creates 11 engineered features
│   └── feature_selection.py
│
├── Dataload/
│   ├── dataloader.py
│   └── data_preprocessing.py
│
├── utils/
│   ├── helpers.py
│   ├── visualization.py
│   └── metrics.py
│
├── data/
│   ├── product_metadata.csv   → 10,421 records
│   └── images/                → Product images
│
├── checkpoints/               → Saved models & metrics
│   ├── training_log_*.csv     ⭐ Epoch tracking
│   ├── best_model_*.pt        ⭐ Best models
│   └── history_*.json
│
├── requirements.txt
├── README.md                  → This file
├── GPU_OPTIMIZATION_GUIDE.md  ⭐ GPU optimization details
├── SETUP_SUMMARY.md           → Complete setup guide
├── TRAINING_OUTPUT_GUIDE.md   → Output files reference
├── CONFIG_UNIFIED.md          → Config consolidation guide
└── TRAIN_FIXES.md             → Bug fixes log
```

---

## 🚀 Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/adnanul-islam-jisun/Weight_mannagemner.git
cd Weight_mannagemner

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
```

---

## 🎯 Usage

### **Training**

```bash
# Run full training pipeline (50 epochs)
python train.py

# Output:
# - Phase 1: 10 epochs with frozen ViT
# - Phase 2: 40 epochs fine-tuning
# - Saves: models, metrics CSV, training history
```

### **Monitor Training**

```bash
# View training progress
cat checkpoints/training_log_*.csv | column -t -s,

# Watch live updates
tail -f checkpoints/training_log_*.csv
```

### **Load Best Model**

```python
import torch
from config.training_config import create_optimized_model

# Load best model
checkpoint = torch.load('checkpoints/best_model_phase2_TIMESTAMP.pt')

# Create model
model, preprocessor, loss_fn = create_optimized_model(
    num_categories=10,
    num_numerical_features=11,
    device='mps'  # or 'cuda' or 'cpu'
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check performance
print(f"Val MAE: {checkpoint['val_mae']:.2f} kg")
print(f"Val RMSE: {checkpoint['val_rmse']:.2f} kg")
```

### **Make Predictions**

```python
# See predict.py for inference pipeline
python predict.py --image path/to/image.jpg --type metal --features V_x,V_y,V_z,D_x,D_y
```

---

## � GPU Optimizations

This system is **fully optimized for GPU training** with the following enhancements:

### **Automatic Mixed Precision (AMP)**
- ✅ **2-3x faster training** (tested on RTX 3090/4090, A100)
- ✅ **50% memory reduction** - train with larger batch sizes
- ✅ **Minimal accuracy impact** - uses FP16 where safe, FP32 where needed
- ✅ **Automatic gradient scaling** - prevents underflow

### **Auto Batch Size Adjustment**
```python
# Automatically adjusts based on GPU memory
if gpu_memory >= 24GB: BATCH_SIZE = 64
elif gpu_memory >= 16GB: BATCH_SIZE = 32
elif gpu_memory >= 12GB: BATCH_SIZE = 16
elif gpu_memory >= 8GB: BATCH_SIZE = 8
else: BATCH_SIZE = 4
```

### **Optimized Data Loading**
- ✅ **Pin memory** - Faster CPU→GPU transfer (2-3x speedup)
- ✅ **Persistent workers** - Avoid DataLoader process respawn
- ✅ **Prefetch factor** - Pre-load batches while GPU trains
- ✅ **Multi-process loading** - Parallelize data preprocessing

### **CuDNN Optimizations**
- ✅ **cudnn.benchmark = True** - Auto-tune kernels for your hardware
- ✅ **TF32 support** - Faster matrix ops on Ampere GPUs (RTX 30xx+)

### **Performance Monitoring**
```bash
# Training output shows GPU memory usage
Epoch [5/50] | Train: 0.0234 | Val MAE: 45.23kg | GPU Mem: 3.42/4.00GB
```

### **Detailed Guide**
See [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) for:
- Complete optimization details
- Performance benchmarks
- Troubleshooting tips
- Best practices

---

## �📊 Model Performance

### **Dataset Statistics**
- **Samples:** 10,421 records
- **Weight Range:** 3.5 - 3,450 kg (985× ratio)
- **Distribution:** Right-skewed, heavy-tailed, NOT normal
- **Outliers:** ~2% (200 samples)

### **Expected Performance** (After 50 epochs)
| Metric | Target | Excellent |
|--------|--------|-----------|
| **MAE** | < 150 kg | < 100 kg |
| **RMSE** | < 200 kg | < 150 kg |
| **MAPE** | < 20% | < 15% |
| **R²** | > 0.85 | > 0.90 |

---

## ⚙️ Configuration

Edit `config/training_config.py`:

```python
class TrainingConfig:
    # Model
    IMAGE_OUTPUT_DIM = 768           # ViT-B/16 output
    METADATA_OUTPUT_DIM = 256
    
    # Data
    BATCH_SIZE = 8                   # Adjust for GPU memory
    USE_LOG_TRANSFORM = True         # Essential for wide range
    
    # Training
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    LOSS_TYPE = 'msle'               # Recommended
    
    # Progressive Training
    FREEZE_IMAGE_ENCODER_EPOCHS = 10
    
    # Device
    DEVICE = 'mps'  # Auto-detected: mps/cuda/cpu
```

---

## 🔬 Advanced Features

### **Multiple ViT Models Available**
- `vit_b_16` - Base/16 (86M params) ⭐ Default
- `vit_b_32` - Base/32 (88M params, faster)
- `vit_l_16` - Large/16 (304M params, higher capacity)
- `vit_l_32` - Large/32 (306M params, balanced)
- `vit_h_14` - Huge/14 (632M params, maximum capacity)

### **10 Loss Functions Available**
- **MSLE** - Mean Squared Log Error ⭐ Recommended
- Huber - For outliers
- MAE - Most robust
- MSE - Standard
- Smooth L1, MAPE, Quantile, Combined, Log-Cosh, Weighted MAE

### **Data Augmentation**
- Random Horizontal Flip (p=0.5)
- Color Jitter (brightness ±20%, contrast ±20%)
- Resize to 224×224
- ImageNet normalization

---

## 📈 Visualize Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training log
df = pd.read_csv('checkpoints/training_log_TIMESTAMP.csv')

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Validation')
plt.axvline(x=10, color='red', linestyle='--', label='Phase 2 Start')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSLE)')
plt.legend()
plt.title('Training Progress')

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['val_mae_kg'], label='MAE')
plt.plot(df['epoch'], df['val_rmse_kg'], label='RMSE')
plt.xlabel('Epoch')
plt.ylabel('Error (kg)')
plt.legend()
plt.title('Validation Errors')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🐛 Troubleshooting

### **Out of Memory**
```python
# Reduce batch size in config/training_config.py
BATCH_SIZE = 4  # or even 2
```

### **Slow Training**
```python
# Use smaller ViT model
from models.image_encoder import create_fast_image_encoder  # ViT-B/32

# Reduce workers
NUM_WORKERS = 2
```

### **Poor Performance**
- Increase epochs (try 100)
- Check data quality
- Verify LOG transformation is enabled
- Use MSLE loss for wide weight ranges

---

## 📚 Documentation

- **`SETUP_SUMMARY.md`** - Complete setup guide for your dataset
- **`TRAINING_OUTPUT_GUIDE.md`** - Output files reference
- **`TRAIN_FIXES.md`** - Bug fixes and solutions
- **`models/ENCODERS_README.md`** - Model architecture details
- **`models/LOSS_FUNCTIONS_GUIDE.md`** - Loss function selection

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **torchvision** - Pretrained Vision Transformers
- **ViT Paper** - "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

---

## 📞 Contact

**Project Maintainer:** Adnanul Islam Jisun  
**Repository:** [Weight_mannagemner](https://github.com/adnanul-islam-jisun/Weight_mannagemner)

---

## ⭐ Key Features

✅ State-of-the-art Vision Transformer (ViT-B/16)  
✅ Multimodal fusion (images + metadata)  
✅ Progressive training (freeze → fine-tune)  
✅ LOG transformation for wide weight ranges  
✅ MSLE loss optimized for your data  
✅ CSV tracking for every epoch  
✅ Automatic best model selection  
✅ 10 loss function options  
✅ Production-ready inference pipeline  

---

**Ready to predict weights with deep learning! 🚀**
