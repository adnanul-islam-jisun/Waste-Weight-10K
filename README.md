# Weight Prediction System - Multimodal Deep Learning

> **Vision Transformer + Mutual Attention Fusion for Product Weight Estimation**  
> **🚀 GPU-Optimized with Automatic Mixed Precision (AMP)**

Predicts product weights (50 - 3,450 kg) using RGB images and metadata features with state-of-the-art Vision Transformers and **Mutual Attention Fusion** architecture for advanced cross-modal feature interaction.

---

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training (120 epochs, progressive training, GPU-optimized)
python train.py

# 3. Check results
cat checkpoints/training_log_*.csv
```

---

## 🏗️ Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL WEIGHT PREDICTION SYSTEM                         │
│                  Vision Transformer + Stacked Mutual Attention                   │
│                            ~88M Total Parameters                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────────────┐
    │         RGB IMAGE             │   │            METADATA                    │
    │       (224 × 224 × 3)         │   │  Category (Type) + 9 Numerical        │
    └───────────────┬───────────────┘   └───────────────────┬───────────────────┘
                    │                                       │
                    ▼                                       ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────────────┐
    │      IMAGE ENCODER            │   │       METADATA ENCODER                │
    │   ViT-B/16 (Vision            │   │                                       │
    │    Transformer)               │   │  ┌─────────────┐  ┌─────────────────┐ │
    │                               │   │  │ Embedding   │  │  MLP Branch     │ │
    │  ┌─────────────────────────┐  │   │  │ (32-dim)    │  │ 128→64→32       │ │
    │  │ Patch Embedding         │  │   │  │             │  │ BatchNorm+GELU  │ │
    │  │ 16×16 patches → 196     │  │   │  └──────┬──────┘  └────────┬────────┘ │
    │  └───────────┬─────────────┘  │   │         │                  │          │
    │              ▼                │   │         └────────┬─────────┘          │
    │  ┌─────────────────────────┐  │   │                  ▼                    │
    │  │ 12 Transformer Blocks   │  │   │     ┌────────────────────────┐        │
    │  │ • Multi-Head Attention  │  │   │     │    Concatenate         │        │
    │  │ • Layer Normalization   │  │   │     │    32 + 32 = 64        │        │
    │  │ • MLP (3072-dim)        │  │   │     └───────────┬────────────┘        │
    │  │ • 12 Attention Heads    │  │   │                 ▼                     │
    │  └───────────┬─────────────┘  │   │     ┌────────────────────────┐        │
    │              ▼                │   │     │    Fusion MLP          │        │
    │  ┌─────────────────────────┐  │   │     │    64→512→256          │        │
    │  │ [CLS] Token Extraction  │  │   │     │    BatchNorm+ReLU      │        │
    │  │ → 768-dimensional       │  │   │     └───────────┬────────────┘        │
    │  └─────────────────────────┘  │   │                 │                     │
    │                               │   │                 ▼                     │
    │  Parameters: ~86M             │   │     Output: 256-dimensional           │
    │  Pretrained: ImageNet-1K      │   │     Parameters: ~50K                  │
    └───────────────┬───────────────┘   └───────────────────┬───────────────────┘
                    │                                       │
                    │         768-dim                       │        256-dim
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │               ⭐ STACKED MUTUAL ATTENTION FUSION (2 Layers)                  │
    │                         (Bidirectional Cross-Attention)                      │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                              │
    │    Visual Features (768-dim)              Metadata Features (256-dim)        │
    │           │                                        │                         │
    │           ▼                                        ▼                         │
    │    ┌──────────────┐                         ┌──────────────┐                 │
    │    │  Project to  │                         │              │                 │
    │    │   256-dim    │                         │   256-dim    │                 │
    │    └──────┬───────┘                         └──────┬───────┘                 │
    │           │                                        │                         │
    │           ▼                                        ▼                         │
    │  ┌─────────────────────────────────────────────────────────────────────┐    │
    │  │              BIDIRECTIONAL CROSS-ATTENTION (Layer 1 & 2)            │    │
    │  │                                                                      │    │
    │  │   ┌────────────────────────┐      ┌────────────────────────┐        │    │
    │  │   │  Visual → Metadata     │      │  Metadata → Visual     │        │    │
    │  │   │                        │      │                        │        │    │
    │  │   │  Q: Visual features    │      │  Q: Metadata features  │        │    │
    │  │   │  K,V: Metadata         │      │  K,V: Visual           │        │    │
    │  │   │                        │      │                        │        │    │
    │  │   │  "What metadata is     │      │  "What visual features │        │    │
    │  │   │   relevant to this     │      │   support this         │        │    │
    │  │   │   visual pattern?"     │      │   metadata?"           │        │    │
    │  │   │                        │      │                        │        │    │
    │  │   │  8 Attention Heads     │      │  8 Attention Heads     │        │    │
    │  │   │  Head dim: 32          │      │  Head dim: 32          │        │    │
    │  │   └───────────┬────────────┘      └───────────┬────────────┘        │    │
    │  │               │                               │                      │    │
    │  │               ▼                               ▼                      │    │
    │  │         LayerNorm                       LayerNorm                    │    │
    │  │               │                               │                      │    │
    │  └───────────────┴───────────────┬───────────────┴──────────────────────┘    │
    │                                  │                                           │
    │                                  ▼                                           │
    │    ┌─────────────────────────────────────────────────────────────────────┐  │
    │    │                    FEATURE CONCATENATION                            │  │
    │    │                                                                      │  │
    │    │  [V→M Attended] + [M→V Attended] + [V Residual] + [M Residual]      │  │
    │    │      256-dim   +     256-dim    +    256-dim   +    256-dim         │  │
    │    │                                                                      │  │
    │    │                    = 1024-dimensional                                │  │
    │    └──────────────────────────────┬──────────────────────────────────────┘  │
    │                                   │                                          │
    │                                   ▼                                          │
    │    ┌─────────────────────────────────────────────────────────────────────┐  │
    │    │                      FUSION MLP                                      │  │
    │    │              1024 → 512 → ReLU → Dropout(0.2)                       │  │
    │    │              512 → 256 → ReLU → LayerNorm                           │  │
    │    └──────────────────────────────┬──────────────────────────────────────┘  │
    │                                   │                                          │
    │                         Output: 256-dimensional                              │
    └───────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         REGRESSION HEAD                                      │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                              │
    │     256 → Linear → GELU → LayerNorm → Dropout(0.1)                           │
    │             ↓                                                                │
    │     128 → Linear → GELU → LayerNorm → Dropout(0.06)                          │
    │             ↓                                                                │
    │      64 → Linear → GELU → Dropout(0.04)                                      │
    │             ↓                                                                │
    │           1 (Softplus → log(weight) prediction)                              │
    │                                                                              │
    └───────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   expm1(pred)   │  ← Inverse log transform
                              │   = weight (kg) │
                              └─────────────────┘
```

---

## 📊 Feature Engineering Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING (20 Features)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  RAW INPUT: V_x, V_y, V_z (Volume dims), D_x, D_y (Distance)                    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ SIZE FEATURES (5) - Primary weight determinants                             ││
│  │                                                                              ││
│  │  log_volume        = log1p(V_x × V_y × V_z)                                 ││
│  │  log_surface_area  = log1p(2×(V_x×V_y + V_y×V_z + V_x×V_z))                ││
│  │  max_dimension     = max(V_x, V_y, V_z)                                     ││
│  │  log_max_dimension = log1p(max_dimension)                                   ││
│  │  log_geo_mean_dim  = log1p((V_x × V_y × V_z)^(1/3))                         ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ SHAPE FEATURES (8) - Density & structural indicators                        ││
│  │                                                                              ││
│  │  aspect_ratio_xy   = V_x / (V_y + ε)                                        ││
│  │  aspect_ratio_xz   = V_x / (V_z + ε)                                        ││
│  │  aspect_ratio_yz   = V_y / (V_z + ε)                                        ││
│  │  compactness       = min_dim / (max_dim + ε)     [0=elongated, 1=cube]     ││
│  │  flatness          = min_dim / (mid_dim + ε)                                ││
│  │  elongation        = max_dim / (mid_dim + ε)                                ││
│  │  sphericity        = (π^⅓ × (6V)^⅔) / (A + ε)   [closeness to sphere]     ││
│  │  log_vol_surf_ratio= log1p(volume / (surface_area + ε))                     ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ PERSPECTIVE FEATURES (4) - Camera/viewing adjustments                       ││
│  │                                                                              ││
│  │  log_distance        = log1p(√(D_x² + D_y²))                                ││
│  │  log_apparent_volume = log1p(volume / (D_x² + ε))   [inverse square law]   ││
│  │  view_angle_rad      = arctan2(D_y, D_x)                                    ││
│  │  depth_ratio         = D_x / (distance + ε)                                 ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ INTERACTION FEATURES (3) - Non-linear relationships                         ││
│  │                                                                              ││
│  │  volume_compactness  = log_volume × compactness                             ││
│  │  surface_sphericity  = log_surface_area × sphericity                        ││
│  │  size_dist_interact  = log_volume / (log_distance + ε)                      ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  OUTPUT: 20 engineered features                                                 │
│          → 9 selected features after correlation filtering                      │
│          → Normalized via StandardScaler (mean=0, std=1)                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ SELECTED FEATURES (9) - Used for training after removing correlations      ││
│  │                                                                              ││
│  │  ✓ log_volume          (size - primary weight determinant)                  ││
│  │  ✓ log_max_dimension   (size)                                               ││
│  │  ✓ aspect_ratio_xy     (shape)                                              ││
│  │  ✓ aspect_ratio_yz     (shape)                                              ││
│  │  ✓ compactness         (shape - how cube-like)                              ││
│  │  ✓ elongation          (shape - how stretched)                              ││
│  │  ✓ log_vol_surface_ratio (shape)                                            ││
│  │  ✓ log_distance        (perspective)                                        ││
│  │  ✓ surface_sphericity  (interaction)                                        ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING STRATEGY                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: WARM-UP (Epochs 1-10)                                         │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                          │    │
│  │  🔒 FROZEN:  Image Encoder (ViT) - 86M parameters locked                │    │
│  │  🔓 TRAINED: Metadata Encoder + Attention Block + Regression Head       │    │
│  │                                                                          │    │
│  │  Learning Rate: 1e-4                                                     │    │
│  │  Purpose: Learn fusion without disturbing pretrained ViT weights        │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: FINE-TUNING (Epochs 11-120)                                   │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                          │    │
│  │  🔓 UNFROZEN: All parameters trainable                                  │    │
│  │                                                                          │    │
│  │  Learning Rates (Differentiated):                                        │    │
│  │    • Image Encoder: 2e-5 (×0.2 reduction - careful fine-tuning)         │    │
│  │    • Other layers:  5e-5 (×0.5 reduction - maintain adaptation)         │    │
│  │                                                                          │    │
│  │  Purpose: End-to-end optimization with ViT adaptation                   │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  OPTIMIZATION SETTINGS                                                   │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                          │    │
│  │  Optimizer:      AdamW (weight_decay=1e-4)                              │    │
│  │  Loss Function:  MSLE (Mean Squared Log Error)                          │    │
│  │  Target Transform: log1p(weight) → expm1(prediction)                    │    │
│  │  LR Scheduler:   CosineAnnealingWarmRestarts (T_0=20, T_mult=2)         │    │
│  │  Gradient Clip:  max_norm=1.0                                           │    │
│  │  Mixed Precision: AMP (FP16) on CUDA                                    │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Loss Function: MSLE (Mean Squared Log Error)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MSLE - OPTIMAL FOR WIDE WEIGHT RANGES                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Formula:   MSLE = mean( (log(1 + pred) - log(1 + target))² )                   │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  WHY MSLE IS OPTIMAL FOR WEIGHT PREDICTION:                             │    │
│  │                                                                          │    │
│  │  Weight Range: 50 kg → 3,450 kg (69× ratio)                             │    │
│  │                                                                          │    │
│  │  Problem with MAE/MSE:                                                   │    │
│  │    • 100kg error on 3000kg object: 3.3% relative error                  │    │
│  │    • 100kg error on 100kg object:  100% relative error                  │    │
│  │    • MAE/MSE treats both equally → model ignores light objects          │    │
│  │                                                                          │    │
│  │  MSLE Solution:                                                          │    │
│  │    • Works in LOG space: log(100) ≈ 4.6, log(3000) ≈ 8.0               │    │
│  │    • 10% relative error = same loss regardless of absolute weight       │    │
│  │    • Automatically balances learning across weight range                │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Implementation:                                                                 │
│    1. Target: weight_log = log1p(weight_kg)                                     │
│    2. Model predicts: pred_log                                                  │
│    3. Loss: mean((pred_log - weight_log)²)                                      │
│    4. Inference: weight_kg = expm1(pred_log)                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Attention Mechanism Details

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD CROSS-ATTENTION (8 Heads)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V                                │
│                                                                                  │
│  Where:                                                                          │
│    • Q (Query):   256-dim → 8 heads × 32-dim                                    │
│    • K (Key):     256-dim → 8 heads × 32-dim                                    │
│    • V (Value):   256-dim → 8 heads × 32-dim                                    │
│    • d_k = 32 (head dimension)                                                  │
│    • √d_k scaling prevents gradient explosion                                   │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  BRANCH 1: Visual → Metadata                                            │    │
│  │                                                                          │    │
│  │  Query:     Visual features (what visual pattern am I?)                 │    │
│  │  Key/Value: Metadata features (what attributes are available?)          │    │
│  │  Output:    Visual features enriched with relevant metadata context     │    │
│  │                                                                          │    │
│  │  Example: "This looks like a large cylindrical shape → attend to        │    │
│  │           'cylinder' category and high volume features"                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  BRANCH 2: Metadata → Visual                                            │    │
│  │                                                                          │    │
│  │  Query:     Metadata features (what attributes do I have?)             │    │
│  │  Key/Value: Visual features (what visual evidence supports this?)       │    │
│  │  Output:    Metadata features enriched with visual confirmation         │    │
│  │                                                                          │    │
│  │  Example: "Type='metal' + high volume → attend to shiny surface         │    │
│  │           and dense appearance patterns"                                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Weight_management/
│
├── train.py                   ⭐ MAIN TRAINING SCRIPT
│   ├─ train_model()           → Progressive training with AMP
│   ├─ evaluate_model()        → Test set evaluation
│   └─ Main pipeline           → Data loading → Training → Evaluation
│
├── predict.py                 → Inference pipeline
│
├── config/
│   ├── config.py              ⭐ UNIFIED CONFIGURATION
│   │   ├─ Device auto-detection (CUDA/MPS/CPU)
│   │   ├─ Model architecture settings
│   │   ├─ Training hyperparameters
│   │   └─ GPU optimizations
│   ├── training_config.py     → Model creation & weight preprocessing
│   └── hyperparameters.py     → Additional hyperparameters
│
├── models/
│   ├── image_encoder.py       ⭐ Vision Transformer (ViT-B/16)
│   │   ├─ 86M parameters
│   │   ├─ 768-dim output
│   │   └─ ImageNet pretrained
│   │
│   ├── metadata_encoder.py    → Categorical + Numerical encoder
│   │   ├─ Embedding layer (32-dim)
│   │   ├─ MLP for numerical features
│   │   └─ 256-dim output
│   │
│   ├── mutual_attention_fusion.py  ⭐ CROSS-ATTENTION FUSION
│   │   ├─ MultiHeadCrossAttention
│   │   ├─ MutualAttentionBlock
│   │   └─ MultimodalWeightPredictor_WithAttention
│   │
│   ├── multimodal_fusion.py   → Late fusion (simple concatenation)
│   └── loss_functions.py      ⭐ 10 loss options (MSLE recommended)
│
├── features/
│   ├── feature_engineering.py → Creates 20 physics-informed features
│   │   ├─ Size features (5)
│   │   ├─ Shape features (8)
│   │   ├─ Perspective features (4)
│   │   └─ Interaction features (3)
│   └── feature_selection.py
│
├── Dataload/
│   ├── data_preprocessing.py  ⭐ Data pipeline
│   │   ├─ StandardScaler normalization
│   │   ├─ Train/Val/Test splits
│   │   ├─ Image augmentation
│   │   └─ DataLoader creation
│   └── dataloader.py
│
├── utils/
│   ├── helpers.py
│   ├── visualization.py
│   └── metrics.py
│
├── checkpoints/               → Saved models & metrics
│   ├── training_log_*.csv     ⭐ Epoch-by-epoch metrics
│   ├── best_model_phase1_*.pt → Best model (Phase 1)
│   ├── best_model_phase2_*.pt ⭐ BEST MODEL (use this!)
│   ├── latest_checkpoint.pt   → Resume training
│   └── history_*.json         → Training history
│
└── requirements.txt
```

---

## ⚙️ Configuration Summary

| Category | Parameter | Value | Description |
|----------|-----------|-------|-------------|
| **Model** | IMAGE_MODEL | `vit_b_16` | Vision Transformer Base/16 |
| | IMAGE_OUTPUT_DIM | 768 | ViT output dimension |
| | METADATA_OUTPUT_DIM | 256 | Metadata encoder output |
| | ATTENTION_NUM_HEADS | 8 | Cross-attention heads |
| | ATTENTION_EMBED_DIM | 256 | Attention embedding dim |
| **Training** | EPOCHS | 120 | Total training epochs |
| | BATCH_SIZE | Auto (16-128) | Based on GPU memory |
| | LEARNING_RATE | 1e-4 | Initial learning rate |
| | WEIGHT_DECAY | 1e-4 | AdamW regularization |
| | FREEZE_EPOCHS | 10 | Phase 1 frozen epochs |
| **Loss** | LOSS_TYPE | `msle` | Mean Squared Log Error |
| | USE_LOG_TRANSFORM | True | log1p target transform |
| **Scheduler** | TYPE | `cosine_warm` | Cosine Annealing Warm Restarts |
| | T_0 | 20 | Initial restart period |
| | T_MULT | 2 | Period multiplier |
| **GPU** | USE_AMP | True (CUDA) | Automatic Mixed Precision |
| | PIN_MEMORY | True (CUDA) | Faster data transfer |

---

## 🚀 GPU Optimizations

| Feature | Description | Speedup |
|---------|-------------|---------|
| **AMP (FP16)** | Automatic Mixed Precision training | 2-3× faster |
| **CuDNN Benchmark** | Auto-tune convolution algorithms | 10-20% faster |
| **TF32** | Tensor Float 32 on Ampere GPUs | 3× matmul speedup |
| **Pin Memory** | Faster CPU→GPU data transfer | 2-3× data loading |
| **Persistent Workers** | Keep DataLoader workers alive | Reduced overhead |
| **Auto Batch Size** | Adjust based on GPU memory | Optimal memory usage |

---

## 📊 Model Parameters

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PARAMETER BREAKDOWN                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Component                          Parameters       Trainable                   │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Image Encoder (ViT-B/16)           86,567,680      Phase 1: ❌  Phase 2: ✅    │
│    ├── Patch Embedding                  590,592                                  │
│    ├── Transformer Blocks (12×)      85,054,464                                  │
│    └── Class Token + Position           151,296                                  │
│                                                                                  │
│  Metadata Encoder                       ~50,000      ✅ Always trainable         │
│    ├── Category Embedding               ~5,120       (num_types × 32)            │
│    ├── Numerical MLP                   ~20,000       (128→64→32)                 │
│    └── Fusion Layer                    ~25,000       (64→512→256)                │
│                                                                                  │
│  Stacked Mutual Attention (2x)       ~1,600,000      ✅ Always trainable         │
│    ├── Visual→Metadata Attention       ~400,000      (2 layers)                  │
│    ├── Metadata→Visual Attention       ~400,000      (2 layers)                  │
│    ├── Residual Projections            ~520,000      (2 layers)                  │
│    └── Fusion MLP                      ~280,000      (2 layers)                  │
│                                                                                  │
│  Regression Head                        ~25,000      ✅ Always trainable         │
│    └── 256→128→64→1                                                              │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  TOTAL                              ~88,200,000                                  │
│  Phase 1 Trainable                  ~1,700,000      (2% of total)                │
│  Phase 2 Trainable                  ~88,200,000     (100% of total)              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Expected Performance

| Metric | Target | Excellent | Notes |
|--------|--------|-----------|-------|
| **MAE** | < 150 kg | < 100 kg | Mean Absolute Error |
| **RMSE** | < 200 kg | < 150 kg | Root Mean Squared Error |
| **MAPE** | < 20% | < 15% | Mean Absolute Percentage Error |
| **R²** | > 0.85 | > 0.90 | Coefficient of Determination |

---

## 📖 Usage

### Training

```bash
# Start fresh training
rm -f checkpoints/latest_checkpoint.pt
python train.py

# Resume from checkpoint
python train.py  # Automatically resumes if checkpoint exists
```

### Monitor Training

```bash
# View training progress
cat checkpoints/training_log_*.csv | column -t -s,

# Watch live updates
tail -f checkpoints/training_log_*.csv
```

### Load Best Model

```python
import torch
from config.training_config import create_optimized_model

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model_phase2_TIMESTAMP.pt')

# Create model
model, preprocessor, loss_fn = create_optimized_model(
    num_categories=15,
    num_numerical_features=9,  # Selected features after correlation filtering
    device='cuda'
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check performance
print(f"Val MAE: {checkpoint['val_mae']:.2f} kg")
print(f"Val RMSE: {checkpoint['val_rmse']:.2f} kg")
```

---

## 🔄 Data Flow Summary

```
Input Image (224×224×3) + Metadata (Type + 9 selected features)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  1. Feature Extraction        │
    │     ViT: 86M params → 768-dim │
    │     MLP: 50K params → 256-dim │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  2. Mutual Attention Fusion   │
    │     8-head cross-attention    │
    │     Bidirectional interaction │
    │     → 256-dim fused features  │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  3. Regression Head           │
    │     256 → 128 → 64 → 1        │
    │     → log(weight) prediction  │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  4. Inverse Transform         │
    │     expm1(pred) = weight (kg) │
    └───────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size in config/config.py
BATCH_SIZE = 16  # or even 8
```

### Slow Training
```python
# Use ViT-B/32 (faster, larger patches)
IMAGE_MODEL = 'vit_b_32'

# Reduce workers
NUM_WORKERS = 2
```

### Poor Performance
1. Increase epochs (try 100+)
2. Check that `USE_LOG_TRANSFORM = True`
3. Verify `LOSS_TYPE = 'msle'`
4. Ensure data has valid weights (> 50 kg)

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
**Repository:** [Weight_management](https://github.com/adnanul-islam-jisun/Weight_management)

---

**Ready to predict weights with deep learning! 🚀**
