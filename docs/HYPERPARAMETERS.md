# Hyperparameter Reference & Architecture Specifications

This document catalogs all hyperparameters, architectural dimensions, and optimization settings configured in `config/config.py` and `config/training_config.py`.

---

## 1. Network Architecture Parameters

### A. Image Encoder (Vision Transformer)
| Hyperparameter | Value | Description |
|---|---|---|
| Backbone Architecture | `ViT-B/16` | Vision Transformer Base with $16 \times 16$ patch resolution |
| Pretraining Dataset | ImageNet-1K | Initialized from standard PyTorch pretrained weights |
| Input Image Size | $224 \times 224 \times 3$ | Standardized RGB resolution |
| Patch Size | $16 \times 16$ | Produces $14 \times 14 = 196$ patch tokens + 1 `[CLS]` token |
| Transformer Layers | 12 | Stacked self-attention blocks |
| Attention Heads | 12 | Multi-head self-attention mechanisms |
| Embedding Dimension | 768 | ViT-B/16 hidden representation size |
| MLP Hidden Dimension | 3072 | Feedforward expansion dimension |
| Image Output Dimension | 768 | Extracted from `[CLS]` representation |

### B. Metadata Encoder
| Hyperparameter | Value | Description |
|---|---|---|
| Categorical Embedding Dim | 32 | Trainable embedding for 11 material classes |
| Numerical Features Count | 9 (selected) | Physics-informed features (log-volume, compactness, etc.) |
| Numerical MLP Layers | $[128, 64, 32]$ | Fully connected layers with BatchNorm + GELU |
| Concatenated Representation | $32 + 32 = 64$ | Joined categorical embedding + numerical representation |
| Fusion Projection MLP | $64 \to 512 \to 256$ | BatchNorm + ReLU projection |
| Metadata Output Dimension | 256 | Output feature vector feeding fusion network |

### C. Stacked Mutual Attention Fusion
| Hyperparameter | Value | Description |
|---|---|---|
| Cross-Attention Embed Dim | 256 | Shared dimension for cross-modal query/key/value projections |
| Cross-Attention Heads | 8 | Multi-head bidirectional attention |
| Attention Dropout | 0.1 | Dropout applied to cross-attention maps |
| Fusion Hidden Dimensions | $[512, 256, 128]$ | Post-attention regression head |
| Regression Head Dropout | 0.3 | Dropout rate for final prediction MLP |
| Residual Connections | Enabled (`True`) | Skip connections across attention blocks |

---

## 2. Training & Optimization Settings

| Parameter | Configuration | Rationale / Reference |
|---|---|---|
| **Base Epochs** | 120 | Total training budget |
| **Progressive Training** | Freeze 10 Epochs | ViT backbone frozen for first 10 epochs to train metadata & fusion head first |
| **Batch Size** | 64 (GPU $\ge 24\text{GB}$) / 48 / 32 | Auto-scaled based on detected GPU memory |
| **Initial Learning Rate** | $1 \times 10^{-4}$ | AdamW optimizer initial step size |
| **Weight Decay** | $1 \times 10^{-4}$ | $L_2$ regularization penalty |
| **Loss Function** | MSLE (Mean Squared Log Error) | Optimized for wide-range target distributions ($50\text{--}3,450\text{ kg}$) |
| **Target Preprocessing** | Log1p ($y = \ln(1 + w)$) | Normalizes exponential mass distribution to gaussian-like range |
| **Learning Rate Scheduler** | Cosine Annealing with Warm Restarts | $T_0 = 20\text{ epochs}$, $T_{\text{mult}} = 2$, $\eta_{\min} = 1 \times 10^{-7}$ |
| **Gradient Clipping Norm** | 1.0 | Prevents gradient explosion in cross-attention layers |
| **Mixed Precision (AMP)** | Enabled on CUDA (`torch.amp`) | Accelerates training using FP16 / TF32 tensor cores |
| **Exponential Moving Avg (EMA)** | Enabled (`decay = 0.999`) | Maintains shadow model weights for stable evaluation |
| **Early Stopping Patience** | 20 epochs | Monitors validation MSLE loss |
| **Random Seed** | 42 | Applied to Python, NumPy, PyTorch, and CUDA |

---

## 3. Feature Selection: Physics-Informed Numerical Features

| Feature Name | Mathematical Definition | Physical Role |
|---|---|---|
| `log_volume` | $\ln(1 + V_x \cdot V_y \cdot V_z)$ | Primary determinant of object mass ($m \propto \rho \cdot V$) |
| `log_max_dimension` | $\ln(1 + \max(V_x, V_y, V_z))$ | Discriminates elongated vs compact geometry |
| `aspect_ratio_xy` | $V_x / (V_y + \epsilon)$ | 2D projection aspect ratio |
| `aspect_ratio_yz` | $V_y / (V_z + \epsilon)$ | Transverse aspect ratio |
| `compactness` | $\min(V) / (\max(V) + \epsilon)$ | Volumetric compactness descriptor ($1.0 = \text{cube}$) |
| `elongation` | $\max(V) / (\text{median}(V) + \epsilon)$ | Degree of geometric elongation |
| `log_vol_surface_ratio` | $\ln(1 + V / (\text{SurfaceArea} + \epsilon))$ | Distinguishes solid mass from hollow structural shells |
| `log_distance` | $\ln(1 + \sqrt{D_x^2 + D_y^2})$ | Compensates for camera distance foreshortening |
| `surface_sphericity` | $\text{SurfaceArea} \cdot \text{Sphericity}$ | Surface-to-mass interaction term |
