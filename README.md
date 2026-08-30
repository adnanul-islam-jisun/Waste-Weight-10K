# Multimodal Weight Prediction System for Industrial & Commercial Scrap

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)](docs/REPRODUCIBILITY.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end multimodal deep learning framework for contactless mass and weight estimation of industrial and commercial scrap ($50 - 3,450\text{ kg}$). The system combines **Vision Transformers (ViT-B/16)** with **Stacked Bidirectional Mutual Attention Fusion**, integrating RGB visual features with physics-informed spatial and bounding metadata.

---

## 🌟 Key Highlights

- **Multimodal Mutual Attention**: Bidirectional cross-attention mechanism aligning visual representations with 3D bounding geometry, aspect ratios, and spatial perspective features.
- **Robust Field Performance**: Evaluated against measurement noise ($\pm 1\%$ to $\pm 15\%$) and visual environmental perturbations.
- **Waste-Weight-10K Dataset**: Benchmark comprising 10,421 multimodal pairs across 11 material categories and 75 physical setups.
- **Comprehensive Ablation Suite**: 6 full architectural variants benchmarked under identical splits and random seeds (`seed = 42`).
- **Post-Hoc Explainability**: Optional Stage 2 explanation pipeline utilizing feature attribution and Large Language Models (Llama 3.1 8B).

---

## 🏗️ System Architecture

```
                                    INPUT
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────────────┐
    │         RGB IMAGE             │   │            METADATA                    │
    │       (224 × 224 × 3)         │   │   Category (Type) + 9 Numerical       │
    └───────────────┬───────────────┘   └───────────────────┬───────────────────┘
                    │                                       │
                    ▼                                       ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────────────┐
    │      IMAGE ENCODER            │   │       METADATA ENCODER                │
    │   ViT-B/16 (768-dim)          │   │   Categorical (32d) + MLP (32d)       │
    │   Pretrained ImageNet-1K      │   │   Projected to 256-dim                │
    └───────────────┬───────────────┘   └───────────────────┬───────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                 STACKED BIDIRECTIONAL MUTUAL ATTENTION                    │
    │  • Visual-to-Metadata Attention: Spatial tokens query physical properties │
    │  • Metadata-to-Visual Attention: Physical properties query image regions │
    │  • Multi-Head Cross-Attention (8 heads, 256 embedding dimension)          │
    └───────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                     REGRESSION & PREDICTION HEAD                          │
    │          Dense [512 → 256 → 128] + Dropout (0.3) + Residual Skip          │
    │                         Target: Log-Transformed Weight                    │
    └───────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
                          Predicted Weight (in kg)
```

---

## 📊 Quantitative Benchmarks

### 1. Architecture Ablation Results (Test Set, $N = 1,543$)

| Experiment | Model Architecture | MAE (kg) ↓ | RMSE (kg) ↓ | MAPE (%) ↓ | $R^2$ Score ↑ |
|---|---|---|---|---|---|
| **`exp1`** | **Proposed Full Model (ViT-B/16 + Mutual Attention)** | **117.81** | **191.03** | **8.73%** | **0.95** |
| `exp6` | ViT-L/16 Backbone Variant | 135.78 | 189.62 | 11.04% | 0.95 |
| `exp5` | ViT-B/32 Backbone Variant | 179.97 | 268.96 | 13.63% | 0.90 |
| `exp4` | No Attention (Concatenation Late Fusion) | 1020.74 | 1327.59 | 95.28% | -1.42 |
| `exp3` | Metadata-Only (Tabular Baseline) | 1022.24 | 1325.01 | 98.28% | -1.41 |
| `exp2` | Image-Only (Pure Visual Regression) | 1040.88 | 1346.51 | 100.00% | -1.48 |

### 2. Traditional Tabular ML Baselines (Metadata Only)

| Model | MAE (kg) | RMSE (kg) | MAPE (%) | $R^2$ Score | Within $\pm 50\text{kg}$ (%) |
|---|---|---|---|---|---|
| **Random Forest** | 2.06 | 37.74 | 1.37% | 0.998 | 98.7% |
| **ExtraTrees** | 2.19 | 40.01 | 1.40% | 0.998 | 98.6% |
| **XGBoost** | 2.25 | 40.62 | 1.43% | 0.998 | 98.7% |
| **Gradient Boosting** | 2.42 | 40.89 | 1.43% | 0.998 | 98.7% |
| **LightGBM** | 3.23 | 39.85 | 1.60% | 0.998 | 98.6% |
| **CatBoost** | 5.78 | 38.32 | 1.63% | 0.998 | 98.5% |

---

## 📁 Repository Structure

```
Weight_management/
├── README.md                      # Main overview & quick start
├── requirements.txt               # Pinned dependencies
├── Dockerfile                     # Containerization setup
├── .gitignore                     # Git tracking exclusions
│
├── config/                        # Unified system configuration
│   ├── config.py                  # Global hyperparameters & device settings
│   ├── training_config.py         # Model factory & weight preprocessors
│   └── explanation_config.py      # LLM reasoning configuration
│
├── dataload/                      # Data pipeline & PyTorch datasets
│   └── data_preprocessing.py      # Normalization, transforms & DataLoaders
│
├── features/                      # Physics-informed feature engineering
│   └── feature_engineering.py     # 20 derived geometric & perspective descriptors
│
├── models/                        # Neural network architectures
│   ├── image_encoder.py           # ViT-B/16 wrapper & patch extraction
│   ├── metadata_encoder.py        # Tabular categorical + numerical encoder
│   ├── multimodal_fusion.py       # Baseline late fusion network
│   ├── mutual_attention_fusion.py # Bidirectional mutual cross-attention network
│   ├── architecture_variants.py   # Ablation predictor variants
│   └── loss_functions.py          # MSLE, Huber, and robust regression losses
│
├── scripts/                       # Executable entry points
│   ├── train.py                   # Full end-to-end model training
│   ├── evaluate.py                # Test set evaluation & metrics
│   ├── predict.py                 # Test-Time Augmentation (TTA) inference
│   ├── run_sensitivity_analysis.py# Gaussian measurement noise robustness
│   ├── run_visual_sensitivity_analysis.py # Visual perturbation robustness
│   ├── metadata_only_ml.py        # Classical ML benchmarking
│   ├── train_evaluate_image_only.py # Pure image baseline trainer
│   ├── debug_training.py          # Leakage & normalization diagnostic tool
│   ├── explain.py                 # Post-hoc prediction explanation engine
│   └── run_explain.py             # Interactive explanation CLI
│
├── ablation/                      # Automated ablation study suite
│   ├── README.md                  # Detailed ablation documentation
│   ├── run_ablation_study.py      # Main ablation orchestrator
│   ├── visualize_ablation_results.py # Comparative figures & LaTeX tables
│   ├── test_ablation_setup.py     # Environment pre-flight test
│   ├── quick_start_ablation.sh    # Interactive menu launcher
│   └── run_ablation_background.sh # Tmux background execution script
│
├── visualization/                 # Analysis and paper figure generators
│   ├── visualize_attention.py     # Cross-modal attention heatmaps
│   ├── visualize_attention_v2.py  # ViT rollout & GradCAM visualization
│   ├── visualize_dataset_paper.py # 3D dimension and class distributions
│   ├── visualize_features.py      # Feature correlation & collinearity matrices
│   └── visualize_predictions_explanations.py # Multi-panel paper figures
│
├── results/                       # Empirical artifacts & visual outputs
│   ├── ablation/                  # Ablation logs, summary tables & charts
│   ├── attention_analysis/        # Feature importance & correlation plots
│   ├── dataset_visualizations/    # Class and target distribution figures
│   ├── sensitivity/               # Noise & perturbation benchmark CSVs
│   ├── benchmarks/                # Classical ML benchmark CSVs
│   └── figures/                   # High-resolution architectural figures
│
├── data/                          # Dataset storage
│   ├── README.md                  # Dataset download & formatting guide
│   └── .gitkeep
│
└── docs/                          # Reviewer & technical documentation
    ├── DATASET.md                 # Complete dataset specification & units
    ├── REPRODUCIBILITY.md         # Step-by-step reproduction instructions
    └── HYPERPARAMETERS.md         # Comprehensive hyperparameter catalogue
```

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/adnanul-islam-jisun/Weight_management.git
cd Weight_management

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Setup

Place your dataset inside `./data/` or configure the path:
```bash
export DATA_PATH="/path/to/waste_dataset"
```
*(For layout requirements and units, see [`docs/DATASET.md`](docs/DATASET.md)).*

### 3. Training & Evaluation

```bash
# Train the proposed Full Model (120 epochs, GPU accelerated)
python scripts/train.py

# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best_model_phase2_*.pt

# Run measurement noise robustness test (±1% to ±15% noise)
python scripts/run_sensitivity_analysis.py
```

### 4. Ablation Study

```bash
# Run all 6 ablation experiments
python ablation/run_ablation_study.py --all

# Generate comparative visualizations and LaTeX table
python ablation/visualize_ablation_results.py
```

---

## 📖 In-Depth Documentation

- 📐 [**Dataset Specification (`docs/DATASET.md`)**](docs/DATASET.md): Spatial coordinate conventions, units ($\text{in}$, $\text{kg}$, $\text{m}^3$), load-cell calibration, and physical setup details.
- 🔬 [**Reproducibility Guide (`docs/REPRODUCIBILITY.md`)**](docs/REPRODUCIBILITY.md): Exact command walkthroughs, split protocols, random seeds, and expected benchmark outputs.
- ⚙️ [**Hyperparameters Reference (`docs/HYPERPARAMETERS.md`)**](docs/HYPERPARAMETERS.md): Full listing of optimization settings, layer sizes, and learning rates.
- 🧪 [**Ablation Study Guide (`ablation/README.md`)**](ablation/README.md): Instructions for running and analyzing architectural variants.

---

## 📄 Citation

```bibtex
@article{waste_weight_multimodal_2026,
  title   = {Multimodal Weight Prediction for Industrial Scrap via Vision Transformers and Mutual Attention Fusion},
  author  = {Adnanul Islam et al.},
  journal = {IEEE Transactions on Big Data},
  year    = {2026}
}
```

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
