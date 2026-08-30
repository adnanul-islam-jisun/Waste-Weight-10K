# Model Architecture Ablation Study Suite

This suite provides an automated experimental framework to quantify the contribution of every architectural component in the multimodal weight prediction system.

---

## 1. The 6 Experimental Variants

| ID | Experiment Name | Image Encoder | Metadata Encoder | Cross-Modal Fusion | Purpose |
|---|---|---|---|---|---|
| **`exp1`** | **Full Model (Baseline)** | ViT-B/16 (768-dim) | Yes (256-dim) | **Mutual Attention** | Complete proposed architecture |
| **`exp2`** | **Image Only** | ViT-B/16 (768-dim) | None | Direct MLP | Isolates visual contribution without metadata |
| **`exp3`** | **Metadata Only** | None | Yes (256-dim) | Direct MLP | Isolates tabular metadata without visual input |
| **`exp4`** | **No Mutual Attention** | ViT-B/16 (768-dim) | Yes (256-dim) | Naive Concatenation | Measures contribution of bidirectional attention |
| **`exp5`** | **ViT-B/32 Variant** | ViT-B/32 ($32 \times 32$ patches) | Yes (256-dim) | Mutual Attention | Evaluates coarser patch tokenization |
| **`exp6`** | **ViT-L/16 Variant** | ViT-L/16 ($307\text{M}$ params) | Yes (256-dim) | Mutual Attention | Tests scaling to a higher-capacity vision backbone |

---

## 2. Quick Start

```bash
# 1. Validate setup
python ablation/test_ablation_setup.py

# 2. Run interactive menu
bash ablation/quick_start_ablation.sh

# 3. Or run all experiments directly via CLI
python ablation/run_ablation_study.py --all

# 4. Generate comparison plots & LaTeX tables
python ablation/visualize_ablation_results.py
```

### Advanced CLI Options

```bash
# Run specific experiments (e.g. baseline vs no attention)
python ablation/run_ablation_study.py --experiments 1,4

# Resume an interrupted run
python ablation/run_ablation_study.py --resume

# Quick debug run (1 batch per epoch)
python ablation/run_ablation_study.py --debug --experiments 1
```

---

## 3. Results & Outputs

All ablation outputs are stored in `results/ablation/`:

- `summary_report.csv`: Aggregated test metrics across all experiments.
- `summary_report.tex`: Ready-to-use LaTeX table for paper publication.
- `visualizations/`:
  - `mae_comparison.png`
  - `rmse_comparison.png`
  - `metrics_comparison.png`
  - `model_size_vs_accuracy.png`
  - `speed_vs_accuracy.png`
  - `training_curves.png`
  - `weight_range_performance.png`
- `exp*/`: Per-experiment training histories, test predictions, and configs.
