# Reproducibility Guide & Benchmark Audit

This guide provides end-to-end instructions for independent researchers and reviewers to reproduce all empirical findings, benchmark tables, and ablation figures reported in the paper.

---

## 1. System Requirements & Determinism

### Hardware Requirements
- **GPU**: NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (Tested on RTX 3090 24GB, RTX 4090 24GB, V100 16GB, and RTX 3060 12GB).
- **RAM**: Minimum 16 GB system memory.
- **Disk**: $\sim 5\text{ GB}$ for dataset and model checkpoints.

### Seed Determinism
All scripts set fixed random seeds across all pseudo-random generators:
```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
```

---

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/adnanul-islam-jisun/Weight_management.git
cd Weight_management

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Data Preparation & Leakage-Free Preprocessing

1. Place the dataset inside `data/` or set the `DATA_PATH` environment variable:
   ```bash
   export DATA_PATH="/path/to/waste_dataset"
   ```
2. **Split Protocol**:
   - **Train Split**: $70\%$
   - **Validation Split**: $15\%$
   - **Test Split**: $15\%$
   - Stratified by material category using `train_test_split(..., random_state=42)`.
3. **Data Leakage Safeguards**:
   - `StandardScaler` for numerical features is fitted **strictly on the training split** and applied transform-only to validation and test splits.
   - Target weight is trained in log space: $y = \ln(1 + \text{weight})$, and inverted at inference: $\hat{w} = \exp(\hat{y}) - 1$.

---

## 4. End-to-End Reproduction Commands

### A. Train the Proposed Full Model (Vision Transformer + Mutual Attention)
```bash
python scripts/train.py
```
- **Output Checkpoint**: `checkpoints/best_model_phase2_*.pt`
- **Output Log**: `checkpoints/training_log_*.csv`
- **Expected Duration**: $\approx 45 - 90\text{ minutes}$ on a single modern GPU.

### B. Evaluate Test Set Performance
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model_phase2_*.pt
```
Expected output metrics on test set ($N \approx 1,543$ samples):
- **MAE**: $\approx 64.9 - 88.1\text{ kg}$
- **RMSE**: $\approx 144.8 - 191.0\text{ kg}$
- **MAPE**: $\approx 4.9\% - 8.7\%$
- **$R^2$ Score**: $\approx 0.95 - 0.98$

### C. Run Metadata Sensitivity & Robustness Analysis
Simulates field measurement noise ($\pm 1\%$ to $\pm 15\%$ Gaussian perturbation on $V_x, V_y, V_z, D_x, D_y$):
```bash
python scripts/run_sensitivity_analysis.py --noise-type gaussian --trials 5
```
Results saved to `results/sensitivity/sensitivity_results.csv`.

### D. Run Visual Sensitivity Analysis
Simulates environmental perturbations (lighting jitter, brightness variation, rotations):
```bash
python scripts/run_visual_sensitivity_analysis.py
```
Results saved to `results/sensitivity/visual_sensitivity_results.csv`.

### E. Run Traditional ML Benchmarks (Metadata Baselines)
Trains Random Forest, ExtraTrees, XGBoost, LightGBM, CatBoost, Gradient Boosting, MLP, and Ridge:
```bash
python scripts/metadata_only_ml.py
```
Results saved to `results/benchmarks/metadata_ml_benchmark_results.csv`.

### F. Run Complete 6-Experiment Ablation Suite
Executes the full automated ablation grid:
1. Full Model (ViT-B/16 + Metadata + Stacked Mutual Attention)
2. Image Only (pure ViT regression)
3. Metadata Only (pure tabular MLP)
4. No Attention (naive concatenation late fusion)
5. ViT-B/32 backbone variant
6. ViT-L/16 backbone variant

```bash
# Automated run of all 6 experiments
python ablation/run_ablation_study.py --all

# Generate comparative figures and LaTeX summary table
python ablation/visualize_ablation_results.py
```
Outputs are compiled in `results/ablation/summary_report.csv` and `results/ablation/summary_report.tex`.

---

## 5. Summary of Main Experimental Results

| Model Architecture | Image Backbone | Cross-Modal Fusion | MAE (kg) | RMSE (kg) | MAPE (%) | $R^2$ |
|---|---|---|---|---|---|---|
| **Proposed Full Model** | **ViT-B/16** | **Mutual Attention** | **117.81** | **191.03** | **8.73%** | **0.95** |
| ViT-L/16 Variant | ViT-L/16 | Mutual Attention | 135.78 | 189.62 | 11.04% | 0.95 |
| ViT-B/32 Variant | ViT-B/32 | Mutual Attention | 179.97 | 268.96 | 13.63% | 0.90 |
| No Attention (Concat) | ViT-B/16 | Direct Concatenation | 1020.74 | 1327.59 | 95.28% | -1.42 |
| Metadata-Only Baseline | None | Tabular MLP | 1022.24 | 1325.01 | 98.28% | -1.41 |
| Image-Only Baseline | ViT-B/16 | None | 1040.88 | 1346.51 | 100.00% | -1.48 |
