# Model Architecture Ablation Study

Automated ablation study system for evaluating different model architectures in the weight prediction system.

## 📋 Overview

This ablation study systematically evaluates the contribution of different components in our multimodal weight prediction model:

### Experiments

1. **Full Model (Baseline)** - ViT-B/16 + Metadata + Mutual Attention
2. **Image Only** - Only ViT-B/16, no metadata features
3. **Metadata Only** - Only metadata (categories + numerical), no images
4. **No Mutual Attention** - ViT + Metadata with simple concatenation
5. **ViT-B/32** - Faster ViT variant with larger patches
6. **ViT-L/16** - Larger ViT variant with higher capacity

## 🚀 Quick Start

### 1. Run All Experiments

```bash
python run_ablation_study.py --all
```

This will:
- Run all 6 experiments sequentially
- Train each model for 30 epochs
- Save results in `ablation_results/`
- Generate summary CSV and LaTeX table
- Take approximately 4-6 hours total

### 2. Run Specific Experiments

```bash
# Run only experiments 1, 2, and 4
python run_ablation_study.py --experiments 1,2,4
```

### 3. Resume Interrupted Run

```bash
python run_ablation_study.py --resume
```

### 4. Debug Mode (Fast Testing)

```bash
# Test with single batch per epoch
python run_ablation_study.py --debug --experiments 1
```

### 5. Generate Visualizations

After experiments complete:

```bash
python visualize_ablation_results.py
```

This generates:
- MAE and RMSE comparison charts
- Training curves overlay
- Error distribution plots
- Speed vs accuracy scatter plots
- Publication-ready summary table

## 📁 Output Structure

```
ablation_results/
├── summary_report.csv          # Quick overview of all results
├── summary_report.json         # Detailed JSON format
├── summary_report.tex          # LaTeX table for papers
├── progress.json              # Resume tracking
├── visualizations/            # All charts and plots
│   ├── mae_comparison.png
│   ├── rmse_comparison.png
│   ├── training_curves.png
│   ├── error_distribution.png
│   ├── weight_range_performance.png
│   ├── speed_vs_accuracy.png
│   ├── model_size_vs_accuracy.png
│   └── summary_table.png
├── exp1_full_model/
│   ├── config.json            # Experiment configuration
│   ├── test_results.json      # Final test metrics
│   ├── training_log.csv       # Per-epoch training log
│   ├── training_history.json  # Complete training history
│   ├── predictions.csv        # All test predictions
│   └── checkpoints/
│       └── best_model.pt      # Best model weights
├── exp2_image_only/
│   └── ... (same structure)
├── exp3_metadata_only/
│   └── ...
├── exp4_no_attention/
│   └── ...
├── exp5_vit_b32/
│   └── ...
└── exp6_vit_l16/
    └── ...
```

## 📊 Metrics Tracked

### Primary Metrics
- **MAE** - Mean Absolute Error (kg)
- **RMSE** - Root Mean Squared Error (kg)
- **MAPE** - Mean Absolute Percentage Error (%)
- **R²** - R-squared Score

### Secondary Metrics
- Training time (total & per epoch)
- Inference time (ms per sample)
- Model parameters count
- Model size (MB)
- GPU memory usage (MB)
- Best validation loss & epoch

### Per Weight Range
- Light (50-100 kg): MAE, RMSE
- Medium (100-500 kg): MAE, RMSE
- Heavy (500+ kg): MAE, RMSE

## 🔧 Configuration

Edit `ablation_study_config.py` to customize:

```python
# Training settings
ABLATION_EPOCHS = 30  # Number of epochs per experiment
ABLATION_EARLY_STOPPING_PATIENCE = 10
ABLATION_BATCH_SIZE = 32

# Save settings
SAVE_CHECKPOINTS = True  # Set to False to save disk space

# Visualization settings
FIGURE_DPI = 300  # High resolution for publication
FIGURE_FORMAT = "png"
```

## 📝 Example Results

```csv
Experiment,MAE,RMSE,MAPE,R2,Train_Time,Params,GPU_Mem
Full_Model,45.2,67.3,8.5,0.92,1234,98M,4.2GB
Image_Only,78.5,102.1,15.2,0.75,892,86M,3.1GB
Metadata_Only,125.3,156.7,24.8,0.42,345,12M,0.8GB
No_Attention,52.1,74.8,9.8,0.89,1156,98M,3.9GB
ViT_B32,48.9,71.2,9.1,0.91,956,88M,3.8GB
ViT_L16,42.8,64.1,7.9,0.93,1678,316M,6.5GB
```

## 💡 Tips

### For Faster Results
1. Reduce `ABLATION_EPOCHS` to 20
2. Increase `ABLATION_BATCH_SIZE` if you have more GPU memory
3. Run overnight or on weekend
4. Use `--experiments` to test subset first

### For Publication
1. Run with `ABLATION_EPOCHS = 50` for best results
2. Set `FIGURE_DPI = 300` for high-resolution figures
3. Use generated LaTeX table directly in paper
4. Include training curves and error distributions

### Memory Optimization
1. Set `SAVE_CHECKPOINTS = False` to save disk space
2. Experiments run sequentially with GPU memory clearing
3. Each experiment is independent

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size in ablation_study_config.py
ABLATION_BATCH_SIZE = 16  # or 8
```

### Experiment Failed
```bash
# Resume will skip completed experiments
python run_ablation_study.py --resume
```

### Missing Results
```bash
# Check progress file
cat ablation_results/progress.json

# Re-run specific failed experiment
python run_ablation_study.py --experiments 3
```

## 📚 Files in This System

1. **ablation_study_config.py** - All experiment configurations
2. **models/architecture_variants.py** - Image-only and metadata-only models
3. **run_ablation_study.py** - Main orchestration script
4. **ablation_utils.py** - Helper functions and metrics
5. **visualize_ablation_results.py** - Visualization generation

## 🔬 Adding New Experiments

Edit `ablation_study_config.py`:

```python
EXPERIMENTS = {
    # ... existing experiments ...
    
    "exp7_custom": {
        "name": "Custom Experiment",
        "description": "Your custom configuration",
        "config": {
            "use_image": True,
            "use_metadata": True,
            "use_attention_fusion": True,
            # ... your settings ...
        },
        "expected_params": "~XXM",
        "notes": "What this tests"
    }
}
```

Then run:
```bash
python run_ablation_study.py --experiments 7
```

## 📄 Citation

If you use this ablation study system in your research, please cite:

```bibtex
@article{yourpaper2026,
  title={Multimodal Weight Prediction with Ablation Analysis},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## ❓ Questions?

For issues or questions about the ablation study:
1. Check `ablation_results/progress.json` for status
2. Review experiment logs in each experiment folder
3. Verify data paths in `config/config.py`
