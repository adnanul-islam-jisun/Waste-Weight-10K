"""
Validation Script for Ablation Study Setup
Tests that all components are properly configured before running full study.

Usage:
    python test_ablation_setup.py
"""

import os
import sys
import torch
import importlib

print("\n" + "=" * 80)
print("🔍 ABLATION STUDY SETUP VALIDATION")
print("=" * 80)

errors = []
warnings = []

# ============================================================================
# 1. Check Python Version
# ============================================================================
print("\n1️⃣ Checking Python version...")
if sys.version_info < (3, 8):
    errors.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
else:
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# ============================================================================
# 2. Check Required Modules
# ============================================================================
print("\n2️⃣ Checking required modules...")
required_modules = [
    'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 
    'seaborn', 'tqdm', 'scipy', 'PIL'
]

for module_name in required_modules:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✓ {module_name}: {version}")
    except ImportError:
        errors.append(f"Required module '{module_name}' not installed")

# ============================================================================
# 3. Check Project Files
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n3️⃣ Checking project files...")
required_files = [
    'ablation/ablation_study_config.py',
    'ablation/run_ablation_study.py',
    'ablation/visualize_ablation_results.py',
    'ablation/ablation_utils.py',
    'models/architecture_variants.py',
    'config/config.py',
    'config/training_config.py',
    'models/image_encoder.py',
    'models/metadata_encoder.py',
    'models/multimodal_fusion.py',
    'models/mutual_attention_fusion.py',
    'dataload/data_preprocessing.py',
    'features/feature_engineering.py',
]

for file_rel in required_files:
    file_path = os.path.join(PROJECT_ROOT, file_rel)
    if os.path.exists(file_path):
        print(f"   ✓ {file_rel}")
    else:
        errors.append(f"Missing required file: {file_rel}")

# ============================================================================
# 4. Check GPU Availability
# ============================================================================
print("\n4️⃣ Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available")
    print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    warnings.append("No GPU detected - training will be very slow on CPU")

# ============================================================================
# 5. Check Data Paths
# ============================================================================
print("\n5️⃣ Checking data paths...")
try:
    from config.config import CSV_PATH, BASE_IMAGE_PATH
    
    if os.path.exists(CSV_PATH):
        print(f"   ✓ CSV file found: {CSV_PATH}")
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print(f"   ✓ Loaded {len(df)} records")
    else:
        errors.append(f"CSV file not found: {CSV_PATH}")
    
    if os.path.exists(BASE_IMAGE_PATH):
        print(f"   ✓ Image directory found: {BASE_IMAGE_PATH}")
    else:
        errors.append(f"Image directory not found: {BASE_IMAGE_PATH}")
        
except Exception as e:
    errors.append(f"Error loading data paths: {str(e)}")

# ============================================================================
# 6. Test Model Creation
# ============================================================================
print("\n6️⃣ Testing model creation...")
try:
    from models.architecture_variants import create_image_only_model, create_metadata_only_model
    
    # Test image-only model
    print("   Testing image-only model...")
    model_img = create_image_only_model(
        image_model="vit_b_16",
        image_output_dim=768,
        fusion_hidden_dims=[512, 256, 128],
        dropout=0.3,
        pretrained=False,  # Don't download weights in test
        device="cpu"
    )
    params = sum(p.numel() for p in model_img.parameters())
    print(f"   ✓ Image-only model created ({params/1e6:.1f}M params)")
    del model_img
    
    # Test metadata-only model
    print("   Testing metadata-only model...")
    model_meta = create_metadata_only_model(
        num_categories=10,
        num_numerical_features=20,
        category_embedding_dim=32,
        metadata_output_dim=256,
        fusion_hidden_dims=[512, 256, 128],
        dropout=0.3,
        device="cpu"
    )
    params = sum(p.numel() for p in model_meta.parameters())
    print(f"   ✓ Metadata-only model created ({params/1e6:.1f}M params)")
    del model_meta
    
except Exception as e:
    errors.append(f"Error creating models: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 7. Test Configuration Loading
# ============================================================================
print("\n7️⃣ Testing configuration loading...")
try:
    from ablation_study_config import EXPERIMENTS, ABLATION_EPOCHS, WEIGHT_RANGES
    print(f"   ✓ Found {len(EXPERIMENTS)} experiments configured")
    print(f"   ✓ Training epochs: {ABLATION_EPOCHS}")
    print(f"   ✓ Weight ranges: {list(WEIGHT_RANGES.keys())}")
    
    # List experiments
    for exp_key, exp_info in EXPERIMENTS.items():
        print(f"      - {exp_info['name']}")
        
except Exception as e:
    errors.append(f"Error loading configuration: {str(e)}")

# ============================================================================
# 8. Check Disk Space
# ============================================================================
print("\n8️⃣ Checking disk space...")
try:
    import shutil
    stats = shutil.disk_usage('.')
    free_gb = stats.free / (1024 ** 3)
    print(f"   ✓ Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 10:
        warnings.append(f"Low disk space ({free_gb:.1f} GB) - may need more for checkpoints")
    
except Exception as e:
    warnings.append(f"Could not check disk space: {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 VALIDATION SUMMARY")
print("=" * 80)

if len(errors) == 0 and len(warnings) == 0:
    print("\n✅ ALL CHECKS PASSED!")
    print("\n🚀 Ready to run ablation study:")
    print("   python run_ablation_study.py --all")
    print("\nOr test with a single experiment first:")
    print("   python run_ablation_study.py --debug --experiments 1")
    
elif len(errors) == 0:
    print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
    print("\n⚠️  Warnings:")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")
    print("\n💡 You can proceed, but review warnings above")
    
else:
    print("\n❌ VALIDATION FAILED")
    print("\n❌ Errors:")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    
    if len(warnings) > 0:
        print("\n⚠️  Warnings:")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    print("\n🔧 Please fix the errors above before running ablation study")
    sys.exit(1)

print("=" * 80 + "\n")
