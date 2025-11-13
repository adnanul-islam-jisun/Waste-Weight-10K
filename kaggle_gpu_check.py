"""
Kaggle GPU Diagnostic Script
Run this to check if GPU is properly configured
"""

import torch
import os
import sys

print("="*80)
print("KAGGLE GPU DIAGNOSTIC")
print("="*80)

# 1. Check PyTorch CUDA
print("\n1. PyTorch CUDA Check:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   PyTorch Version: {torch.__version__}")

if not torch.cuda.is_available():
    print("\n   ❌ CUDA NOT AVAILABLE!")
    print("   ⚠️  ACTION REQUIRED:")
    print("      1. Go to Settings (right sidebar)")
    print("      2. Under 'Accelerator', select 'GPU T4 x2'")
    print("      3. Click 'Save'")
    print("      4. Notebook will restart with GPU")
    sys.exit(1)

# 2. Check GPU details
print("\n2. GPU Details:")
gpu_count = torch.cuda.device_count()
print(f"   GPU Count: {gpu_count}")

for i in range(gpu_count):
    print(f"\n   GPU {i}:")
    print(f"     Name: {torch.cuda.get_device_name(i)}")
    mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"     Memory: {mem_gb:.1f} GB")
    print(f"     Compute Capability: {torch.cuda.get_device_capability(i)}")

# 3. Check device setting
print("\n3. Config Device Check:")
try:
    from config.config import DEVICE
    print(f"   DEVICE: {DEVICE}")
    if DEVICE != "cuda":
        print(f"   ❌ DEVICE is '{DEVICE}', should be 'cuda'!")
    else:
        print(f"   ✅ DEVICE correctly set to 'cuda'")
except Exception as e:
    print(f"   ❌ Error importing DEVICE: {e}")

# 4. Check dataset paths
print("\n4. Dataset Path Check:")
dataset_path = "/kaggle/input/disaster/waste_dataset"
csv_path = "/kaggle/input/disaster/waste_dataset/image.csv"

if os.path.exists(dataset_path):
    print(f"   ✅ Dataset found: {dataset_path}")
    contents = os.listdir(dataset_path)
    print(f"   Files/folders: {len(contents)}")
    print(f"   Contents: {contents[:5]}")
    
    if os.path.exists(csv_path):
        print(f"   ✅ CSV found: {csv_path}")
    else:
        print(f"   ❌ CSV not found: {csv_path}")
        print(f"      Available files: {[f for f in contents if f.endswith('.csv')]}")
else:
    print(f"   ❌ Dataset not found: {dataset_path}")
    print(f"   ⚠️  ACTION REQUIRED:")
    print(f"      1. Go to Settings > Input")
    print(f"      2. Click 'Add Input'")
    print(f"      3. Select your dataset")

# 5. Test GPU computation
print("\n5. GPU Computation Test:")
try:
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"   ✅ GPU computation successful!")
    print(f"   Matrix multiply time: {elapsed*1000:.2f} ms")
    print(f"   GPU Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
except Exception as e:
    print(f"   ❌ GPU computation failed: {e}")

# 6. Check batch size
print("\n6. Batch Size Check:")
try:
    from config.config import BATCH_SIZE
    print(f"   BATCH_SIZE: {BATCH_SIZE}")
    
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if mem_gb >= 14:  # T4
        recommended = 16
        if BATCH_SIZE > 32:
            print(f"   ⚠️  Batch size might be too large for T4!")
            print(f"   Recommended: {recommended} for ViT-B/16")
        elif BATCH_SIZE < 8:
            print(f"   ⚠️  Batch size is small, GPU underutilized")
            print(f"   Recommended: {recommended} for ViT-B/16")
        else:
            print(f"   ✅ Batch size is reasonable for T4")
except Exception as e:
    print(f"   ⚠️  Could not check batch size: {e}")

# 7. Summary
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

checks = []
checks.append(("CUDA Available", torch.cuda.is_available()))
checks.append(("GPU Detected", torch.cuda.device_count() > 0))

try:
    from config.config import DEVICE
    checks.append(("DEVICE = 'cuda'", DEVICE == "cuda"))
except:
    checks.append(("DEVICE = 'cuda'", False))

checks.append(("Dataset exists", os.path.exists(dataset_path)))
checks.append(("CSV exists", os.path.exists(csv_path)))

all_passed = all(check[1] for check in checks)

for name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {name}")

print("="*80)

if all_passed:
    print("\n🎉 ALL CHECKS PASSED!")
    print("   Your Kaggle environment is ready for GPU training!")
    print("\n   Run: python train.py")
else:
    print("\n⚠️  SOME CHECKS FAILED!")
    print("   Please fix the issues above before training.")
    print("\n   Most common fix:")
    print("   → Settings > Accelerator > GPU T4 x2 > Save")

print("="*80)
