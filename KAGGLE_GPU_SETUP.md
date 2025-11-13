# Kaggle GPU Setup Guide - T4 x2 Optimization

## 🎯 How to Enable Kaggle GPU (T4 x2)

### **Step 1: Enable GPU in Notebook Settings**

**CRITICAL:** You must enable GPU acceleration in Kaggle notebook settings!

1. Click **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2**
3. Click **Save**
4. Notebook will restart with GPU enabled

### **Step 2: Verify GPU is Available**

Run this in your Kaggle notebook:

```python
import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"  Compute: {torch.cuda.get_device_capability(i)}")
else:
    print("❌ NO GPU DETECTED!")
    print("   Go to Settings > Accelerator > Select 'GPU T4 x2'")
```

**Expected Output:**
```
CUDA Available: True
CUDA Version: 12.1
GPU Count: 2

GPU 0:
  Name: Tesla T4
  Memory: 15.0 GB
  Compute: (7, 5)

GPU 1:
  Name: Tesla T4
  Memory: 15.0 GB
  Compute: (7, 5)
```

### **Step 3: Fix Common Issues**

#### **Issue 1: "No GPU Detected" (CUDA Available: False)**

**Solution:**
```python
# In Kaggle notebook:
# 1. Go to Settings (right sidebar)
# 2. Accelerator > Select "GPU T4 x2"
# 3. Click Save
# 4. Notebook will restart
```

#### **Issue 2: "GPU Enabled but Not Used in Training"**

**Check these:**

```python
# 1. Check DEVICE is set to cuda
from config.config import DEVICE
print(f"Using device: {DEVICE}")  # Should be "cuda"

# 2. Check model is on GPU
import torch
model = ...  # your model
print(f"Model device: {next(model.parameters()).device}")  # Should be "cuda:0"

# 3. Check data is moving to GPU
for batch in train_loader:
    print(f"Batch device: {batch['image'].device}")  # Should be "cuda:0"
    break
```

#### **Issue 3: "Only Using 1 GPU (not both T4s)"**

**Note:** PyTorch uses 1 GPU by default. To use both GPUs:

```python
# Option 1: DataParallel (Simple)
import torch.nn as nn

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to('cuda')

# Option 2: DistributedDataParallel (Advanced - Better performance)
# See PyTorch DDP documentation
```

**For this project:** Using 1 GPU is fine! T4 has 15GB, enough for ViT-B/16.

### **Step 4: Kaggle Dataset Paths**

Your paths are correct:

```python
CSV_PATH = "/kaggle/input/disaster/waste_dataset/image.csv"
BASE_IMAGE_PATH = "/kaggle/input/disaster/waste_dataset"
```

**Verify dataset exists:**

```python
import os

# Check if dataset is available
if os.path.exists('/kaggle/input/disaster/waste_dataset'):
    print("✓ Dataset found!")
    
    # Count files
    files = os.listdir('/kaggle/input/disaster/waste_dataset')
    print(f"  Files/folders: {len(files)}")
    print(f"  Contents: {files[:10]}")  # First 10 items
    
    # Check CSV
    if os.path.exists('/kaggle/input/disaster/waste_dataset/image.csv'):
        print("✓ CSV found!")
    else:
        print("❌ CSV not found!")
else:
    print("❌ Dataset not found!")
    print("   Make sure you added the dataset in Settings > Input")
```

### **Step 5: Kaggle-Specific Optimizations**

Update your `config.py` for Kaggle:

```python
# Batch size for Kaggle T4 (15GB)
if DEVICE == "cuda":
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 14:  # T4 has ~15GB
        BATCH_SIZE = 16  # Safe for ViT-B/16
        print(f"✓ Kaggle T4 detected - Using batch size: {BATCH_SIZE}")
    # ... rest of your logic

# Workers for Kaggle
NUM_WORKERS = 2  # Kaggle works best with 2 workers
PIN_MEMORY = True  # Always True for CUDA
PERSISTENT_WORKERS = True
```

### **Step 6: Monitor GPU Usage**

Run this **during training** to check GPU utilization:

```python
# In a separate cell (while training is running)
!nvidia-smi

# Or continuous monitoring
!watch -n 1 nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   45C    P0    28W /  70W |   3500MiB / 15360MiB |     85%      Default |
+-------------------------------+----------------------+----------------------+

GPU Util should be 70-95% during training!
Memory should be 3-8GB used
```

### **Step 7: Complete Kaggle Notebook Setup**

```python
# Cell 1: Install packages (if needed)
!pip install -q torch torchvision pillow pandas scikit-learn tqdm

# Cell 2: Verify GPU
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Cell 3: Clone your code (if not uploaded)
!git clone https://github.com/adnanul-islam-jisun/Weight_mannagemner.git
%cd Weight_mannagemner

# Cell 4: Verify dataset
import os
print("Dataset exists:", os.path.exists('/kaggle/input/disaster/waste_dataset'))
print("CSV exists:", os.path.exists('/kaggle/input/disaster/waste_dataset/image.csv'))

# Cell 5: Run training
!python train.py
```

---

## 🔧 Troubleshooting Checklist

- [ ] **GPU Accelerator enabled** in Settings → Accelerator → GPU T4 x2
- [ ] **torch.cuda.is_available()** returns True
- [ ] **Dataset added** in Settings → Input → Add Dataset
- [ ] **Paths correct** (/kaggle/input/...)
- [ ] **DEVICE = "cuda"** in config.py
- [ ] **Model moved to GPU** (model.to(device))
- [ ] **Data moved to GPU** (batch['image'].to(device))

---

## 💡 Why GPU Might Not Be Used

### **1. Accelerator Not Enabled**
```
❌ Settings > Accelerator = "None" or "TPU"
✅ Settings > Accelerator = "GPU T4 x2"
```

### **2. Wrong Device**
```python
# Check this
from config.config import DEVICE
print(DEVICE)  # Should be "cuda", not "cpu"
```

### **3. Data Loading Bottleneck**
```python
# GPU is ready but waiting for data
# Increase NUM_WORKERS
NUM_WORKERS = 2  # For Kaggle
```

### **4. Model Not on GPU**
```python
# Make sure model is on GPU
device = torch.device('cuda')
model = model.to(device)

# Check
print(next(model.parameters()).device)  # Should show "cuda:0"
```

---

## 🚀 Expected Performance on Kaggle T4

| Metric | Value |
|--------|-------|
| GPU | Tesla T4 (15GB) |
| Batch Size | 16-24 (for ViT-B/16) |
| GPU Utilization | 80-95% |
| Training Time | ~30-60 min for 50 epochs |
| Memory Used | ~6-8 GB |

---

## 📊 Quick GPU Test

Run this to verify GPU is working:

```python
import torch
import time

# Create tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test computation
x = torch.randn(10000, 10000).to(device)
y = torch.randn(10000, 10000).to(device)

start = time.time()
z = torch.matmul(x, y)
elapsed = time.time() - start

print(f"Matrix multiply time: {elapsed:.3f}s")

if device.type == 'cuda':
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("✅ GPU is working!")
else:
    print("❌ GPU not available!")
```

**Expected:** ~0.1-0.3 seconds on GPU, 5-10 seconds on CPU

---

**If GPU still not detected after all this, restart your Kaggle notebook and re-enable GPU accelerator!**
