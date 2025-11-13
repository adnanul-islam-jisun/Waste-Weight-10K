# Google Colab Setup - Fast Data Loading
# Run this ONCE at the beginning of your Colab session

## Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2: Clone your repository
```python
!git clone https://github.com/adnanul-islam-jisun/Weight_mannagemner.git
%cd Weight_mannagemner
```

## Step 3: Install requirements
```python
!pip install -q torch torchvision torchaudio
!pip install -q pillow pandas numpy scikit-learn tqdm
```

## Step 4: Copy dataset to local SSD (IMPORTANT!)
```python
# This copies data from Google Drive to Colab's local SSD
# Makes training 10-20x faster!

!python setup_colab.py
```

**Or manually copy:**
```python
import shutil
import os

# Copy dataset from Drive to local storage
print("Copying dataset to local SSD...")
!cp -r /content/drive/MyDrive/KaggleData/disaster/waste_dataset /content/dataset
print("✓ Copy complete!")

# Verify
if os.path.exists('/content/dataset/image.csv'):
    print("✓ Dataset ready at /content/dataset/")
else:
    print("❌ Copy failed!")
```

## Step 5: Update config.py paths
```python
# Update paths in config.py to use local storage
config_file = '/content/Weight_mannagemner/config/config.py'

# Read config
with open(config_file, 'r') as f:
    config = f.read()

# Update paths to local storage
config = config.replace(
    'CSV_PATH = "/content/drive/MyDrive/KaggleData/disaster/waste_dataset/image.csv"',
    'CSV_PATH = "/content/dataset/image.csv"'
)
config = config.replace(
    'BASE_IMAGE_PATH = "/content/drive/MyDrive/KaggleData/disaster/waste_dataset"',
    'BASE_IMAGE_PATH = "/content/dataset"'
)

# Write back
with open(config_file, 'w') as f:
    f.write(config)

print("✓ Config updated to use local SSD!")
```

## Step 6: Run training
```python
%cd /content/Weight_mannagemner
!python train.py
```

---

## Performance Comparison

### Before (Google Drive):
```
Data loading: SLOW (~5-10 MB/s)
GPU utilization: 20-30% (waiting for data)
Training time: ~4-6 hours for 50 epochs
```

### After (Local SSD):
```
Data loading: FAST (~500-1000 MB/s)
GPU utilization: 80-95% (fully utilized!)
Training time: ~20-40 minutes for 50 epochs ⚡
```

**Speedup: 10-20x faster!** 🚀

---

## Tips

1. **Run setup_colab.py once** at the start of each session
2. **Data persists** only during the session (12 hours max)
3. **Models save to checkpoints/** - download after training
4. **Monitor GPU usage**: Click Runtime > Manage Sessions
5. **Free tier limits**: T4 GPU, 12-hour sessions

---

## Troubleshooting

### "Not enough space"
```python
# Check available space
!df -h /content

# Clear space
!rm -rf /content/sample_data
!rm -rf /content/dataset  # If re-running
```

### "Dataset not found"
```python
# Check if Drive is mounted
!ls /content/drive/MyDrive/KaggleData/disaster/waste_dataset/

# Re-mount if needed
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### "OOM error during training"
```python
# Reduce batch size in config.py
# For T4 GPU (15GB): Use batch_size = 8-16
# For P100 (16GB): Use batch_size = 16-24
# For V100 (16GB): Use batch_size = 24-32
```
