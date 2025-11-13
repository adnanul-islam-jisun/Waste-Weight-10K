# Image Encoder & Metadata Encoder Modules

## Overview

This directory contains advanced encoder modules for multimodal weight prediction:

1. **Image Encoder** (`image_encoder.py`) - Vision Transformer-based visual feature extractor
2. **Metadata Encoder** (`metadata_encoder.py`) - Heterogeneous data processor for categorical and numerical features
3. **Multimodal Fusion** (`multimodal_fusion.py`) - Combined model that fuses image and metadata features

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Module Details](#module-details)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Training Guide](#training-guide)

## 🏗️ Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │       MULTIMODAL WEIGHT PREDICTOR       │
                    └─────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
          ┌─────────▼─────────┐           ┌────────────▼────────────┐
          │   IMAGE ENCODER   │           │   METADATA ENCODER      │
          │  (ViT-Base/16)    │           │  (Embedding + MLP)      │
          └─────────┬─────────┘           └────────────┬────────────┘
                    │                                  │
          ┌─────────▼─────────┐           ┌────────────▼────────────┐
          │   Product Image   │           │    Product Metadata     │
          │   (224x224 RGB)   │           │  • Category (Type)      │
          └───────────────────┘           │  • Volume (V_x,V_y,V_z) │
                                          │  • Distance (D_x,D_y)    │
                                          │  • View Angle            │
                                          └─────────────────────────┘
```

### Data Flow

1. **Image Path**: RGB Image (224×224) → **Image Encoder** → Visual Features (768-dim)
2. **Metadata Path**: Category + Numerical → **Metadata Encoder** → Metadata Features (256-dim)
3. **Fusion**: Concatenate → Fusion Network → Regression Head → Weight Prediction

## 📦 Module Details

### 1. Image Encoder (`image_encoder.py`)

**Objective**: Extract rich, high-dimensional visual representations from product images.

**Architecture**:
- **Backbone**: Vision Transformer ONLY (No CNN fallback)
- **Supported Models**:
  - `vit_b_16`: ViT-Base/16 (768-dim, 86M params) - **DEFAULT**
  - `vit_b_32`: ViT-Base/32 (768-dim, 88M params) - Faster
  - `vit_l_16`: ViT-Large/16 (1024-dim, 304M params) - Higher capacity
  - `vit_l_32`: ViT-Large/32 (1024-dim, 306M params) - Large + fast
  - `vit_h_14`: ViT-Huge/14 (1280-dim, 632M params) - Maximum capacity
- **Pre-training**: ImageNet-1K (SWAG for Huge variant)
- **Output**: Configurable visual feature vector

**Key Features**:
- Multiple ViT variants for different use cases
- Pre-trained weights for transfer learning
- Optional backbone freezing for fine-tuning control
- Configurable output dimensions
- GELU activation (transformer-native)
- Dropout regularization
- Native dimension support (no projection needed)

**Input**: `(batch_size, 3, 224, 224)` - RGB images
**Output**: `(batch_size, output_dim)` - Visual feature vectors

---

### 2. Metadata Encoder (`metadata_encoder.py`)

**Objective**: Convert mixed-type metadata into unified dense feature vectors.

**Architecture**:
- **Categorical Branch**: Embedding layer for product categories
- **Numerical Branch**: MLP for volume, distance, and angle features
- **Fusion**: Concatenation + dense layers

**Key Features**:
- Trainable category embeddings
- Automatic feature normalization with StandardScaler
- Parallel processing of heterogeneous data
- Batch normalization and dropout

**Input**:
- Category indices: `(batch_size,)`
- Numerical features: `(batch_size, num_features)`

**Output**: `(batch_size, 256)` - Unified metadata features

---

### 3. Multimodal Fusion (`multimodal_fusion.py`)

**Objective**: Combine visual and metadata features for accurate weight prediction.

**Architecture**:
- Late fusion strategy
- Multi-layer fusion network
- Optional residual connections
- Regression head for weight prediction

**Key Features**:
- Flexible fusion layer configuration
- Separate learning rates for different components
- Feature extraction for analysis
- Freeze/unfreeze capabilities

**Input**:
- Images, category indices, numerical features

**Output**: `(batch_size, 1)` - Predicted weights

## 🚀 Quick Start

### Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision scikit-learn pandas numpy pillow

# Verify torchvision version (must be >= 0.13.0 for ViT support)
python -c "import torchvision; print(f'torchvision version: {torchvision.__version__}')"
```

### Basic Usage

```python
from models import (
    create_default_image_encoder,
    create_large_image_encoder,
    create_huge_image_encoder,
    create_fast_image_encoder,
    create_metadata_encoder_from_data,
    create_multimodal_model
)

# 1. Create encoders - Choose based on your needs

# Option A: Default (recommended for most cases)
image_encoder = create_default_image_encoder(output_dim=768)

# Option B: Large model (for complex images & large datasets)
# image_encoder = create_large_image_encoder(output_dim=1024)

# Option C: Huge model (maximum capacity, requires 100K+ images)
# image_encoder = create_huge_image_encoder(output_dim=1280, freeze_backbone=True)

# Option D: Fast model (optimized for speed)
# image_encoder = create_fast_image_encoder(output_dim=768)

# 2. Create metadata encoder
metadata_encoder, scaler, category_map = create_metadata_encoder_from_data(
    dataframe=df,
    category_column='Type',
    numerical_columns=['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'view_angle_rad']
)

# 3. Create complete model
model = create_multimodal_model(
    num_categories=len(category_map),
    num_numerical_features=6,
    scaler=scaler
)

# 4. Make predictions
predictions = model(images, category_indices, numerical_features)
```

## 🎯 Choosing the Right Vision Transformer

### Model Comparison

| Model | Parameters | Output Dim | Speed | Memory | Best For |
|-------|-----------|-----------|-------|---------|----------|
| **vit_b_16** ⭐ | 86M | 768 | ●●●○○ | ●●○○○ | **Default choice - balanced** |
| **vit_b_32** | 88M | 768 | ●●●●○ | ●●○○○ | Speed-critical applications |
| **vit_l_16** | 304M | 1024 | ●●○○○ | ●●●●○ | Complex images, large data |
| **vit_l_32** | 306M | 1024 | ●●●○○ | ●●●●○ | Balance of capacity & speed |
| **vit_h_14** ⚠️ | 632M | 1280 | ●○○○○ | ●●●●● | Huge datasets (100K+ images) |

### Decision Tree

```
Dataset Size?
├── < 10K images
│   ├── Need speed? → vit_b_32 (freeze_backbone=True)
│   └── Need quality? → vit_b_16 (freeze_backbone=False)
│
├── 10K - 100K images
│   ├── GPU < 8GB? → vit_b_16 (freeze_backbone=True)
│   └── GPU >= 8GB? → vit_l_16 (freeze_backbone=False)
│
└── > 100K images
    ├── Maximum accuracy? → vit_h_14 (freeze_backbone=True, then fine-tune)
    └── Balance? → vit_l_16 (freeze_backbone=False)
```

### When to Use Each Model

**ViT-Base/16 (vit_b_16)** ⭐ **RECOMMENDED DEFAULT**
```python
encoder = create_default_image_encoder(output_dim=768)
```
- ✅ Most balanced option
- ✅ Works well with 1K-50K images
- ✅ Good speed-accuracy tradeoff
- ✅ Fits on 4GB+ GPU
- Use for: General purpose, prototyping, most production scenarios

**ViT-Base/32 (vit_b_32)** ⚡ **FAST**
```python
encoder = create_fast_image_encoder(output_dim=768)
```
- ✅ 2x faster inference than vit_b_16
- ✅ Lower memory usage
- ⚠️ Slightly lower accuracy
- Use for: Real-time applications, edge deployment, limited compute

**ViT-Large/16 (vit_l_16)** 🔥 **HIGH PERFORMANCE**
```python
encoder = create_large_image_encoder(output_dim=1024)
```
- ✅ Higher capacity (304M params)
- ✅ Better for complex visual patterns
- ⚠️ Requires 8GB+ GPU
- ⚠️ Needs 10K+ images to avoid overfitting
- Use for: Complex products, large datasets, when accuracy is critical

**ViT-Huge/14 (vit_h_14)** 🚀 **MAXIMUM CAPACITY**
```python
encoder = create_huge_image_encoder(output_dim=1280, freeze_backbone=True)
```
- ✅ Maximum capacity (632M params)
- ✅ SWAG pre-training (best transfer learning)
- ⚠️ **WARNING**: Requires 100K+ images when fine-tuning
- ⚠️ Requires 16GB+ GPU
- ⚠️ Slow training and inference
- Use for: Very large datasets, when you need absolute best accuracy

### Advanced: Custom Model Selection

```python
from models.image_encoder import ImageEncoder

# Create any ViT variant directly
encoder = ImageEncoder(
    model_name='vit_l_16',  # Choose: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
    pretrained=True,         # Use ImageNet pre-trained weights
    freeze_backbone=False,   # Allow fine-tuning
    output_dim=512,          # Custom output dimension
    dropout=0.2              # Regularization
)

# Print model info
print(f"Model: {encoder.model_name}")
print(f"Native dimension: {encoder.get_native_dim()}")
print(f"Output dimension: {encoder.get_output_dim()}")
print(f"Total parameters: {encoder.get_num_parameters():,}")
print(f"Trainable parameters: {encoder.get_num_parameters(trainable_only=True):,}")
```

## 📚 Usage Examples

### Example 1: Image Encoder Only

```python
from models.image_encoder import ImageEncoder
import torch

# Create encoder
encoder = ImageEncoder(
    pretrained=True,
    freeze_backbone=False,
    output_dim=768,
    dropout=0.1
)

# Extract features
images = torch.randn(4, 3, 224, 224)  # Batch of 4 images
visual_features = encoder(images)  # (4, 768)
```

### Example 2: Metadata Encoder Only

```python
from models.metadata_encoder import MetadataEncoder
from sklearn.preprocessing import StandardScaler
import torch

# Setup
scaler = StandardScaler()
scaler.fit(numerical_data)

# Create encoder
encoder = MetadataEncoder(
    num_categories=10,
    category_embedding_dim=32,
    num_numerical_features=6,
    output_dim=256,
    scaler=scaler
)

# Extract features
category_idx = torch.tensor([0, 1, 2, 3])
numerical = torch.randn(4, 6)
metadata_features = encoder(category_idx, numerical)  # (4, 256)
```

### Example 3: Complete Training Pipeline

```python
from models.multimodal_fusion import MultimodalTrainer

# Create model (as shown in Quick Start)
model = create_multimodal_model(...)

# Create trainer
trainer = MultimodalTrainer(
    model=model,
    device='cuda',
    learning_rate=1e-4,
    weight_decay=1e-5
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = trainer.train_step(batch)
    
    # Validation
    for batch in val_loader:
        val_loss, preds, targets = trainer.validate_step(batch)
```

### Example 4: Feature Analysis

```python
# Extract intermediate representations
features = model.get_feature_representations(
    images, category_indices, numerical_features
)

print(f"Visual features: {features['visual_features'].shape}")
print(f"Metadata features: {features['metadata_features'].shape}")
print(f"Fused features: {features['fused_features'].shape}")
```

## 🔧 API Reference

### ImageEncoder

**Constructor Parameters**:
- `pretrained` (bool): Use pre-trained weights. Default: `True`
- `freeze_backbone` (bool): Freeze ViT backbone. Default: `False`
- `output_dim` (int): Output feature dimension. Default: `768`
- `dropout` (float): Dropout rate. Default: `0.1`

**Methods**:
- `forward(x)`: Extract visual features from images
- `get_output_dim()`: Return output dimension

---

### MetadataEncoder

**Constructor Parameters**:
- `num_categories` (int): Number of unique categories
- `category_embedding_dim` (int): Category embedding dimension. Default: `32`
- `num_numerical_features` (int): Number of numerical features
- `numerical_hidden_dims` (List[int]): MLP hidden dimensions. Default: `[64, 32]`
- `output_dim` (int): Output feature dimension. Default: `256`
- `dropout` (float): Dropout rate. Default: `0.1`
- `scaler` (StandardScaler): Fitted scaler for numerical features

**Methods**:
- `forward(category_indices, numerical_features)`: Extract metadata features
- `get_output_dim()`: Return output dimension
- `update_scaler(scaler)`: Update normalization scaler

---

### MultimodalWeightPredictor

**Constructor Parameters**:
- `image_encoder` (ImageEncoder): Pre-configured image encoder
- `metadata_encoder` (MetadataEncoder): Pre-configured metadata encoder
- `fusion_hidden_dims` (List[int]): Fusion layer dimensions. Default: `[512, 256, 128]`
- `dropout` (float): Dropout rate. Default: `0.2`
- `use_residual` (bool): Use residual connections. Default: `True`

**Methods**:
- `forward(images, category_indices, numerical_features)`: Predict weights
- `get_feature_representations(...)`: Extract intermediate features
- `freeze_image_encoder()`: Freeze image encoder parameters
- `unfreeze_image_encoder()`: Unfreeze image encoder parameters
- `freeze_metadata_encoder()`: Freeze metadata encoder parameters
- `unfreeze_metadata_encoder()`: Unfreeze metadata encoder parameters

## 🎓 Training Guide

### Best Practices

1. **Progressive Training**:
   ```python
   # Stage 1: Train with frozen image encoder
   model.freeze_image_encoder()
   train(epochs=10)
   
   # Stage 2: Fine-tune entire model
   model.unfreeze_image_encoder()
   train(epochs=20, lr=1e-5)
   ```

2. **Learning Rate Scheduling**:
   ```python
   # Different LR for different components
   optimizer = torch.optim.AdamW([
       {'params': model.image_encoder.parameters(), 'lr': 1e-5},
       {'params': model.metadata_encoder.parameters(), 'lr': 1e-4},
       {'params': model.fusion_network.parameters(), 'lr': 1e-4}
   ])
   ```

3. **Data Augmentation**:
   ```python
   from torchvision import transforms
   
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.ColorJitter(brightness=0.2, contrast=0.2),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
   ])
   ```

4. **Batch Size Recommendations**:
   - GPU (8GB): batch_size = 16-32
   - GPU (16GB): batch_size = 32-64
   - CPU: batch_size = 4-8

### Hyperparameter Tuning

```python
config = {
    'image_output_dim': [512, 768, 1024],
    'metadata_output_dim': [128, 256, 512],
    'category_embedding_dim': [16, 32, 64],
    'fusion_hidden_dims': [
        [512, 256, 128],
        [1024, 512, 256],
        [768, 384, 192]
    ],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}
```

## 🧪 Running the Demo

Run the comprehensive demo to see all modules in action:

```bash
python demo_encoders.py
```

This will demonstrate:
1. Image encoder feature extraction
2. Metadata encoder processing
3. Multimodal fusion pipeline
4. Training step example

## 📊 Expected Performance

With proper training on your dataset, you should expect:

- **Visual Features**: Capture product shape, material texture, and visual properties
- **Metadata Features**: Encode size, distance, and category information
- **Combined Model**: Achieve better accuracy than image-only or metadata-only models

## 🐛 Troubleshooting

**Issue**: `ViT not available in torchvision`
- **Solution**: Update torchvision: `pip install --upgrade torchvision>=0.13.0`
- **Fallback**: Model automatically uses ResNet50

**Issue**: Out of memory errors
- **Solution**: Reduce batch size or freeze image encoder backbone

**Issue**: Poor convergence
- **Solution**: Try progressive training (freeze → unfreeze) or adjust learning rates

## 📝 Citation

If you use these modules in your research, please cite:

```bibtex
@software{weight_predictor_encoders,
  title={Multimodal Encoders for Weight Prediction},
  author={Your Name},
  year={2025},
  description={Vision Transformer and metadata encoders for product weight estimation}
}
```

## 📄 License

MIT License - See LICENSE file for details

---

**Happy Training! 🚀**

For questions or issues, please open an issue in the repository.
