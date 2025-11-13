# Loss Function Summary - Quick Reference

## 📊 Current Status

**Default Loss Function**: `nn.HuberLoss()` (Changed from MSELoss)

The multimodal weight predictor now supports **8 loss functions** optimized for different scenarios.

## 🎯 Best Loss Function for Weight Prediction

### **Recommended: Huber Loss** ⭐

```python
trainer = MultimodalTrainer(
    model=model,
    loss_fn='huber',
    huber_delta=1.0  # Tune based on your weight scale
)
```

**Why Huber is Best**:
1. ✅ **Robust to outliers** - Uses MAE for large errors
2. ✅ **Smooth optimization** - Uses MSE for small errors
3. ✅ **Best of both worlds** - Combines benefits of L1 and L2 loss
4. ✅ **Flexible** - Tunable delta parameter
5. ✅ **Proven** - Standard in robust regression

## 🔄 Quick Comparison

| Loss | When to Use | Pros | Cons |
|------|------------|------|------|
| **Huber** ⭐ | **Default choice** | Robust + Smooth | Need to tune delta |
| **MSE** | Clean data, no outliers | Standard, smooth gradients | Sensitive to outliers |
| **MAE** | Many outliers | Very robust | Not differentiable at 0 |
| **MAPE** | Wide weight ranges | Scale-independent | Fails at weight=0 |
| **MSLE** | 0.1kg to 1000kg | Handles exponential ranges | Only positive weights |
| **Smooth L1** | Quick prototype | No hyperparameters | Less flexible |
| **Quantile** | Need uncertainty | Confidence intervals | Complex |
| **Combined** | Uncertain | Balanced | Fixed weights |

## 🎓 Decision Tree

```
Start Here
    │
    ├─ Do you have outliers? 
    │   ├─ YES → Use Huber or MAE
    │   └─ NO → Continue
    │
    ├─ Wide weight range (0.1kg - 1000kg)?
    │   ├─ YES → Use MSLE or MAPE
    │   └─ NO → Continue
    │
    ├─ Clean, well-behaved data?
    │   ├─ YES → Use MSE
    │   └─ NO → Use Huber (safest)
    │
    └─ Need prediction intervals?
        └─ YES → Use Quantile
```

## 💻 Usage Examples

### Standard Training (Huber)
```python
trainer = MultimodalTrainer(
    model=model,
    loss_fn='huber',
    huber_delta=1.0
)
```

### Outlier-Heavy Data (MAE)
```python
trainer = MultimodalTrainer(
    model=model,
    loss_fn='mae'
)
```

### Wide Weight Ranges (MSLE)
```python
trainer = MultimodalTrainer(
    model=model,
    loss_fn='msle'
)
```

### Percentage Accuracy (MAPE)
```python
trainer = MultimodalTrainer(
    model=model,
    loss_fn='mape'
)
```

## 🔧 Tuning Delta (Huber Loss)

```python
import numpy as np

# Method 1: Based on weight standard deviation
weight_std = np.std(train_weights)
delta = weight_std * 0.5

# Method 2: Based on weight range
weight_range = np.max(train_weights) - np.min(train_weights)
delta = weight_range * 0.05

# Method 3: Manual tuning (try these values)
for delta in [0.1, 0.5, 1.0, 2.0, 5.0]:
    trainer = MultimodalTrainer(model, loss_fn='huber', huber_delta=delta)
    # Train and evaluate...
```

## 📈 Expected Performance

| Dataset Type | Best Loss | Expected MAE | Expected MAPE |
|--------------|-----------|--------------|---------------|
| Small (0.1-10 kg) | Huber | 0.2-0.5 kg | 5-10% |
| Medium (10-100 kg) | MSLE | 1.0-3.0 kg | 3-8% |
| Large (100-1000 kg) | MSLE | 5-15 kg | 2-5% |
| Mixed sizes | MAPE | Varies | 5-15% |

## 📚 Full Documentation

For comprehensive guide, see: **`models/LOSS_FUNCTIONS_GUIDE.md`**

Covers:
- Detailed mathematical formulas
- Pros/cons of each loss
- Usage examples
- Performance tips
- Common pitfalls
- Experimental comparison

## 🚀 Quick Start

```python
from models import create_multimodal_model, MultimodalTrainer

# 1. Create model
model = create_multimodal_model(
    num_categories=10,
    num_numerical_features=6
)

# 2. Create trainer with best loss
trainer = MultimodalTrainer(
    model=model,
    device='cuda',
    loss_fn='huber',       # Best default
    huber_delta=1.0,
    learning_rate=1e-4
)

# 3. Train
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = trainer.train_step(batch)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## ⚡ Key Takeaways

1. **Start with Huber** - Best default for most cases
2. **Monitor MAE** - Most interpretable metric (in kg)
3. **Tune delta** - Based on your weight distribution
4. **Consider data** - Outliers? Wide range? Choose accordingly
5. **Experiment** - Try 2-3 losses, pick best validation MAE

---

**Default Changed**: MSE → Huber Loss ✅

**Why**: Better robustness to outliers while maintaining smooth gradients.
