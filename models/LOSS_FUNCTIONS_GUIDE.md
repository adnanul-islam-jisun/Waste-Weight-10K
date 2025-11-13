# Loss Functions Guide for Weight Prediction

## 📊 Overview

The multimodal weight predictor now supports **8 different loss functions**, each optimized for different scenarios. This guide helps you choose the best one for your use case.

## 🎯 Quick Recommendation

**For most weight prediction tasks: Use `huber` loss** ✅

```python
trainer = MultimodalTrainer(
    model=model,
    loss_fn='huber',  # Best default choice
    huber_delta=1.0
)
```

## 📚 Available Loss Functions

### 1. **Huber Loss** (RECOMMENDED) ⭐

```python
loss_fn='huber', huber_delta=1.0
```

**Formula**: 
- If |error| ≤ δ: L = 0.5 × error²
- If |error| > δ: L = δ × (|error| - 0.5δ)

**Pros**:
- ✅ **Best of both worlds**: MSE for small errors, MAE for large errors
- ✅ Robust to outliers
- ✅ Smooth gradients
- ✅ Works well across different weight ranges

**Cons**:
- ❌ One extra hyperparameter (delta)

**Use When**:
- You have outliers in your dataset
- Weight measurements have occasional large errors
- You want a robust, general-purpose loss

**Tuning `delta`**:
- `delta=1.0`: Good default for normalized weights
- `delta=5.0`: For weights in kg (1-100 range)
- `delta=0.1`: For normalized weights (0-1 range)

---

### 2. **Mean Squared Error (MSE / L2)**

```python
loss_fn='mse'
```

**Formula**: L = mean((prediction - target)²)

**Pros**:
- ✅ Standard regression loss
- ✅ Penalizes large errors heavily
- ✅ Smooth gradients
- ✅ Works well when errors are normally distributed

**Cons**:
- ❌ Very sensitive to outliers
- ❌ Can be dominated by a few large errors

**Use When**:
- Your data has no outliers
- Errors are normally distributed
- You want to heavily penalize large mistakes

**Typical Results**:
- Loss value: 1.0 - 100.0 (squared units)
- Predictions tend to be close to mean

---

### 3. **Mean Absolute Error (MAE / L1)**

```python
loss_fn='mae'  # or loss_fn='l1'
```

**Formula**: L = mean(|prediction - target|)

**Pros**:
- ✅ **Very robust to outliers**
- ✅ Easy to interpret (same units as weight)
- ✅ Treats all errors equally

**Cons**:
- ❌ Not differentiable at zero (can cause optimization issues)
- ❌ Doesn't heavily penalize large errors

**Use When**:
- Your dataset has many outliers
- You care about median prediction quality
- You want loss in same units as weight (kg)

**Typical Results**:
- Loss value: 0.5 - 10.0 kg
- More robust predictions

---

### 4. **Smooth L1 Loss**

```python
loss_fn='smooth_l1'
```

**Formula**: Similar to Huber with delta=1.0

**Pros**:
- ✅ Combines MSE and MAE benefits
- ✅ PyTorch's original implementation
- ✅ No hyperparameter to tune

**Cons**:
- ❌ Less flexible than Huber (fixed delta)

**Use When**:
- You want Huber-like behavior without tuning delta
- Quick prototyping

---

### 5. **Mean Absolute Percentage Error (MAPE)**

```python
loss_fn='mape'
```

**Formula**: L = mean(|prediction - target| / target) × 100

**Pros**:
- ✅ **Scale-independent** (works for any weight range)
- ✅ Easy to interpret (percentage error)
- ✅ Good for comparing different products

**Cons**:
- ❌ Undefined when target = 0
- ❌ Asymmetric (penalizes under-prediction more)
- ❌ Can be unstable for very small weights

**Use When**:
- Weights span large ranges (1kg to 1000kg)
- You care about relative error, not absolute
- You want to report percentage accuracy

**Typical Results**:
- Loss value: 5.0% - 20.0%
- Better for products with varying sizes

---

### 6. **Mean Squared Log Error (MSLE)**

```python
loss_fn='msle'
```

**Formula**: L = mean((log(1 + prediction) - log(1 + target))²)

**Pros**:
- ✅ **Excellent for wide weight ranges**
- ✅ Scale-invariant
- ✅ Penalizes under-prediction more than over-prediction
- ✅ Naturally handles exponential distributions

**Cons**:
- ❌ Only works for positive weights
- ❌ Less intuitive to interpret

**Use When**:
- Weights range from 0.1kg to 1000kg
- Your weight distribution is log-normal
- You care more about relative errors for small weights

**Typical Results**:
- Loss value: 0.01 - 1.0
- Better predictions for small weights

---

### 7. **Quantile Loss**

```python
loss_fn='quantile', quantile_alpha=0.5  # 0.5 = median
```

**Formula**: 
- If error > 0: L = α × error
- If error < 0: L = (1-α) × error

**Pros**:
- ✅ **Uncertainty estimation**
- ✅ Asymmetric penalties
- ✅ Can predict confidence intervals

**Cons**:
- ❌ Requires understanding of quantiles
- ❌ More complex interpretation

**Use When**:
- You need prediction intervals
- You want to be conservative (α > 0.5) or optimistic (α < 0.5)
- Asymmetric costs (e.g., over-predicting is worse)

**Tuning `alpha`**:
- `alpha=0.5`: Median prediction (symmetric)
- `alpha=0.9`: Conservative (90th percentile)
- `alpha=0.1`: Optimistic (10th percentile)

---

### 8. **Combined Loss (MSE + MAE)**

```python
loss_fn='combined'
```

**Formula**: L = 0.7 × MSE + 0.3 × MAE

**Pros**:
- ✅ Benefits of both MSE and MAE
- ✅ Balanced approach
- ✅ More robust than pure MSE

**Cons**:
- ❌ Fixed weights (not tunable by default)

**Use When**:
- You want a balanced loss
- Uncertain which loss to use

---

## 🎓 Comparison Table

| Loss Function | Outlier Robustness | Interpretability | Best For | Typical Loss Value |
|---------------|-------------------|------------------|----------|-------------------|
| **Huber** ⭐ | ●●●●○ | ●●●○○ | General use | 0.5 - 10.0 |
| **MSE** | ●○○○○ | ●●●○○ | Clean data | 1.0 - 100.0 |
| **MAE** | ●●●●● | ●●●●● | Outlier-heavy | 0.5 - 10.0 kg |
| **Smooth L1** | ●●●●○ | ●●●○○ | Quick prototype | 0.5 - 10.0 |
| **MAPE** | ●●●○○ | ●●●●● | Wide ranges | 5% - 20% |
| **MSLE** | ●●●●○ | ●●○○○ | Log-normal data | 0.01 - 1.0 |
| **Quantile** | ●●●●○ | ●●○○○ | Uncertainty | Varies |
| **Combined** | ●●●○○ | ●●●○○ | Balanced | 0.5 - 50.0 |

## 🔬 Experimental Comparison

### Dataset Characteristics Decision Tree

```
1. Do you have outliers in your weight measurements?
   ├── YES → Use Huber or MAE
   └── NO → Continue to 2

2. Do weights span a wide range (e.g., 0.1kg to 1000kg)?
   ├── YES → Use MSLE or MAPE
   └── NO → Continue to 3

3. Are errors normally distributed?
   ├── YES → Use MSE
   └── NO → Use Huber (safest choice)

4. Do you need uncertainty estimates?
   └── YES → Use Quantile with α=0.1, 0.5, 0.9 (train 3 models)
```

## 💡 Usage Examples

### Example 1: Standard Training (Huber Loss)

```python
from models import create_multimodal_model, MultimodalTrainer

model = create_multimodal_model(...)

trainer = MultimodalTrainer(
    model=model,
    device='cuda',
    learning_rate=1e-4,
    loss_fn='huber',
    huber_delta=1.0  # Tune based on your weight scale
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = trainer.train_step(batch)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Example 2: Wide Weight Range (MSLE)

```python
# For products ranging from 100g to 500kg
trainer = MultimodalTrainer(
    model=model,
    loss_fn='msle',  # Better for log-normal distributions
    learning_rate=1e-4
)
```

### Example 3: Outlier-Heavy Data (MAE)

```python
# When you have measurement errors/outliers
trainer = MultimodalTrainer(
    model=model,
    loss_fn='mae',  # Most robust to outliers
    learning_rate=1e-4
)
```

### Example 4: Uncertainty Estimation (Quantile)

```python
# Train 3 models for 10th, 50th, 90th percentiles
models = []
for alpha in [0.1, 0.5, 0.9]:
    model = create_multimodal_model(...)
    trainer = MultimodalTrainer(
        model=model,
        loss_fn='quantile',
        quantile_alpha=alpha
    )
    # Train...
    models.append(model)

# Predict with confidence intervals
lower = models[0].predict(x)  # 10th percentile
median = models[1].predict(x)  # 50th percentile (median)
upper = models[2].predict(x)   # 90th percentile

print(f"Prediction: {median:.2f}kg (90% CI: {lower:.2f} - {upper:.2f})")
```

### Example 5: Percentage-Based Evaluation (MAPE)

```python
# When you care about relative accuracy
trainer = MultimodalTrainer(
    model=model,
    loss_fn='mape',  # Reports percentage error
    learning_rate=1e-4
)

# Loss will be in percentage (e.g., 8.5 = 8.5% error)
```

## 🎯 Recommended Workflow

### Phase 1: Baseline (Huber Loss)
```python
# Start with Huber - good for most cases
trainer = MultimodalTrainer(model, loss_fn='huber', huber_delta=1.0)
train(epochs=20)
baseline_mae = evaluate()  # Record MAE as baseline
```

### Phase 2: Experiment
```python
# Try different losses
for loss_fn in ['mse', 'mae', 'msle', 'smooth_l1']:
    trainer = MultimodalTrainer(model, loss_fn=loss_fn)
    train(epochs=20)
    mae = evaluate()
    print(f"{loss_fn}: MAE = {mae:.4f}")
```

### Phase 3: Fine-tune
```python
# Once you find the best loss, tune its hyperparameters
# For Huber: try different delta values
for delta in [0.5, 1.0, 2.0, 5.0]:
    trainer = MultimodalTrainer(model, loss_fn='huber', huber_delta=delta)
    train(epochs=20)
    mae = evaluate()
    print(f"Delta {delta}: MAE = {mae:.4f}")
```

## 📊 Performance Tips

### 1. **Loss Scaling**
```python
# If losses are very large or small, normalize your weights first
scaler = StandardScaler()
weights_normalized = scaler.fit_transform(weights.reshape(-1, 1))

# Train on normalized weights
trainer = MultimodalTrainer(model, loss_fn='huber')

# After prediction, denormalize
predictions_original = scaler.inverse_transform(predictions)
```

### 2. **Adaptive Delta for Huber**
```python
# Set delta based on your data statistics
import numpy as np

weight_std = np.std(train_weights)
optimal_delta = weight_std * 0.5  # Rule of thumb

trainer = MultimodalTrainer(
    model,
    loss_fn='huber',
    huber_delta=optimal_delta
)
```

### 3. **Monitor Multiple Metrics**
```python
# Even if training with MSE, evaluate with MAE and MAPE
def evaluate(model, test_loader):
    mse_loss = 0
    mae_loss = 0
    mape_loss = 0
    
    for batch in test_loader:
        preds = model(...)
        targets = batch['weight']
        
        mse_loss += ((preds - targets) ** 2).mean().item()
        mae_loss += (preds - targets).abs().mean().item()
        mape_loss += ((preds - targets).abs() / targets).mean().item() * 100
    
    print(f"MSE: {mse_loss:.4f}")
    print(f"MAE: {mae_loss:.4f} kg")
    print(f"MAPE: {mape_loss:.2f}%")
```

## 🚨 Common Pitfalls

### ❌ Don't Use MAPE if You Have Zero Weights
```python
# MAPE divides by target - fails when target = 0
# Solution: Add small epsilon or use MAE instead
```

### ❌ Don't Use MSLE for Negative Weights
```python
# MSLE uses log(1 + weight) - only for positive values
# Solution: Ensure all weights are > 0
```

### ❌ Don't Mix Loss Functions During Training
```python
# Bad: Changing loss mid-training
for epoch in range(10):
    trainer.loss_fn = 'mse'
    train()
    
for epoch in range(10, 20):
    trainer.loss_fn = 'mae'  # Don't do this!
    train()

# Good: Stick with one loss for entire training
```

## 📈 Expected Performance

Based on typical weight prediction tasks:

| Dataset | Best Loss | Expected MAE | Expected MAPE |
|---------|-----------|-------------|---------------|
| Small objects (0.1-10 kg) | Huber | 0.2-0.5 kg | 5-10% |
| Medium objects (10-100 kg) | MSLE | 1.0-3.0 kg | 3-8% |
| Large objects (100-1000 kg) | MSLE | 5.0-15.0 kg | 2-5% |
| Mixed sizes | MAPE | Varies | 5-15% |

## 🎓 Further Reading

1. **Huber Loss**: Original paper by Peter J. Huber (1964)
2. **Quantile Regression**: Koenker & Bassett (1978)
3. **Robust Statistics**: "Robust Statistics: Theory and Methods" by Maronna et al.

---

## 🆘 Quick Help

**Q: Which loss should I start with?**  
A: **Huber loss** with delta=1.0. It's robust and works well in most cases.

**Q: My loss is very large (>1000). Is this bad?**  
A: Not necessarily - it depends on the loss function. MSE squares errors, so it can be large. Monitor MAE for interpretability.

**Q: Should I normalize weights before training?**  
A: Yes, recommended! Normalize weights to 0-1 or standardize (mean=0, std=1). Then denormalize predictions.

**Q: Can I create my own custom loss?**  
A: Yes! Add it to the `_get_loss_function` method in `MultimodalTrainer`.

**Q: How do I know if my loss is working?**  
A: Check if validation loss decreases over epochs. Also monitor MAE - it's the most interpretable metric.

---

**Happy Training! 🚀**

For more help, check the main documentation in `ENCODERS_README.md`.
