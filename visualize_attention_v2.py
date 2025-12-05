#!/usr/bin/env python3
"""
Improved Attention Visualization for Weight Prediction Model
============================================================

Compares different attention visualization methods:
1. Last Layer Attention - Standard CLS token attention
2. Attention Rollout - Information flow through all layers
3. GradCAM - Gradient-based (what influences output)
4. Individual Attention Heads - Different heads focus on different patterns
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config.config import *
from features.feature_engineering import engineer_features
from config.training_config import create_optimized_model, WeightPreprocessor
from Dataload.data_preprocessing import prepare_data


def get_vit_attention(model, image):
    """
    Extract attention weights from ViT model.
    Returns attention from each layer.
    """
    model.eval()
    
    # Get the ViT model
    vit = model.image_encoder.vit_model
    
    all_attentions = []
    
    def make_hook(storage):
        def hook(module, input, output):
            # For MultiheadAttention or self-attention layers
            x = input[0]  # Query input
            B, N, C = x.shape
            
            # Check if this is MultiheadAttention
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                qkv_weight = module.in_proj_weight
                qkv_bias = module.in_proj_bias if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None else None
                
                qkv = F.linear(x, qkv_weight, qkv_bias)
                num_heads = module.num_heads
                head_dim = C // num_heads
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Compute attention scores
                scale = head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)  # [B, heads, N, N]
                
                storage.append(attn.detach().cpu())
        return hook
    
    # Register hooks on attention layers
    handles = []
    
    # For torchvision ViT: encoder.layers[i].self_attention
    if hasattr(vit, 'encoder') and hasattr(vit.encoder, 'layers'):
        for layer in vit.encoder.layers:
            if hasattr(layer, 'self_attention'):
                h = layer.self_attention.register_forward_hook(make_hook(all_attentions))
                handles.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model.image_encoder(image)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    return all_attentions


def compute_attention_rollout(all_attentions, discard_ratio=0.1):
    """
    Compute attention rollout - multiply attention matrices through layers.
    This captures how information flows from input to output.
    """
    if len(all_attentions) == 0:
        return None
    
    # Average over heads for each layer
    layer_attns = []
    for attn in all_attentions:
        # attn shape: [B, heads, N, N]
        attn_avg = attn.mean(dim=1)  # [B, N, N]
        layer_attns.append(attn_avg)
    
    B, N, _ = layer_attns[0].shape
    device = layer_attns[0].device
    
    # Start with identity matrix
    result = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    
    for attn in layer_attns:
        # Add residual connection
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        attn = 0.5 * attn + 0.5 * eye
        
        # Multiply through
        result = torch.bmm(attn, result)
    
    # Get CLS token attention to patches (skip CLS token itself)
    cls_attn = result[:, 0, 1:]  # [B, num_patches]
    
    # Reshape to spatial
    num_patches = cls_attn.shape[1]
    h = w = int(np.sqrt(num_patches))
    cls_attn = cls_attn.reshape(B, h, w)
    
    # Normalize
    cls_attn = cls_attn - cls_attn.min()
    cls_attn = cls_attn / (cls_attn.max() + 1e-8)
    
    return cls_attn.numpy()


def compute_gradcam(model, image, meta_cat, meta_num, device):
    """
    Compute GradCAM - gradient-weighted activation map.
    Shows which regions influence the output.
    """
    model.eval()
    
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    # Hook the last encoder layer
    vit = model.image_encoder.vit_model
    target_layer = None
    
    if hasattr(vit, 'encoder') and hasattr(vit.encoder, 'layers'):
        target_layer = vit.encoder.layers[-1]
    
    if target_layer is None:
        return None
    
    # Register hooks
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass with gradients
        image_input = image.clone().requires_grad_(True)
        output = model(image_input, meta_cat, meta_num)
        
        # Backward pass
        model.zero_grad()
        output.sum().backward()
        
        if activations is None or gradients is None:
            return None
        
        # Compute GradCAM
        # activations: [B, N, C], gradients: [B, N, C]
        weights = gradients.mean(dim=-1, keepdim=True)  # Global average pooling
        cam = (weights * activations).sum(dim=-1)  # [B, N]
        
        # Remove CLS token
        cam = cam[:, 1:]  # [B, num_patches]
        
        # Reshape to spatial
        B, num_patches = cam.shape
        h = w = int(np.sqrt(num_patches))
        cam = cam.reshape(B, h, w)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    except Exception as e:
        print(f"GradCAM error: {e}")
        return None
    
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def create_attention_comparison(
    image_np, 
    last_layer_attn,
    rollout_attn, 
    gradcam_attn,
    head_attns,
    true_weight, 
    pred_weight,
    save_path
):
    """Create comparison visualization of different attention methods."""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    def overlay_attention(ax, img, attn, title):
        ax.imshow(img)
        if attn is not None:
            attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]))
            ax.imshow(attn_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    # Row 1: Main comparison
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    overlay_attention(axes[0, 1], image_np, last_layer_attn, 
                     'Last Layer Attention\n(CLS token → patches)')
    
    overlay_attention(axes[0, 2], image_np, rollout_attn,
                     'Attention Rollout\n(All layers combined)')
    
    overlay_attention(axes[0, 3], image_np, gradcam_attn,
                     'GradCAM\n(Gradient-based)')
    
    # Row 2: Individual attention heads (first 4)
    if head_attns is not None and len(head_attns) >= 4:
        for i in range(4):
            overlay_attention(axes[1, i], image_np, head_attns[i],
                            f'Attention Head {i+1}')
    else:
        for i in range(4):
            axes[1, i].text(0.5, 0.5, 'Not Available', ha='center', va='center')
            axes[1, i].axis('off')
    
    # Title with prediction info
    error = pred_weight - true_weight
    error_pct = abs(error) / true_weight * 100 if true_weight > 0 else 0
    fig.suptitle(
        f'Attention Comparison | True: {true_weight:.0f}kg | Pred: {pred_weight:.0f}kg | '
        f'Error: {error:+.0f}kg ({error_pct:.1f}%)',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_samples(model, dataloader, device, output_dir, num_samples=8):
    """Generate attention visualizations for samples."""
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    samples_done = 0
    
    print(f"\n  Processing samples...")
    
    for batch in dataloader:
        images = batch['image'].to(device)
        meta_cat = batch['category_idx'].to(device)
        meta_num = batch['numerical'].to(device)
        weights = batch['weight'].cpu().numpy()
        
        # Get predictions
        with torch.no_grad():
            predictions = model(images, meta_cat, meta_num)
            if USE_LOG_TRANSFORM:
                predictions = torch.expm1(predictions)
            predictions = predictions.cpu().numpy().flatten()
        
        for i in range(len(images)):
            if samples_done >= num_samples:
                break
            
            image = images[i:i+1]
            cat = meta_cat[i:i+1]
            num = meta_num[i:i+1]
            true_w = weights[i]
            pred_w = predictions[i]
            
            # Get original image for display
            img_np = image[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # 1. Get attention from all layers
            all_attns = get_vit_attention(model, image)
            
            # 2. Last layer attention (average over heads)
            if len(all_attns) > 0:
                last_attn = all_attns[-1]  # [B, heads, N, N]
                # CLS attention to patches, average over heads
                cls_attn = last_attn[0, :, 0, 1:].mean(dim=0)  # [num_patches]
                h = w = int(np.sqrt(cls_attn.shape[0]))
                last_layer_attn = cls_attn.reshape(h, w).numpy()
                # Normalize
                last_layer_attn = (last_layer_attn - last_layer_attn.min()) / (last_layer_attn.max() - last_layer_attn.min() + 1e-8)
                
                # Individual heads
                head_attns = []
                num_heads = last_attn.shape[1]
                for hi in range(min(num_heads, 4)):
                    ha = last_attn[0, hi, 0, 1:].numpy()
                    ha = ha.reshape(h, w)
                    ha = (ha - ha.min()) / (ha.max() - ha.min() + 1e-8)
                    head_attns.append(ha)
            else:
                last_layer_attn = None
                head_attns = None
            
            # 3. Attention rollout
            rollout_attn = compute_attention_rollout(all_attns)
            if rollout_attn is not None:
                rollout_attn = rollout_attn[0]
            
            # 4. GradCAM
            gradcam_attn = compute_gradcam(model, image, cat, num, device)
            if gradcam_attn is not None:
                gradcam_attn = gradcam_attn[0]
            
            # Create visualization
            save_path = os.path.join(output_dir, f'attention_sample_{samples_done:03d}.png')
            create_attention_comparison(
                img_np, last_layer_attn, rollout_attn, gradcam_attn,
                head_attns, true_w, pred_w, save_path
            )
            print(f"    ✓ Sample {samples_done+1}: True={true_w:.0f}kg, Pred={pred_w:.0f}kg")
            
            samples_done += 1
        
        if samples_done >= num_samples:
            break
    
    return samples_done


def create_grid_visualization(model, dataloader, device, output_dir, num_samples=16):
    """Create a single grid showing multiple samples with attention."""
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect samples
    samples = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        meta_cat = batch['category_idx'].to(device)
        meta_num = batch['numerical'].to(device)
        weights = batch['weight'].cpu().numpy()
        
        with torch.no_grad():
            predictions = model(images, meta_cat, meta_num)
            if USE_LOG_TRANSFORM:
                predictions = torch.expm1(predictions)
            predictions = predictions.cpu().numpy().flatten()
        
        for i in range(len(images)):
            if len(samples) >= num_samples:
                break
            
            image = images[i:i+1]
            
            # Get attention
            all_attns = get_vit_attention(model, image)
            rollout = compute_attention_rollout(all_attns)
            
            img_np = image[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            samples.append({
                'image': img_np,
                'attention': rollout[0] if rollout is not None else None,
                'true': weights[i],
                'pred': predictions[i]
            })
        
        if len(samples) >= num_samples:
            break
    
    # Create grid
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, sample in enumerate(samples):
        ax = axes[idx]
        ax.imshow(sample['image'])
        
        if sample['attention'] is not None:
            attn_resized = cv2.resize(sample['attention'], 
                                     (sample['image'].shape[1], sample['image'].shape[0]))
            ax.imshow(attn_resized, cmap='jet', alpha=0.5)
        
        error = sample['pred'] - sample['true']
        ax.set_title(f"True: {sample['true']:.0f}kg | Pred: {sample['pred']:.0f}kg\n"
                    f"Error: {error:+.0f}kg", fontsize=9)
        ax.axis('off')
    
    # Hide empty axes
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Attention Rollout Visualization (Red/Yellow = High Attention)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'attention_grid_improved.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  ✓ Grid saved: {save_path}")
    return len(samples)


def create_readme(output_dir):
    """Create README explaining the visualizations."""
    
    readme = """# Attention Visualization Analysis

## Methods Compared

### 1. Last Layer Attention
- Attention weights from the final transformer layer
- Shows what CLS token "attends to" at the end
- **Issue**: Often diffuse because late layers capture semantic concepts globally

### 2. Attention Rollout  
- Multiplies attention matrices through ALL 12 layers
- Better captures how information flows from input to output
- Accounts for residual connections
- **Best for**: Understanding complete information flow

### 3. GradCAM (Gradient-weighted Class Activation Mapping)
- Uses gradients to find which regions influence the weight prediction
- **Best for**: Understanding what actually affects the model's decision

### 4. Individual Attention Heads
- 12 different attention heads learn different patterns:
  - Some focus on edges/boundaries
  - Some focus on textures  
  - Some focus on global context
- Averaging heads can blur important patterns

## Why Attention Looks Diffuse

1. **ImageNet Pretraining**: ViT learned attention for classification (what is it?), 
   not localization (where is it?)

2. **No Object Segmentation**: Model doesn't explicitly know object vs background

3. **Global Context for Weight**: Weight depends on entire object, not just edges

4. **Background Information**: Scale cues (floor, walls) might actually be useful!

## Recommendations for Improvement

1. **Add Object Detection**: Use YOLO/SAM to mask objects first
2. **Attention Loss**: Add auxiliary loss to focus attention on objects
3. **Longer Fine-tuning**: More epochs with lower LR helps attention
4. **Different Architecture**: ResNet + GradCAM often gives cleaner attention maps
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description='Improved Attention Visualization')
    parser.add_argument('--num-samples', type=int, default=8, help='Samples for detailed view')
    parser.add_argument('--grid-samples', type=int, default=16, help='Samples for grid view')
    parser.add_argument('--output-dir', type=str, default='attention_analysis', help='Output directory')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/best_model_phase1_20251203_071600.pt')
    args = parser.parse_args()
    
    print("=" * 80)
    print("IMPROVED ATTENTION VISUALIZATION")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    cols_to_convert = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'weight_in_kg']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['weight_in_kg'], inplace=True)
    df.fillna(0.0, inplace=True)
    df = df[df['weight_in_kg'] >= 50]
    
    df_featured = engineer_features(df)
    if 'weight_in_kg' in df_featured.columns and 'weight' not in df_featured.columns:
        df_featured.rename(columns={'weight_in_kg': 'weight'}, inplace=True)
    
    data_dict = prepare_data(df_featured, BASE_IMAGE_PATH)
    
    # Create model
    print("\n📦 Creating model...")
    model, preprocessor, _ = create_optimized_model(
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(data_dict['numerical_features']),
        scaler=data_dict['numerical_scaler'],
        device=DEVICE
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = Path(CHECKPOINT_DIR) / "latest_checkpoint.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    # 1. Detailed comparisons
    print(f"\n  📊 Detailed comparison ({args.num_samples} samples)...")
    visualize_samples(model, data_dict['test_loader'], device, 
                     args.output_dir, args.num_samples)
    
    # 2. Grid view
    print(f"\n  📊 Grid visualization ({args.grid_samples} samples)...")
    create_grid_visualization(model, data_dict['test_loader'], device,
                             args.output_dir, args.grid_samples)
    
    # 3. README
    create_readme(args.output_dir)
    
    print("\n" + "=" * 80)
    print("🎉 VISUALIZATION COMPLETE!")
    print(f"📁 Output: {args.output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
