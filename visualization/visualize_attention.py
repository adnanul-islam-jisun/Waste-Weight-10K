"""
Attention Visualization Script
Visualizes where the model focuses when making predictions
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *
from features.feature_engineering import engineer_features
from config.training_config import create_optimized_model, WeightPreprocessor
from dataload.data_preprocessing import prepare_data


def get_attention_maps(model, images, category_indices, numerical_features, device):
    """
    Extract attention weights from the mutual attention fusion block.
    
    Returns:
        - visual_to_metadata_attn: How visual features attend to metadata
        - metadata_to_visual_attn: How metadata attends to visual features
        - vit_attention: Attention from Vision Transformer layers
    """
    model.eval()
    
    with torch.no_grad():
        # Get prediction with attention weights
        prediction, attention_weights = model(
            images, category_indices, numerical_features,
            return_attention=True
        )
    
    return prediction, attention_weights


def get_vit_attention_rollout(model, images, device):
    """
    Extract and aggregate attention from all ViT layers using attention rollout.
    This shows which image patches the ViT focuses on.
    """
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # ViT attention output shape: (batch, num_heads, seq_len, seq_len)
        attention_weights.append(output)
    
    # Register hooks on all attention layers
    hooks = []
    for name, module in model.image_encoder.vit_model.encoder.layers.named_modules():
        if 'self_attention' in name and hasattr(module, 'out_proj'):
            # We need to hook the attention computation
            pass
    
    # Alternative: Use gradient-based attention (Grad-CAM style)
    # This works with any model architecture
    
    images.requires_grad = True
    
    # Forward pass
    visual_features = model.image_encoder(images)
    
    # Use the norm of features as importance
    importance = visual_features.norm(dim=1)
    importance.backward(torch.ones_like(importance))
    
    # Get gradients
    gradients = images.grad
    
    # Average over color channels and take absolute value
    saliency = gradients.abs().mean(dim=1)
    
    return saliency


def visualize_attention_single(model, image_path, category_idx, numerical_features, 
                                weight_preprocessor, device, save_path=None):
    """
    Visualize attention for a single image.
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Prepare inputs
    category_tensor = torch.tensor([category_idx], dtype=torch.long).to(device)
    numerical_tensor = torch.tensor([numerical_features], dtype=torch.float32).to(device)
    
    model.eval()
    
    # ========================================================================
    # Method 1: Gradient-based Saliency Map
    # ========================================================================
    image_tensor.requires_grad = True
    
    # Forward pass
    prediction, attention_weights = model(
        image_tensor, category_tensor, numerical_tensor,
        return_attention=True
    )
    
    # Backward pass to get gradients
    prediction.backward()
    
    # Get saliency map
    gradients = image_tensor.grad.data.abs()
    saliency = gradients.squeeze().cpu()
    
    # Average over color channels
    saliency_gray = saliency.mean(dim=0).numpy()
    
    # Normalize
    saliency_gray = (saliency_gray - saliency_gray.min()) / (saliency_gray.max() - saliency_gray.min() + 1e-8)
    
    # ========================================================================
    # Method 2: Attention Rollout from Cross-Attention
    # ========================================================================
    v_to_m_attn = attention_weights.get('visual_to_metadata', None)
    m_to_v_attn = attention_weights.get('metadata_to_visual', None)
    
    # ========================================================================
    # Get prediction value
    # ========================================================================
    with torch.no_grad():
        pred_value = model(image_tensor, category_tensor, numerical_tensor)
        pred_kg = weight_preprocessor.inverse_transform(pred_value.cpu().numpy().flatten())[0]
    
    # ========================================================================
    # Visualize
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nPredicted: {pred_kg:.1f} kg', fontsize=12)
    axes[0].axis('off')
    
    # Saliency map
    axes[1].imshow(saliency_gray, cmap='hot')
    axes[1].set_title('Gradient Saliency Map\n(Where gradients are strongest)', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    original_resized = original_image.resize((224, 224))
    axes[2].imshow(original_resized)
    heatmap = axes[2].imshow(saliency_gray, cmap='jet', alpha=0.5)
    axes[2].set_title('Attention Overlay\n(Important regions for prediction)', fontsize=12)
    axes[2].axis('off')
    
    plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    plt.close()
    
    return pred_kg, saliency_gray


def visualize_attention_batch(model, data_loader, weight_preprocessor, 
                               product_idx_to_name, device, num_samples=10, 
                               save_dir='attention_visualizations'):
    """
    Visualize attention for multiple samples from the dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Denormalization transform
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    sample_count = 0
    results = []
    
    for batch in tqdm(data_loader, desc="Generating attention maps"):
        if sample_count >= num_samples:
            break
            
        images = batch['image'].to(device)
        category_indices = batch['category_idx'].to(device)
        numerical = batch['numerical'].to(device)
        targets = batch['weight']
        
        batch_size = images.size(0)
        
        for i in range(min(batch_size, num_samples - sample_count)):
            # Get single sample
            image = images[i:i+1]
            cat_idx = category_indices[i:i+1]
            num_feat = numerical[i:i+1]
            target = targets[i].item()
            
            # Enable gradients for saliency
            image.requires_grad = True
            
            # Forward pass with attention
            prediction, attention_weights = model(
                image, cat_idx, num_feat,
                return_attention=True
            )
            
            # Backward for saliency
            prediction.backward()
            
            # Get saliency
            gradients = image.grad.data.abs()
            saliency = gradients.squeeze().cpu()
            saliency_gray = saliency.mean(dim=0).numpy()
            saliency_gray = (saliency_gray - saliency_gray.min()) / (saliency_gray.max() - saliency_gray.min() + 1e-8)
            
            # Get prediction in kg
            with torch.no_grad():
                pred_value = model(image, cat_idx, num_feat)
                pred_kg = weight_preprocessor.inverse_transform(pred_value.cpu().numpy().flatten())[0]
                target_kg = weight_preprocessor.inverse_transform(np.array([target]))[0]
            
            # Denormalize image for display
            img_display = denormalize(image.squeeze().detach().cpu())
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(img_display)
            axes[0].set_title(f'Original Image', fontsize=12)
            axes[0].axis('off')
            
            # Saliency map
            axes[1].imshow(saliency_gray, cmap='hot')
            axes[1].set_title('Saliency Map\n(Gradient importance)', fontsize=12)
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(img_display)
            heatmap = axes[2].imshow(saliency_gray, cmap='jet', alpha=0.5)
            axes[2].set_title('Attention Overlay', fontsize=12)
            axes[2].axis('off')
            
            # Prediction info
            error = pred_kg - target_kg
            error_pct = (error / target_kg) * 100
            
            info_text = f"""
            Prediction Results
            ─────────────────────
            Predicted: {pred_kg:,.1f} kg
            Actual:    {target_kg:,.1f} kg
            Error:     {error:+,.1f} kg
            Rel Error: {error_pct:+.1f}%
            
            Cross-Attention Scores
            ─────────────────────
            V→M: Visual attending to metadata
            M→V: Metadata attending to visual
            
            Higher saliency (red/yellow) = 
            More important for prediction
            """
            
            axes[3].text(0.1, 0.5, info_text, fontsize=11, 
                        verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[3].axis('off')
            axes[3].set_title('Prediction Details', fontsize=12)
            
            plt.suptitle(f'Sample {sample_count + 1}: Weight Prediction Attention Analysis', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(save_dir, f'attention_sample_{sample_count + 1:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            results.append({
                'sample': sample_count + 1,
                'predicted_kg': pred_kg,
                'actual_kg': target_kg,
                'error_kg': error,
                'error_pct': error_pct,
                'save_path': save_path
            })
            
            sample_count += 1
    
    print(f"\n✓ Generated {sample_count} attention visualizations in '{save_dir}/'")
    
    return results


def create_attention_grid(model, data_loader, weight_preprocessor, device, 
                          num_samples=16, save_path='attention_grid.png'):
    """
    Create a grid showing attention maps for multiple samples.
    """
    model.eval()
    
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    samples = []
    
    for batch in data_loader:
        if len(samples) >= num_samples:
            break
            
        images = batch['image'].to(device)
        category_indices = batch['category_idx'].to(device)
        numerical = batch['numerical'].to(device)
        targets = batch['weight']
        
        for i in range(min(images.size(0), num_samples - len(samples))):
            image = images[i:i+1]
            cat_idx = category_indices[i:i+1]
            num_feat = numerical[i:i+1]
            target = targets[i].item()
            
            image.requires_grad = True
            
            prediction, _ = model(image, cat_idx, num_feat, return_attention=True)
            prediction.backward()
            
            gradients = image.grad.data.abs()
            saliency = gradients.squeeze().cpu().mean(dim=0).numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            with torch.no_grad():
                pred_value = model(image, cat_idx, num_feat)
                pred_kg = weight_preprocessor.inverse_transform(pred_value.cpu().numpy().flatten())[0]
                target_kg = weight_preprocessor.inverse_transform(np.array([target]))[0]
            
            img_display = denormalize(image.squeeze().detach().cpu())
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            samples.append({
                'image': img_display,
                'saliency': saliency,
                'pred_kg': pred_kg,
                'target_kg': target_kg
            })
    
    # Create grid
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, (ax, sample) in enumerate(zip(axes, samples)):
        ax.imshow(sample['image'])
        ax.imshow(sample['saliency'], cmap='jet', alpha=0.4)
        
        error = sample['pred_kg'] - sample['target_kg']
        color = 'green' if abs(error) < 100 else 'orange' if abs(error) < 200 else 'red'
        
        ax.set_title(f"Pred: {sample['pred_kg']:.0f}kg | Act: {sample['target_kg']:.0f}kg\n"
                    f"Error: {error:+.0f}kg", fontsize=10, color=color)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Attention Visualization Grid\n(Red/Yellow = High attention regions)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved attention grid: {save_path}")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model attention')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/best_model_phase2_20251203_094138.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='attention_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--grid', action='store_true',
                        help='Create a grid visualization')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ATTENTION VISUALIZATION")
    print("="*80)
    
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
    if 'weight_in_kg' in df_featured.columns:
        df_featured.rename(columns={'weight_in_kg': 'weight'}, inplace=True)
    
    data_dict = prepare_data(df_featured, BASE_IMAGE_PATH)
    
    # Load model
    print(f"\n📦 Loading model...")
    device = torch.device(DEVICE)
    
    model, preprocessor, _ = create_optimized_model(
        num_categories=data_dict['num_product_types'],
        num_numerical_features=len(data_dict['numerical_features']),
        scaler=data_dict['numerical_scaler'],
        device=DEVICE
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate visualizations
    if args.grid:
        print(f"\n🎨 Creating attention grid...")
        create_attention_grid(
            model=model,
            data_loader=data_dict['test_loader'],
            weight_preprocessor=data_dict['weight_preprocessor'],
            device=device,
            num_samples=args.num_samples,
            save_path=os.path.join(args.output_dir, 'attention_grid.png')
        )
    else:
        print(f"\n🎨 Generating {args.num_samples} attention visualizations...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = visualize_attention_batch(
            model=model,
            data_loader=data_dict['test_loader'],
            weight_preprocessor=data_dict['weight_preprocessor'],
            product_idx_to_name={},
            device=device,
            num_samples=args.num_samples,
            save_dir=args.output_dir
        )
        
        # Save results summary
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(args.output_dir, 'attention_results.csv'), index=False)
    
    print("\n" + "="*80)
    print("🎉 VISUALIZATION COMPLETE!")
    print(f"📁 Output: {args.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
