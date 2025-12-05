#!/usr/bin/env python3
"""
Explain.py - Stage 2: Explanation Pipeline Entry Point
Generates human-readable explanations for weight predictions.

Usage:
    # Explain predictions on test set
    python explain.py --checkpoint checkpoints/best_model_phase2_*.pt
    
    # Explain single image
    python explain.py --image path/to/image.jpg --category 0 --numerical "1.2,3.4,5.6"
    
    # Batch explanation with custom LLM settings
    python explain.py --checkpoint model.pt --temperature 0.5 --max-samples 50
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Project imports
from config.config import (
    DEVICE, IMAGE_SIZE, CHECKPOINT_DIR,
    CSV_PATH, BASE_IMAGE_PATH
)
from config.explanation_config import (
    EXPLANATION_OUTPUT_DIR,
    SAVE_EXPLANATIONS_TO_FILE,
    SAVE_METRICS_JSON,
    print_explanation_config
)
from config.training_config import create_optimized_model, WeightPreprocessor
from Dataload.data_preprocessing import prepare_data
from explanation import (
    PostHocAnalyzer,
    PromptGenerator,
    LLMReasoning,
    ExplanationGenerator
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate explanations for weight predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain test set predictions
  python explain.py --checkpoint checkpoints/best_model_phase2_*.pt
  
  # Explain with specific number of samples
  python explain.py --checkpoint model.pt --max-samples 20
  
  # Single image explanation
  python explain.py --image image.jpg --category 0 --numerical "1.2,3.4,5.6"
  
  # Use 4-bit quantization for LLM (lower memory)
  python explain.py --checkpoint model.pt --load-in-4bit
        """
    )
    
    # Model and data arguments
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='Path to single image for explanation'
    )
    parser.add_argument(
        '--category',
        type=int,
        default=0,
        help='Category index for single image'
    )
    parser.add_argument(
        '--numerical',
        type=str,
        default=None,
        help='Comma-separated numerical features for single image'
    )
    parser.add_argument(
        '--actual-weight',
        type=float,
        default=None,
        help='Actual weight in kg (optional, for error analysis)'
    )
    
    # Batch processing arguments
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10,
        help='Maximum number of samples to explain'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing'
    )
    
    # LLM arguments
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        default=True,
        help='Use 8-bit quantization (default)'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        default=False,
        help='Use 4-bit quantization (lower memory)'
    )
    parser.add_argument(
        '--use-api',
        action='store_true',
        default=False,
        help='Use Hugging Face Inference API'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        default=False,
        help='Skip LLM and only compute metrics'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=EXPLANATION_OUTPUT_DIR,
        help='Output directory for explanations'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        default=True,
        help='Save metrics as JSON'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Verbose output'
    )
    
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> Optional[str]:
    """Find the latest checkpoint in directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for phase2 checkpoints first (fully trained)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model_phase2')]
    if not checkpoints:
        # Fall back to any checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])


def load_model(checkpoint_path: str, device: str = DEVICE):
    """Load trained model from checkpoint."""
    print(f"\n📦 Loading model from: {checkpoint_path}")
    
    # Load checkpoint to get metadata
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Infer model configuration from state dict shapes
    # Category embedding shape: (num_categories, embedding_dim)
    cat_emb_key = 'metadata_encoder.category_embedding.weight'
    if cat_emb_key in state_dict:
        num_categories = state_dict[cat_emb_key].shape[0]
    else:
        num_categories = checkpoint.get('num_categories', 10)
    
    # Numerical MLP input shape: (hidden_dim, num_numerical_features)
    num_mlp_key = 'metadata_encoder.numerical_mlp.0.weight'
    if num_mlp_key in state_dict:
        num_numerical = state_dict[num_mlp_key].shape[1]
    else:
        num_numerical = checkpoint.get('num_numerical_features', 6)
    
    print(f"  Detected: {num_categories} categories, {num_numerical} numerical features")
    
    # Create model architecture
    model, weight_preprocessor, _ = create_optimized_model(
        num_categories=num_categories,
        num_numerical_features=num_numerical,
        device=device
    )
    
    # Load weights (strict=False to handle scaler buffers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Load preprocessor state if available
    if 'weight_preprocessor' in checkpoint:
        weight_preprocessor = checkpoint['weight_preprocessor']
    
    epoch_info = checkpoint.get('epoch', 'N/A')
    val_mae = checkpoint.get('val_mae', None)
    print(f"  ✓ Model loaded (Epoch {epoch_info})")
    if val_mae:
        print(f"  ✓ Val MAE: {val_mae:.2f} kg")
    
    return model, weight_preprocessor, num_categories, num_numerical


def get_image_transform():
    """Get image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def explain_single_image(
    args,
    model,
    weight_preprocessor,
    device: str = DEVICE
):
    """Generate explanation for a single image."""
    print(f"\n🖼️ Processing single image: {args.image}")
    
    # Load and transform image
    image = Image.open(args.image).convert('RGB')
    transform = get_image_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prepare category and numerical features
    category_tensor = torch.tensor([args.category], dtype=torch.long).to(device)
    
    if args.numerical:
        numerical = [float(x) for x in args.numerical.split(',')]
    else:
        numerical = [0.0] * 6  # Default zeros
    numerical_tensor = torch.tensor([numerical], dtype=torch.float32).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor, category_tensor, numerical_tensor)
        pred_value = prediction.squeeze().cpu().numpy()
        pred_kg = weight_preprocessor.inverse_transform(np.array([pred_value]))[0]
    
    print(f"  Predicted weight: {pred_kg:.1f} kg")
    
    # Initialize explanation components
    analyzer = PostHocAnalyzer(model=model, device=device)
    prompt_gen = PromptGenerator()
    
    # Analyze prediction
    metrics = analyzer.analyze(
        image=image_tensor,
        category_idx=category_tensor,
        numerical_features=numerical_tensor,
        predicted_weight=pred_kg,
        actual_weight=args.actual_weight
    )
    
    # Generate prompt
    prompt = prompt_gen.generate(metrics)
    
    # Generate explanation with LLM (unless disabled)
    if not args.no_llm:
        llm = LLMReasoning(
            load_in_8bit=args.load_in_8bit and not args.load_in_4bit,
            load_in_4bit=args.load_in_4bit,
            use_api=args.use_api
        )
        llm.update_generation_config(
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
        explanation = llm.generate(prompt)
    else:
        explanation = "LLM disabled. Metrics computed only."
    
    # Display results
    print("\n" + "="*80)
    print("📊 ANALYSIS RESULTS")
    print("="*80)
    print(f"\n🎯 Prediction: {pred_kg:.1f} kg")
    if args.actual_weight:
        print(f"📏 Actual: {args.actual_weight:.1f} kg")
        print(f"❌ Error: {metrics.get('absolute_error', 0):.1f} kg ({metrics.get('percentage_error', 0):.1f}%)")
    print(f"\n📈 Confidence: {metrics.get('confidence_score', 0):.1%}")
    print(f"🖼️ Image contribution: {metrics.get('image_contribution', 0):.1%}")
    print(f"📋 Metadata contribution: {metrics.get('metadata_contribution', 0):.1%}")
    
    print("\n" + "="*80)
    print("💬 EXPLANATION")
    print("="*80)
    print(f"\n{explanation}")
    
    # Save results
    if args.save_json:
        save_results(
            {'single_prediction': {
                'image': args.image,
                'metrics': metrics,
                'explanation': explanation
            }},
            args.output_dir,
            'single_explanation'
        )
    
    return metrics, explanation


def explain_test_set(
    args,
    model,
    weight_preprocessor,
    device: str = DEVICE
):
    """Generate explanations for test set predictions."""
    print("\n📊 Loading test data...")
    
    # Load and prepare data the same way as training
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} samples from {CSV_PATH}")
    
    # Convert numeric columns to proper types
    numeric_cols = ['V_x', 'V_y', 'V_z', 'D_x', 'D_y', 'D_z', 'weight_in_kg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])
    
    # Filter minimum weight
    MIN_WEIGHT_KG = 50
    if 'weight_in_kg' in df.columns:
        df = df[df['weight_in_kg'] >= MIN_WEIGHT_KG]
    elif 'weight' in df.columns:
        df = df[df['weight'] >= MIN_WEIGHT_KG]
    
    # Feature engineering
    from features.feature_engineering import engineer_features
    df_featured = engineer_features(df)
    
    # Rename weight column if needed
    if 'weight_in_kg' in df_featured.columns and 'weight' not in df_featured.columns:
        df_featured.rename(columns={'weight_in_kg': 'weight'}, inplace=True)
    
    # Prepare data
    data_dict = prepare_data(df_featured, BASE_IMAGE_PATH)
    test_loader = data_dict['test_loader']
    
    # Initialize components
    analyzer = PostHocAnalyzer(model=model, device=device)
    prompt_gen = PromptGenerator()
    
    if not args.no_llm:
        llm = LLMReasoning(
            load_in_8bit=args.load_in_8bit and not args.load_in_4bit,
            load_in_4bit=args.load_in_4bit,
            use_api=args.use_api
        )
        llm.update_generation_config(
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
    else:
        llm = None
    
    # Process samples
    results = []
    sample_count = 0
    
    print(f"\n🔍 Generating explanations for up to {args.max_samples} samples...")
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            if sample_count >= args.max_samples:
                break
            
            images = batch['image'].to(device)
            category_indices = batch['category_idx'].to(device)
            numerical = batch['numerical'].to(device)
            targets = batch['weight'].cpu().numpy()
            
            # Make predictions
            predictions = model(images, category_indices, numerical).squeeze()
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            
            # Process each sample in batch
            for i in range(min(len(images), args.max_samples - sample_count)):
                # Get prediction in kg
                pred_value = predictions[i].cpu().numpy()
                pred_kg = weight_preprocessor.inverse_transform(np.array([pred_value]))[0]
                actual_kg = weight_preprocessor.inverse_transform(np.array([targets[i]]))[0]
                
                # Analyze
                metrics = analyzer.analyze(
                    image=images[i:i+1],
                    category_idx=category_indices[i:i+1],
                    numerical_features=numerical[i:i+1],
                    predicted_weight=pred_kg,
                    actual_weight=actual_kg
                )
                
                # Generate explanation
                if llm:
                    prompt = prompt_gen.generate(metrics)
                    explanation = llm.generate(prompt)
                else:
                    explanation = "LLM disabled"
                
                results.append({
                    'sample_id': sample_count,
                    'predicted_weight': pred_kg,
                    'actual_weight': actual_kg,
                    'metrics': metrics,
                    'explanation': explanation
                })
                
                sample_count += 1
                
                if args.verbose:
                    print(f"\n--- Sample {sample_count} ---")
                    print(f"Predicted: {pred_kg:.1f} kg, Actual: {actual_kg:.1f} kg")
                    print(f"Error: {metrics.get('absolute_error', 0):.1f} kg")
    
    # Compute summary statistics
    summary = analyzer.get_summary_statistics([r['metrics'] for r in results])
    
    # Display summary
    print("\n" + "="*80)
    print("📈 BATCH ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nSamples analyzed: {len(results)}")
    print(f"Mean Absolute Error: {summary.get('mean_absolute_error', 0):.1f} kg")
    print(f"Median Absolute Error: {summary.get('median_absolute_error', 0):.1f} kg")
    print(f"Mean Percentage Error: {summary.get('mean_percentage_error', 0):.1f}%")
    print(f"\nError Distribution:")
    print(f"  Excellent (≤50kg): {summary.get('pct_excellent', 0):.1f}%")
    print(f"  Good (≤100kg): {summary.get('pct_good', 0):.1f}%")
    print(f"  Acceptable (≤200kg): {summary.get('pct_acceptable', 0):.1f}%")
    print(f"  Poor (>200kg): {summary.get('pct_poor', 0):.1f}%")
    print(f"\nModality Contribution:")
    print(f"  Mean Image: {summary.get('mean_image_contribution', 0):.1%}")
    print(f"  Mean Metadata: {summary.get('mean_metadata_contribution', 0):.1%}")
    
    # Save results
    if args.save_json:
        save_results(
            {
                'summary': summary,
                'samples': results
            },
            args.output_dir,
            'batch_explanations'
        )
    
    # Generate batch summary explanation
    if llm:
        print("\n" + "="*80)
        print("💬 BATCH SUMMARY EXPLANATION")
        print("="*80)
        summary_prompt = prompt_gen.generate_batch_summary_prompt(summary)
        summary_explanation = llm.generate(summary_prompt)
        print(f"\n{summary_explanation}")
    
    return results, summary


def save_results(data: Dict, output_dir: str, prefix: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    data = convert(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*80)
    print("🔬 STAGE 2: EXPLANATION PIPELINE")
    print("="*80)
    
    # Print configuration
    if args.verbose:
        print_explanation_config()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoint found. Please specify --checkpoint")
            sys.exit(1)
    
    # Load model
    model, weight_preprocessor, num_categories, num_numerical = load_model(
        checkpoint_path, DEVICE
    )
    
    # Run appropriate mode
    if args.image:
        # Single image mode
        explain_single_image(args, model, weight_preprocessor, DEVICE)
    else:
        # Batch test set mode
        explain_test_set(args, model, weight_preprocessor, DEVICE)
    
    print("\n✅ Explanation pipeline complete!")


if __name__ == "__main__":
    main()
