"""
Create publication-ready panels that show input images, metadata, predicted weight,
and a concise explanation. Produces up to N examples per category.

Usage:
    python visualize_predictions_explanations.py \
        --checkpoint checkpoints/best_model_phase2_20260105_005726.pt \
        --per-category 3 \
        --output-dir paper_examples

Notes:
- Uses the same feature engineering pipeline and scaling as training via prepare_data.
- Can optionally call the LLM for a narrative explanation per sample (enabled by default).
- Images are read from BASE_IMAGE_PATH and must exist on disk.
"""

import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.config import BASE_IMAGE_PATH, CSV_PATH, DEVICE, IMAGE_SIZE
from config.training_config import WeightPreprocessor, create_optimized_model
from dataload.data_preprocessing import prepare_data
from explanation import PostHocAnalyzer, PromptGenerator, LLMReasoning
from features.feature_engineering import engineer_features


# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------

def set_publication_style():
    """Configure matplotlib/seaborn for paper figures."""
    sns.set_theme(style="white", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.25,
    })


# -----------------------------------------------------------------------------
# Data prep helpers
# -----------------------------------------------------------------------------

def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Match training-time type cleaning (lowercase + typo fixes)."""
    df = df.copy()
    df["Type"] = df["Type"].astype(str).str.strip().str.lower()
    type_corrections = {
        "grash": "grass",
        "bonet": "bonnet",
        "card board": "cardboard",
        "cylinder track": "cylinder_track",
        "car door": "car_door",
    }
    df["Type"] = df["Type"].replace(type_corrections)
    return df


def formal_type_names() -> Dict[str, str]:
    """Map cleaned training labels to formal display names for the paper."""
    return {
        "vehicle": "Automotive Scrap",
        "car_door": "Automotive Scrap",
        "bonnet": "Automotive Scrap",
        "back": "Automotive Scrap",
        "tire": "Automotive Scrap",
        "car": "Automotive Scrap",
        "grass": "General Trash",
        "cardboard": "Cardboard",
        "cylinder_track": "Cylindrical Object",
        "metal": "Ferrous Metal",
        "plastic": "Rigid Plastic",
        "wood": "Wood",
        "rubber": "Rubber",
        "battery": "Battery",
        "fridge": "Appliance",
        "foam": "Foam",
    }


def load_dataframe(csv_path: str, min_weight: float = 50.0) -> pd.DataFrame:
    """Load CSV, coerce numerics, filter, and clean types."""
    df = pd.read_csv(csv_path)
    cols_to_convert = ["V_x", "V_y", "V_z", "D_x", "D_y", "weight_in_kg"]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["weight_in_kg"])
    df = df[df["weight_in_kg"] >= min_weight]
    df = clean_types(df)
    return df


def ensure_images_exist(df: pd.DataFrame, base_image_path: str) -> pd.DataFrame:
    """Drop rows whose images are missing on disk."""
    def exists(rel_path: str) -> bool:
        return os.path.exists(os.path.join(base_image_path, rel_path))

    if "image_path" not in df.columns:
        raise ValueError("Expected 'image_path' column in the dataset.")

    mask = df["image_path"].apply(exists)
    kept = mask.sum()
    if kept < len(df):
        print(f"Warning: {len(df) - kept} samples dropped because images are missing")
    return df[mask].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Model + inference helpers
# -----------------------------------------------------------------------------

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model_with_scaler(data_dict: Dict, checkpoint: str, device: str):
    model, weight_preprocessor, _ = create_optimized_model(
        num_categories=data_dict["num_product_types"],
        num_numerical_features=len(data_dict["numerical_features"]),
        scaler=data_dict["numerical_scaler"],
        device=device,
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, weight_preprocessor


def predict_and_explain(
    row: pd.Series,
    model: torch.nn.Module,
    analyzer: PostHocAnalyzer,
    weight_preprocessor: WeightPreprocessor,
    numerical_features: List[str],
    scaler,
    transform,
    product_type_to_idx: Dict[str, int],
    device: str,
):
    """Run model on one row and return prediction, metrics, and tensors."""
    image_path = os.path.join(BASE_IMAGE_PATH, row["image_path"])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    cat_idx = product_type_to_idx.get(row["Type"], 0)
    category_tensor = torch.tensor([cat_idx], dtype=torch.long).to(device)

    numerical_raw = row[numerical_features].astype(float).values.reshape(1, -1)
    # Use DataFrame with column names to avoid sklearn warning about missing feature names
    numerical_df = pd.DataFrame(numerical_raw, columns=numerical_features)
    numerical_scaled = scaler.transform(numerical_df)
    numerical_tensor = torch.tensor(numerical_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(image_tensor, category_tensor, numerical_tensor).squeeze()
    pred_kg = float(weight_preprocessor.inverse_transform(pred.cpu().numpy()))

    actual_kg = float(row["weight"])
    metrics = analyzer.analyze(
        image=image_tensor,
        category_idx=category_tensor,
        numerical_features=numerical_tensor,
        predicted_weight=pred_kg,
        actual_weight=actual_kg,
        weight_preprocessor=weight_preprocessor,
    )
    return pred_kg, actual_kg, metrics, image


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def make_explanation_text(row: pd.Series, pred: float, actual: float, metrics: Dict, pretty: str) -> str:
    err = pred - actual
    img_c = metrics.get("image_contribution", 0.0) * 100
    meta_c = metrics.get("metadata_contribution", 0.0) * 100
    conf = metrics.get("confidence_score", None)
    conf_txt = f", conf {conf*100:.0f}%" if conf is not None else ""
    return (
        f"{pretty}\n"
        f"V=({row['V_x']:.0f},{row['V_y']:.0f},{row['V_z']:.0f}) in, "
        f"D=({row['D_x']:.0f},{row['D_y']:.0f})\n"
        f"Pred {pred:.0f} kg | Actual {actual:.0f} kg | Error {err:+.0f} kg\n"
        f"Modalities: image {img_c:.0f}%, metadata {meta_c:.0f}%{conf_txt}"
    )


def shorten_text(text: str, max_chars: int = 260) -> str:
    """Compact LLM output for figure captions."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    cutoff = compact[:max_chars].rsplit(" ", 1)[0]
    return cutoff + "..."


def plot_examples_grid(
    examples: Dict[str, List[Dict]],
    per_category: int,
    display_names: Dict[str, str],
    output_path: str,
):
    categories = sorted(examples.keys())
    n_rows = len(categories)
    n_cols = per_category
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for r, cat in enumerate(categories):
        cat_examples = examples[cat]
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(cat_examples):
                ax.axis("off")
                continue
            ex = cat_examples[c]
            ax.imshow(ex["image"])
            ax.axis("off")
            title = f"{display_names.get(cat, cat.title())} #{c + 1}"
            ax.set_title(title, fontsize=11)
            ax.text(
                0.02,
                -0.12,
                ex["explanation"],
                transform=ax.transAxes,
                fontsize=8.5,
                va="top",
                ha="left",
                wrap=True,
            )
    plt.tight_layout(h_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper-ready example panels with predictions and explanations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt). If omitted, picks latest in checkpoints/")
    parser.add_argument("--per-category", type=int, default=3, help="Number of examples per category")
    parser.add_argument("--output-dir", type=str, default="paper_examples", help="Directory to save outputs")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM generation; use metrics-only captions")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=180, help="LLM max new tokens")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load LLM in 4-bit (memory-saving)")
    parser.add_argument("--use-api", action="store_true", help="Use HuggingFace Inference API instead of local model")
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    set_publication_style()

    # Resolve checkpoint (allow auto-pick latest)
    if args.checkpoint is None:
        ckpts = [
            f for f in os.listdir("checkpoints")
            if f.endswith(".pt")
        ] if os.path.exists("checkpoints") else []
        if ckpts:
            ckpts.sort(key=lambda f: os.path.getmtime(os.path.join("checkpoints", f)), reverse=True)
            args.checkpoint = os.path.join("checkpoints", ckpts[0])
            print(f"Using latest checkpoint: {args.checkpoint}")
        else:
            raise FileNotFoundError("No checkpoint provided and none found in checkpoints/")

    print("Loading and cleaning data...")
    df = load_dataframe(CSV_PATH)
    df = ensure_images_exist(df, BASE_IMAGE_PATH)

    # Feature engineering
    df_features = engineer_features(df)
    if "weight_in_kg" in df_features.columns:
        df_features = df_features.rename(columns={"weight_in_kg": "weight"})

    # Prepare scaling and mappings using the training helper
    data_dict = prepare_data(df_features, BASE_IMAGE_PATH, batch_size=8)
    product_type_to_idx = data_dict["product_type_to_idx"]
    numerical_features = data_dict["numerical_features"]

    print("Loading model and checkpoint...")
    model, weight_preprocessor = load_model_with_scaler(data_dict, args.checkpoint, DEVICE)
    analyzer = PostHocAnalyzer(model=model, device=DEVICE)
    transform = get_image_transform()

    llm = None
    prompt_gen = None
    if not args.no_llm:
        prompt_gen = PromptGenerator()
        llm = LLMReasoning(
            load_in_8bit=not args.load_in_4bit,
            load_in_4bit=args.load_in_4bit,
            use_api=args.use_api,
        )
        llm.update_generation_config(
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )
        print("LLM explanations enabled.")
    else:
        print("LLM explanations disabled (metrics-only captions).")

    display_map = formal_type_names()
    examples: Dict[str, List[Dict]] = defaultdict(list)

    print("Sampling examples and generating explanations...")
    for cat, idx in product_type_to_idx.items():
        subset = df_features[df_features["Type"] == cat]
        if subset.empty:
            continue
        subset = subset.sample(frac=1.0, random_state=args.random_seed)  # shuffle
        for _, row in subset.iterrows():
            if len(examples[cat]) >= args.per_category:
                break
            try:
                pred, actual, metrics, image = predict_and_explain(
                    row=row,
                    model=model,
                    analyzer=analyzer,
                    weight_preprocessor=weight_preprocessor,
                    numerical_features=numerical_features,
                    scaler=data_dict["numerical_scaler"],
                    transform=transform,
                    product_type_to_idx=product_type_to_idx,
                    device=DEVICE,
                )
                explanation_text = make_explanation_text(row, pred, actual, metrics, display_map.get(cat, cat.title()))

                llm_text = None
                llm_short = None
                if llm and prompt_gen:
                    prompt = prompt_gen.generate(metrics)
                    llm_text = llm.generate(prompt)
                    llm_short = shorten_text(llm_text)

                final_caption = explanation_text if not llm_short else f"{explanation_text}\n\nLLM: {llm_short}"
                examples[cat].append(
                    {
                        "image_path": row["image_path"],
                        "image": image,
                        "predicted_kg": pred,
                        "actual_kg": actual,
                        "explanation": final_caption,
                        "llm_text": llm_text,
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                print(f"Skipping sample in category {cat}: {e}")

    # Filter out categories with no examples
    examples = {k: v for k, v in examples.items() if v}
    if not examples:
        raise RuntimeError("No examples were generated. Check data and paths.")

    print("Rendering figure...")
    figure_path = os.path.join(args.output_dir, "category_examples.png")
    plot_examples_grid(examples, args.per_category, display_map, figure_path)
    print(f"Saved grid figure to {figure_path}")

    # Save tabular summary
    records = []
    for cat, ex_list in examples.items():
        for ex in ex_list:
            records.append(
                {
                    "category": cat,
                    "display_category": display_map.get(cat, cat.title()),
                    "image_path": ex["image_path"],
                    "predicted_kg": ex["predicted_kg"],
                    "actual_kg": ex["actual_kg"],
                    "absolute_error": ex["predicted_kg"] - ex["actual_kg"],
                    "llm_text": ex.get("llm_text", ""),
                }
            )
    summary_path = os.path.join(args.output_dir, "category_examples.csv")
    pd.DataFrame.from_records(records).to_csv(summary_path, index=False)
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
