"""
Configuration for Stage 2: Explanation Pipeline
Settings for Post-Hoc Analysis, Prompt Generation, and LLM Reasoning.
"""

import os
from typing import Optional


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Llama 3.1 8B Configuration
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Local path to downloaded model (set to use local instead of downloading)
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH", 
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/llama-3.1-8b-instruct")
)

# Hugging Face API (alternative to local model)
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", None)
USE_HUGGINGFACE_API = os.getenv("USE_HUGGINGFACE_API", "false").lower() == "true"

# LLM Generation Parameters
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_TOP_K = 50
LLM_DO_SAMPLE = True
LLM_REPETITION_PENALTY = 1.1

# Device for LLM (can be different from main model)
LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")  # "auto", "cuda", "cpu"
LLM_LOAD_IN_8BIT = True  # Use 8-bit quantization to save memory
LLM_LOAD_IN_4BIT = False  # Use 4-bit quantization (even more memory efficient)


# ============================================================================
# SHAP EXPLAINABILITY CONFIGURATION
# ============================================================================

# SHAP Explainer Settings
SHAP_BACKGROUND_SAMPLES = 100  # Number of background samples for SHAP
SHAP_MAX_EVAL_SAMPLES = 500    # Max samples to evaluate
SHAP_ALGORITHM = "kernel"       # "kernel", "deep", "gradient"

# Feature Attribution Settings
ENABLE_MODALITY_CONTRIBUTION = True
ENABLE_FEATURE_IMPORTANCE = True
ENABLE_ATTENTION_WEIGHTS = True


# ============================================================================
# POST-HOC ANALYZER CONFIGURATION
# ============================================================================

# Error Thresholds (in kg)
ERROR_THRESHOLDS = {
    'excellent': 50,    # Error <= 50kg
    'good': 100,        # Error <= 100kg
    'acceptable': 200,  # Error <= 200kg
    'poor': 500         # Error > 200kg
}

# Analysis Components
COMPUTE_ABSOLUTE_ERROR = True
COMPUTE_PERCENTAGE_ERROR = True
COMPUTE_CONFIDENCE_SCORE = True


# ============================================================================
# PROMPT TEMPLATE CONFIGURATION
# ============================================================================

# Template Directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "explanation", "templates")

# Default Template
DEFAULT_TEMPLATE = "explanation_template.txt"

# Output Format Settings
EXPLANATION_MAX_LENGTH = 300  # Maximum words in explanation
EXPLANATION_STYLE = "professional"  # "professional", "casual", "technical"


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output Directory for Explanations
EXPLANATION_OUTPUT_DIR = "explanation_outputs"

# Create directory if not exists
os.makedirs(EXPLANATION_OUTPUT_DIR, exist_ok=True)

# Save Options
SAVE_EXPLANATIONS_TO_FILE = True
SAVE_METRICS_JSON = True
SAVE_SHAP_PLOTS = True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_llm_config() -> dict:
    """Get LLM configuration as dictionary."""
    return {
        'model_name': LLM_MODEL_NAME,
        'model_path': LLM_MODEL_PATH,
        'max_new_tokens': LLM_MAX_NEW_TOKENS,
        'temperature': LLM_TEMPERATURE,
        'top_p': LLM_TOP_P,
        'top_k': LLM_TOP_K,
        'do_sample': LLM_DO_SAMPLE,
        'repetition_penalty': LLM_REPETITION_PENALTY,
        'device': LLM_DEVICE,
        'load_in_8bit': LLM_LOAD_IN_8BIT,
        'load_in_4bit': LLM_LOAD_IN_4BIT,
        'use_api': USE_HUGGINGFACE_API,
        'api_token': HUGGINGFACE_API_TOKEN,
        'model_path': LLM_MODEL_PATH
    }


def get_shap_config() -> dict:
    """Get SHAP configuration as dictionary."""
    return {
        'background_samples': SHAP_BACKGROUND_SAMPLES,
        'max_eval_samples': SHAP_MAX_EVAL_SAMPLES,
        'algorithm': SHAP_ALGORITHM,
        'modality_contribution': ENABLE_MODALITY_CONTRIBUTION,
        'feature_importance': ENABLE_FEATURE_IMPORTANCE,
        'attention_weights': ENABLE_ATTENTION_WEIGHTS
    }


def print_explanation_config():
    """Print current explanation configuration."""
    print("\n" + "="*80)
    print("STAGE 2: EXPLANATION PIPELINE CONFIGURATION")
    print("="*80)
    
    print(f"\n🤖 LLM Settings:")
    print(f"  Model: {LLM_MODEL_NAME}")
    print(f"  Device: {LLM_DEVICE}")
    print(f"  8-bit Quantization: {LLM_LOAD_IN_8BIT}")
    print(f"  Max Tokens: {LLM_MAX_NEW_TOKENS}")
    print(f"  Temperature: {LLM_TEMPERATURE}")
    
    print(f"\n🔍 SHAP Settings:")
    print(f"  Algorithm: {SHAP_ALGORITHM}")
    print(f"  Background Samples: {SHAP_BACKGROUND_SAMPLES}")
    print(f"  Modality Contribution: {ENABLE_MODALITY_CONTRIBUTION}")
    
    print(f"\n📝 Output Settings:")
    print(f"  Output Directory: {EXPLANATION_OUTPUT_DIR}")
    print(f"  Save Explanations: {SAVE_EXPLANATIONS_TO_FILE}")
    print(f"  Save SHAP Plots: {SAVE_SHAP_PLOTS}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_explanation_config()
