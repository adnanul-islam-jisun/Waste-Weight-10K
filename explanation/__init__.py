"""
Explanation Module - Stage 2: Explanation Pipeline
This module provides post-hoc explainability for weight predictions.

Components:
- PostHocAnalyzer: Computes metrics, modality contribution, SHAP importance
- PromptGenerator: Creates structured prompts for LLM
- LLMReasoning: Generates human-readable explanations using Llama 3.1 8B
- ExplanationGenerator: High-level wrapper combining all components
"""

from .post_hoc_analyzer import PostHocAnalyzer
from .prompt_generator import PromptGenerator, create_default_template_file
from .llm_reasoning import LLMReasoning, ExplanationGenerator


__all__ = [
    # Core Components
    'PostHocAnalyzer',
    'PromptGenerator',
    'LLMReasoning',
    
    # High-level Interface
    'ExplanationGenerator',
    
    # Utilities
    'create_default_template_file'
]


def get_explanation_pipeline(model, device='cuda', llm_config=None):
    """
    Factory function to create complete explanation pipeline.
    
    Args:
        model: Trained weight prediction model
        device: Device for analysis ('cuda' or 'cpu')
        llm_config: Optional LLM configuration dictionary
    
    Returns:
        ExplanationGenerator instance
    
    Example:
        >>> from explanation import get_explanation_pipeline
        >>> explainer = get_explanation_pipeline(model, device='cuda')
        >>> result = explainer.explain(image, category_idx, numerical, prediction, actual)
        >>> print(result['explanation'])
    """
    return ExplanationGenerator(
        model=model,
        device=device,
        llm_config=llm_config
    )


def explain_prediction(
    model,
    image,
    category_idx,
    numerical_features,
    predicted_weight,
    actual_weight=None,
    device='cuda'
):
    """
    Convenience function for single prediction explanation.
    
    Args:
        model: Trained weight prediction model
        image: Input image tensor (1, 3, H, W)
        category_idx: Category index tensor (1,)
        numerical_features: Numerical features tensor (1, N)
        predicted_weight: Predicted weight in kg
        actual_weight: Optional actual weight in kg
        device: Device for analysis
    
    Returns:
        Dictionary containing metrics and explanation
    """
    explainer = get_explanation_pipeline(model, device)
    return explainer.explain(
        image=image,
        category_idx=category_idx,
        numerical_features=numerical_features,
        predicted_weight=predicted_weight,
        actual_weight=actual_weight
    )
