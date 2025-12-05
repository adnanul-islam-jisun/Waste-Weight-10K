"""
Structured Prompt Generator
Objective: Translate quantitative metrics into clear, context-rich prompts for LLM.

This module:
1. Takes structured metrics from PostHocAnalyzer
2. Formats them into predefined templates
3. Generates prompts suitable for Llama 3.1 8B
"""

import os
from typing import Dict, Any, Optional
from string import Template

from config.explanation_config import (
    TEMPLATES_DIR,
    DEFAULT_TEMPLATE,
    EXPLANATION_MAX_LENGTH,
    EXPLANATION_STYLE
)


class PromptGenerator:
    """
    Structured Prompt Generator for LLM-based explanations.
    
    Converts quantitative analysis metrics into natural language prompts
    that instruct the LLM on generating human-readable explanations.
    
    Args:
        template_path: Path to custom template file (optional)
        style: Explanation style ('professional', 'casual', 'technical')
        max_length: Maximum words for generated explanation
    """
    
    def __init__(
        self,
        template_path: Optional[str] = None,
        style: str = EXPLANATION_STYLE,
        max_length: int = EXPLANATION_MAX_LENGTH
    ):
        self.style = style
        self.max_length = max_length
        
        # Load template
        if template_path and os.path.exists(template_path):
            self.template = self._load_template(template_path)
        else:
            self.template = self._get_default_template()
    
    def _load_template(self, path: str) -> str:
        """Load template from file."""
        with open(path, 'r') as f:
            return f.read()
    
    def _get_default_template(self) -> str:
        """Get default explanation template for Llama 3.1."""
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert AI assistant specializing in explaining waste weight predictions from a multimodal machine learning model. Your role is to provide clear, insightful explanations of how the model made its prediction based on image and metadata analysis.

Guidelines:
- Be concise but informative (maximum {max_length} words)
- Use {style} language appropriate for waste management professionals
- Focus on actionable insights when possible
- If the prediction has high error, suggest possible reasons
- Highlight which input (image vs metadata) was more influential<|eot_id|><|start_header_id|>user<|end_header_id|>

Please explain the following weight prediction:

## Prediction Summary
- **Predicted Weight**: {prediction:.1f} kg
{actual_weight_section}

## Model Confidence
- **Confidence Score**: {confidence_score:.1%}
- **Error Category**: {error_category}

## Input Contribution Analysis
- **Image Contribution**: {image_contribution:.1%}
- **Metadata Contribution**: {metadata_contribution:.1%}

## Feature Insights
{feature_insights}

{shap_section}

Please provide a clear, {style} explanation of this prediction that:
1. Summarizes what the model predicted and its confidence
2. Explains which input modality (image or metadata) was more influential and why this matters
3. {error_instruction}
4. Provides any actionable insights for improving future predictions<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def generate(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a complete prompt from analysis metrics.
        
        Args:
            metrics: Dictionary of metrics from PostHocAnalyzer
        
        Returns:
            Formatted prompt string ready for LLM
        """
        # Prepare template variables
        template_vars = self._prepare_template_variables(metrics)
        
        # Format template
        try:
            prompt = self.template.format(**template_vars)
        except KeyError as e:
            # Fallback to simple formatting if template has issues
            prompt = self._generate_fallback_prompt(metrics)
        
        return prompt
    
    def _prepare_template_variables(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for template substitution."""
        variables = {
            'max_length': self.max_length,
            'style': self.style,
            'prediction': metrics.get('prediction', 0),
            'confidence_score': metrics.get('confidence_score', 0.5),
            'error_category': metrics.get('error_category', 'unknown'),
            'image_contribution': metrics.get('image_contribution', 0.5),
            'metadata_contribution': metrics.get('metadata_contribution', 0.5),
        }
        
        # Actual weight section (only if available)
        if metrics.get('actual_weight') is not None:
            actual = metrics['actual_weight']
            abs_error = metrics.get('absolute_error', 0)
            pct_error = metrics.get('percentage_error', 0)
            variables['actual_weight_section'] = f"""- **Actual Weight**: {actual:.1f} kg
- **Absolute Error**: {abs_error:.1f} kg
- **Percentage Error**: {pct_error:.1f}%"""
            variables['error_instruction'] = "Discusses the prediction error and potential reasons for any discrepancy"
        else:
            variables['actual_weight_section'] = "- **Actual Weight**: Not available (prediction only)"
            variables['error_instruction'] = "Notes that actual weight is unavailable for error assessment"
        
        # Feature insights
        variables['feature_insights'] = self._format_feature_insights(metrics)
        
        # SHAP section
        variables['shap_section'] = self._format_shap_section(metrics)
        
        return variables
    
    def _format_feature_insights(self, metrics: Dict[str, Any]) -> str:
        """Format feature-related insights."""
        insights = []
        
        # Visual feature magnitude
        if 'visual_feature_magnitude' in metrics:
            insights.append(f"- Visual feature strength: {metrics['visual_feature_magnitude']:.2f}")
        
        # Metadata feature magnitude
        if 'metadata_feature_magnitude' in metrics:
            insights.append(f"- Metadata feature strength: {metrics['metadata_feature_magnitude']:.2f}")
        
        # Feature dimensions
        if 'feature_dimensions' in metrics:
            dims = metrics['feature_dimensions']
            insights.append(f"- Image features: {dims.get('image_features', 'N/A')} dimensions")
            insights.append(f"- Metadata features: {dims.get('metadata_features', 'N/A')} dimensions")
        
        return '\n'.join(insights) if insights else "No detailed feature data available"
    
    def _format_shap_section(self, metrics: Dict[str, Any]) -> str:
        """Format SHAP analysis section."""
        if 'shap_image_importance' not in metrics:
            return "## SHAP Analysis\nSHAP analysis not available for this prediction."
        
        section = f"""## SHAP Feature Importance
- **Image Features Importance**: {metrics.get('shap_image_importance', 0):.1%}
- **Metadata Features Importance**: {metrics.get('shap_metadata_importance', 0):.1%}"""
        
        # Add top influential features if available
        if 'top_influential_features' in metrics:
            top_features = metrics['top_influential_features'][:5]
            if top_features:
                section += "\n\n**Top 5 Most Influential Features:**"
                for i, feat in enumerate(top_features, 1):
                    section += f"\n{i}. {feat['modality'].capitalize()} feature #{feat['feature_index']} (SHAP: {feat['shap_value']:.4f})"
        
        return section
    
    def _generate_fallback_prompt(self, metrics: Dict[str, Any]) -> str:
        """Generate a simple fallback prompt if template fails."""
        prediction = metrics.get('prediction', 0)
        actual = metrics.get('actual_weight')
        img_contrib = metrics.get('image_contribution', 0.5)
        meta_contrib = metrics.get('metadata_contribution', 0.5)
        confidence = metrics.get('confidence_score', 0.5)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant explaining waste weight predictions. Be concise and clear.<|eot_id|><|start_header_id|>user<|end_header_id|>

Explain this weight prediction:
- Predicted: {prediction:.1f} kg
- Actual: {actual:.1f if actual else 'N/A'} kg
- Image contribution: {img_contrib:.1%}
- Metadata contribution: {meta_contrib:.1%}
- Confidence: {confidence:.1%}

Provide a brief, professional explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def generate_batch_summary_prompt(self, summary_stats: Dict[str, float]) -> str:
        """
        Generate a prompt for summarizing batch analysis.
        
        Args:
            summary_stats: Summary statistics from PostHocAnalyzer.get_summary_statistics()
        
        Returns:
            Formatted prompt for batch summary
        """
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert AI assistant analyzing batch predictions from a waste weight prediction model. Provide a comprehensive summary of the model's performance.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please summarize the following batch prediction analysis:

## Error Metrics
- Mean Absolute Error: {summary_stats.get('mean_absolute_error', 0):.1f} kg
- Median Absolute Error: {summary_stats.get('median_absolute_error', 0):.1f} kg
- Standard Deviation: {summary_stats.get('std_absolute_error', 0):.1f} kg
- Mean Percentage Error: {summary_stats.get('mean_percentage_error', 0):.1f}%

## Error Distribution
- Excellent (≤50kg): {summary_stats.get('pct_excellent', 0):.1f}%
- Good (≤100kg): {summary_stats.get('pct_good', 0):.1f}%
- Acceptable (≤200kg): {summary_stats.get('pct_acceptable', 0):.1f}%
- Poor (>200kg): {summary_stats.get('pct_poor', 0):.1f}%

## Modality Analysis
- Average Image Contribution: {summary_stats.get('mean_image_contribution', 0):.1%}
- Average Metadata Contribution: {summary_stats.get('mean_metadata_contribution', 0):.1%}

## Confidence
- Average Confidence Score: {summary_stats.get('mean_confidence', 0):.1%}

Provide a comprehensive summary including:
1. Overall model performance assessment
2. Key strengths and weaknesses observed
3. Recommendations for improvement<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def generate_comparison_prompt(
        self,
        metrics_list: list,
        comparison_type: str = "weight_range"
    ) -> str:
        """
        Generate prompt for comparing predictions across different categories.
        
        Args:
            metrics_list: List of metrics dictionaries
            comparison_type: Type of comparison ('weight_range', 'modality', 'error')
        
        Returns:
            Formatted comparison prompt
        """
        # Group metrics by comparison type
        if comparison_type == "weight_range":
            groups = self._group_by_weight_range(metrics_list)
        elif comparison_type == "modality":
            groups = self._group_by_dominant_modality(metrics_list)
        else:
            groups = {"all": metrics_list}
        
        comparison_text = self._format_group_comparison(groups)
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert AI assistant analyzing waste weight prediction patterns across different categories.<|eot_id|><|start_header_id|>user<|end_header_id|>

Compare the model's performance across the following groups:

{comparison_text}

Provide insights on:
1. Performance differences between groups
2. Which groups the model handles best/worst
3. Potential reasons for observed patterns<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def _group_by_weight_range(self, metrics_list: list) -> Dict[str, list]:
        """Group metrics by weight range."""
        groups = {
            'Light (0-100kg)': [],
            'Medium (100-500kg)': [],
            'Heavy (500-1000kg)': [],
            'Very Heavy (1000+kg)': []
        }
        
        for m in metrics_list:
            weight = m.get('actual_weight') or m.get('prediction', 0)
            if weight <= 100:
                groups['Light (0-100kg)'].append(m)
            elif weight <= 500:
                groups['Medium (100-500kg)'].append(m)
            elif weight <= 1000:
                groups['Heavy (500-1000kg)'].append(m)
            else:
                groups['Very Heavy (1000+kg)'].append(m)
        
        return {k: v for k, v in groups.items() if v}
    
    def _group_by_dominant_modality(self, metrics_list: list) -> Dict[str, list]:
        """Group metrics by dominant modality."""
        groups = {
            'Image Dominant': [],
            'Metadata Dominant': [],
            'Balanced': []
        }
        
        for m in metrics_list:
            img_contrib = m.get('image_contribution', 0.5)
            if img_contrib > 0.6:
                groups['Image Dominant'].append(m)
            elif img_contrib < 0.4:
                groups['Metadata Dominant'].append(m)
            else:
                groups['Balanced'].append(m)
        
        return {k: v for k, v in groups.items() if v}
    
    def _format_group_comparison(self, groups: Dict[str, list]) -> str:
        """Format group comparison for prompt."""
        lines = []
        for group_name, metrics in groups.items():
            if not metrics:
                continue
            
            abs_errors = [m.get('absolute_error', 0) for m in metrics if 'absolute_error' in m]
            img_contribs = [m.get('image_contribution', 0.5) for m in metrics]
            
            lines.append(f"### {group_name}")
            lines.append(f"- Sample Count: {len(metrics)}")
            if abs_errors:
                lines.append(f"- Mean Absolute Error: {sum(abs_errors)/len(abs_errors):.1f} kg")
            lines.append(f"- Mean Image Contribution: {sum(img_contribs)/len(img_contribs):.1%}")
            lines.append("")
        
        return '\n'.join(lines)


# Template file creator
def create_default_template_file():
    """Create default template file in templates directory."""
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    template_path = os.path.join(TEMPLATES_DIR, DEFAULT_TEMPLATE)
    
    template_content = """# Waste Weight Prediction Explanation Template
# This template is used by PromptGenerator to create LLM prompts

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert AI assistant specializing in explaining waste weight predictions from a multimodal machine learning model. Your role is to provide clear, insightful explanations of how the model made its prediction based on image and metadata analysis.

Guidelines:
- Be concise but informative (maximum {max_length} words)
- Use {style} language appropriate for waste management professionals
- Focus on actionable insights when possible
- If the prediction has high error, suggest possible reasons
- Highlight which input (image vs metadata) was more influential<|eot_id|><|start_header_id|>user<|end_header_id|>

Please explain the following weight prediction:

## Prediction Summary
- **Predicted Weight**: {prediction:.1f} kg
{actual_weight_section}

## Model Confidence
- **Confidence Score**: {confidence_score:.1%}
- **Error Category**: {error_category}

## Input Contribution Analysis
- **Image Contribution**: {image_contribution:.1%}
- **Metadata Contribution**: {metadata_contribution:.1%}

## Feature Insights
{feature_insights}

{shap_section}

Please provide a clear, {style} explanation of this prediction.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    with open(template_path, 'w') as f:
        f.write(template_content)
    
    print(f"✓ Created template file: {template_path}")
    return template_path
