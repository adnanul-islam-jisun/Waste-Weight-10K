"""
Post-Hoc Data Extractor & Analyzer
Objective: Gather and compute quantitative evidence about the model's behavior for a single prediction.

This module performs:
1. Performance Metrics: Absolute Error, Percentage Error
2. Explainability Analysis using SHAP
3. Modality Contribution: Image vs Metadata influence
4. Feature Importance: Categorical vs Numerical metadata
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Feature importance analysis will be limited.")

from config.explanation_config import (
    get_shap_config,
    COMPUTE_ABSOLUTE_ERROR,
    COMPUTE_PERCENTAGE_ERROR,
    COMPUTE_CONFIDENCE_SCORE,
    ERROR_THRESHOLDS
)


class PostHocAnalyzer:
    """
    Post-Hoc Data Extractor & Analyzer for weight predictions.
    
    Analyzes model predictions without affecting the prediction itself.
    Computes:
    - Performance metrics (error analysis)
    - Modality contribution (image vs metadata)
    - Feature importance via SHAP
    
    Args:
        model: Trained multimodal weight prediction model
        device: Device to run analysis on
        background_data: Background dataset for SHAP (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        background_data: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.background_data = background_data
        self.shap_config = get_shap_config()
        
        # Get feature dimensions from model
        self._extract_model_dimensions()
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if SHAP_AVAILABLE and background_data is not None:
            self._initialize_shap_explainer()
    
    def _extract_model_dimensions(self):
        """Extract feature dimensions from the model architecture."""
        try:
            # Get image feature dimension
            self.image_feature_dim = self.model.image_encoder.get_output_dim()
            # Get metadata feature dimension
            self.metadata_feature_dim = self.model.metadata_encoder.get_output_dim()
            # Total fused dimension
            self.fused_dim = self.image_feature_dim + self.metadata_feature_dim
        except AttributeError:
            # Fallback to default dimensions
            self.image_feature_dim = 768
            self.metadata_feature_dim = 256
            self.fused_dim = 1024
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for the regression head."""
        if not SHAP_AVAILABLE:
            return
        
        try:
            # Create a wrapper function for the regression head
            def regression_head_wrapper(fused_features):
                """Wrapper for SHAP to analyze regression head."""
                with torch.no_grad():
                    fused_tensor = torch.tensor(fused_features, dtype=torch.float32).to(self.device)
                    
                    # Handle the residual connection if present
                    if hasattr(self.model, 'use_residual') and self.model.use_residual:
                        output = self.model.regression_head(fused_tensor)
                    else:
                        output = self.model.regression_head(fused_tensor)
                    
                    return output.cpu().numpy()
            
            # Get background fused features
            background_fused = self._get_background_fused_features()
            
            if background_fused is not None:
                self.shap_explainer = shap.KernelExplainer(
                    regression_head_wrapper,
                    background_fused[:self.shap_config['background_samples']]
                )
                print("✓ SHAP Explainer initialized successfully")
        except Exception as e:
            warnings.warn(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _get_background_fused_features(self) -> Optional[np.ndarray]:
        """Extract fused features from background data."""
        if self.background_data is None:
            return None
        
        try:
            with torch.no_grad():
                images = self.background_data['images'].to(self.device)
                category_indices = self.background_data['category_indices'].to(self.device)
                numerical = self.background_data['numerical'].to(self.device)
                
                # Get intermediate representations
                features = self.model.get_feature_representations(
                    images, category_indices, numerical
                )
                return features['fused_features'].cpu().numpy()
        except Exception as e:
            warnings.warn(f"Failed to extract background features: {e}")
            return None
    
    def analyze(
        self,
        image: torch.Tensor,
        category_idx: torch.Tensor,
        numerical_features: torch.Tensor,
        predicted_weight: float,
        actual_weight: Optional[float] = None,
        weight_preprocessor=None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive post-hoc analysis on a single prediction.
        
        Args:
            image: Input image tensor (1, 3, H, W)
            category_idx: Category index tensor (1,)
            numerical_features: Numerical features tensor (1, N)
            predicted_weight: Model's predicted weight (in kg, after inverse transform)
            actual_weight: Ground truth weight in kg (optional)
            weight_preprocessor: Preprocessor for weight transformation
        
        Returns:
            Dictionary containing all computed metrics and analysis
        """
        metrics = {
            'prediction': predicted_weight,
            'actual_weight': actual_weight
        }
        
        # ===== PERFORMANCE METRICS =====
        if actual_weight is not None:
            metrics.update(self._compute_error_metrics(predicted_weight, actual_weight))
        
        # ===== FEATURE EXTRACTION =====
        feature_data = self._extract_features(image, category_idx, numerical_features)
        metrics['feature_dimensions'] = {
            'image_features': self.image_feature_dim,
            'metadata_features': self.metadata_feature_dim,
            'fused_features': self.fused_dim
        }
        
        # ===== MODALITY CONTRIBUTION =====
        modality_contrib = self._compute_modality_contribution(feature_data)
        metrics.update(modality_contrib)
        
        # ===== SHAP FEATURE IMPORTANCE =====
        if self.shap_explainer is not None and self.shap_config['feature_importance']:
            shap_analysis = self._compute_shap_importance(feature_data['fused_features'])
            metrics.update(shap_analysis)
        
        # ===== ATTENTION WEIGHTS (if available) =====
        if self.shap_config['attention_weights']:
            attention_data = self._extract_attention_weights(image, category_idx, numerical_features)
            if attention_data:
                metrics['attention_weights'] = attention_data
        
        # ===== CONFIDENCE ESTIMATION =====
        if COMPUTE_CONFIDENCE_SCORE:
            confidence = self._estimate_confidence(feature_data, predicted_weight)
            metrics['confidence_score'] = confidence
        
        # ===== ERROR CLASSIFICATION =====
        if actual_weight is not None:
            metrics['error_category'] = self._classify_error(metrics.get('absolute_error', 0))
        
        return metrics
    
    def _compute_error_metrics(self, predicted: float, actual: float) -> Dict[str, float]:
        """Compute error metrics between prediction and actual."""
        metrics = {}
        
        if COMPUTE_ABSOLUTE_ERROR:
            metrics['absolute_error'] = abs(predicted - actual)
        
        if COMPUTE_PERCENTAGE_ERROR:
            metrics['percentage_error'] = abs(predicted - actual) / (actual + 1e-8) * 100
        
        # Additional error metrics
        metrics['signed_error'] = predicted - actual
        metrics['squared_error'] = (predicted - actual) ** 2
        
        return metrics
    
    def _extract_features(
        self,
        image: torch.Tensor,
        category_idx: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Extract intermediate feature representations."""
        with torch.no_grad():
            image = image.to(self.device)
            category_idx = category_idx.to(self.device)
            numerical_features = numerical_features.to(self.device)
            
            # Use model's get_feature_representations method
            features = self.model.get_feature_representations(
                image, category_idx, numerical_features
            )
            
            return {
                'visual_features': features['visual_features'].cpu().numpy(),
                'metadata_features': features['metadata_features'].cpu().numpy(),
                'fused_features': features['fused_features'].cpu().numpy()
            }
    
    def _compute_modality_contribution(self, feature_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute relative contribution of each modality.
        
        Uses L2 norm ratio as a proxy for feature importance.
        """
        visual_norm = np.linalg.norm(feature_data['visual_features'])
        metadata_norm = np.linalg.norm(feature_data['metadata_features'])
        total_norm = visual_norm + metadata_norm + 1e-8
        
        return {
            'image_contribution': float(visual_norm / total_norm),
            'metadata_contribution': float(metadata_norm / total_norm),
            'visual_feature_magnitude': float(visual_norm),
            'metadata_feature_magnitude': float(metadata_norm)
        }
    
    def _compute_shap_importance(self, fused_features: np.ndarray) -> Dict[str, Any]:
        """Compute SHAP-based feature importance."""
        if self.shap_explainer is None:
            return {}
        
        try:
            shap_values = self.shap_explainer.shap_values(fused_features)
            
            # Separate SHAP values for image and metadata portions
            image_shap = shap_values[:, :self.image_feature_dim]
            metadata_shap = shap_values[:, self.image_feature_dim:]
            
            # Aggregate importance
            image_importance = np.abs(image_shap).sum()
            metadata_importance = np.abs(metadata_shap).sum()
            total_importance = image_importance + metadata_importance + 1e-8
            
            return {
                'shap_image_importance': float(image_importance / total_importance),
                'shap_metadata_importance': float(metadata_importance / total_importance),
                'shap_values': shap_values.tolist(),
                'top_influential_features': self._get_top_features(shap_values)
            }
        except Exception as e:
            warnings.warn(f"SHAP analysis failed: {e}")
            return {}
    
    def _get_top_features(self, shap_values: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Get top-k most influential features based on SHAP values."""
        abs_shap = np.abs(shap_values).flatten()
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            modality = 'image' if idx < self.image_feature_dim else 'metadata'
            feature_idx = idx if idx < self.image_feature_dim else idx - self.image_feature_dim
            top_features.append({
                'index': int(idx),
                'modality': modality,
                'feature_index': int(feature_idx),
                'shap_value': float(shap_values.flatten()[idx])
            })
        
        return top_features
    
    def _extract_attention_weights(
        self,
        image: torch.Tensor,
        category_idx: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """Extract attention weights if the model uses attention fusion."""
        # Check if model has attention mechanism
        if not hasattr(self.model, 'attention_fusion'):
            return None
        
        try:
            with torch.no_grad():
                image = image.to(self.device)
                category_idx = category_idx.to(self.device)
                numerical_features = numerical_features.to(self.device)
                
                # Get visual and metadata features
                visual_features = self.model.image_encoder(image)
                metadata_features = self.model.metadata_encoder(category_idx, numerical_features)
                
                # Get attention weights from fusion module
                if hasattr(self.model.attention_fusion, 'forward_with_attention'):
                    _, attention_weights = self.model.attention_fusion.forward_with_attention(
                        visual_features, metadata_features
                    )
                    return {
                        'visual_to_metadata': attention_weights.get('v2m', None),
                        'metadata_to_visual': attention_weights.get('m2v', None)
                    }
        except Exception as e:
            warnings.warn(f"Attention extraction failed: {e}")
        
        return None
    
    def _estimate_confidence(
        self,
        feature_data: Dict[str, np.ndarray],
        prediction: float
    ) -> float:
        """
        Estimate prediction confidence.
        
        Uses feature statistics as proxy for confidence.
        Higher feature activation variance = lower confidence.
        """
        fused_features = feature_data['fused_features']
        
        # Feature-based confidence estimation
        feature_std = np.std(fused_features)
        feature_mean = np.mean(np.abs(fused_features))
        
        # Coefficient of variation (lower = more confident)
        cv = feature_std / (feature_mean + 1e-8)
        
        # Convert to confidence score (0-1 range)
        # Lower CV means higher confidence
        confidence = 1.0 / (1.0 + cv)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _classify_error(self, absolute_error: float) -> str:
        """Classify error into categories."""
        if absolute_error <= ERROR_THRESHOLDS['excellent']:
            return 'excellent'
        elif absolute_error <= ERROR_THRESHOLDS['good']:
            return 'good'
        elif absolute_error <= ERROR_THRESHOLDS['acceptable']:
            return 'acceptable'
        else:
            return 'poor'
    
    def analyze_batch(
        self,
        images: torch.Tensor,
        category_indices: torch.Tensor,
        numerical_features: torch.Tensor,
        predicted_weights: np.ndarray,
        actual_weights: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a batch of predictions.
        
        Args:
            images: Batch of images (B, 3, H, W)
            category_indices: Batch of category indices (B,)
            numerical_features: Batch of numerical features (B, N)
            predicted_weights: Array of predictions in kg
            actual_weights: Array of actual weights in kg (optional)
        
        Returns:
            List of analysis dictionaries for each sample
        """
        batch_size = images.shape[0]
        results = []
        
        for i in range(batch_size):
            actual = actual_weights[i] if actual_weights is not None else None
            result = self.analyze(
                image=images[i:i+1],
                category_idx=category_indices[i:i+1],
                numerical_features=numerical_features[i:i+1],
                predicted_weight=float(predicted_weights[i]),
                actual_weight=float(actual) if actual is not None else None
            )
            results.append(result)
        
        return results
    
    def get_summary_statistics(self, analysis_results: List[Dict]) -> Dict[str, float]:
        """Compute summary statistics from batch analysis."""
        if not analysis_results:
            return {}
        
        # Aggregate metrics
        abs_errors = [r.get('absolute_error', 0) for r in analysis_results if 'absolute_error' in r]
        pct_errors = [r.get('percentage_error', 0) for r in analysis_results if 'percentage_error' in r]
        img_contribs = [r.get('image_contribution', 0) for r in analysis_results]
        meta_contribs = [r.get('metadata_contribution', 0) for r in analysis_results]
        confidences = [r.get('confidence_score', 0) for r in analysis_results if 'confidence_score' in r]
        
        summary = {}
        
        if abs_errors:
            summary['mean_absolute_error'] = float(np.mean(abs_errors))
            summary['median_absolute_error'] = float(np.median(abs_errors))
            summary['std_absolute_error'] = float(np.std(abs_errors))
        
        if pct_errors:
            summary['mean_percentage_error'] = float(np.mean(pct_errors))
        
        summary['mean_image_contribution'] = float(np.mean(img_contribs))
        summary['mean_metadata_contribution'] = float(np.mean(meta_contribs))
        
        if confidences:
            summary['mean_confidence'] = float(np.mean(confidences))
        
        # Error distribution
        error_categories = [r.get('error_category', 'unknown') for r in analysis_results]
        for cat in ['excellent', 'good', 'acceptable', 'poor']:
            summary[f'pct_{cat}'] = error_categories.count(cat) / len(error_categories) * 100
        
        return summary
