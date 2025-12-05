"""
Loss Functions Module for Weight Prediction
Separated from training logic for better organization and testing.

Supports 8 different loss functions optimized for different scenarios.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class WeightPredictionLoss:
    """
    Comprehensive loss function handler for weight prediction tasks.
    
    Optimized for:
    - Wide weight ranges (50kg - 3450kg)
    - Data with outliers
    - Non-well-behaved distributions
    
    Supported Loss Functions:
    1. MSLE (Mean Squared Log Error) - Best for wide ranges
    2. Huber Loss - Best for outliers
    3. MAE (Mean Absolute Error) - Most robust
    4. MSE (Mean Squared Error) - Classic
    5. Smooth L1 - Huber variant
    6. MAPE (Mean Absolute Percentage Error) - Percentage-based
    7. Quantile Loss - Uncertainty estimation
    8. Combined (MSE + MAE) - Hybrid approach
    9. Adaptive - Combines MSLE + MAE (recommended for weight prediction)
    """
    
    def __init__(
        self,
        loss_type: str = 'msle',
        huber_delta: float = 10.0,
        quantile_alpha: float = 0.5,
        combined_mse_weight: float = 0.7,
        combined_mae_weight: float = 0.3,
        epsilon: float = 1e-8,
        adaptive_msle_weight: float = 0.7,
        adaptive_mae_weight: float = 0.3
    ):
        """
        Initialize loss function handler.
        
        Args:
            loss_type: Type of loss function
            huber_delta: Delta parameter for Huber loss
            quantile_alpha: Alpha parameter for Quantile loss (0.5 = median)
            combined_mse_weight: Weight for MSE in combined loss
            combined_mae_weight: Weight for MAE in combined loss
            epsilon: Small constant to avoid division by zero
        """
        self.loss_type = loss_type.lower()
        self.huber_delta = huber_delta
        self.quantile_alpha = quantile_alpha
        self.combined_mse_weight = combined_mse_weight
        self.combined_mae_weight = combined_mae_weight
        self.epsilon = epsilon
        self.adaptive_msle_weight = adaptive_msle_weight
        self.adaptive_mae_weight = adaptive_mae_weight
        
        # Get the loss function
        self.criterion = self._get_loss_function()
        
        # Store info for logging
        self.info = self._get_loss_info()
    
    def _get_loss_function(self) -> Callable:
        """Get the appropriate loss function."""
        
        if self.loss_type == 'msle':
            return self._msle_loss
        elif self.loss_type == 'huber':
            return nn.HuberLoss(delta=self.huber_delta)
        elif self.loss_type == 'mae' or self.loss_type == 'l1':
            return nn.L1Loss()
        elif self.loss_type == 'mse' or self.loss_type == 'l2':
            return nn.MSELoss()
        elif self.loss_type == 'smooth_l1':
            return nn.SmoothL1Loss()
        elif self.loss_type == 'mape':
            return self._mape_loss
        elif self.loss_type == 'quantile':
            return self._quantile_loss
        elif self.loss_type == 'combined':
            return self._combined_loss
        elif self.loss_type == 'log_cosh':
            return self._log_cosh_loss
        elif self.loss_type == 'weighted_mae':
            return self._weighted_mae_loss
        elif self.loss_type == 'adaptive':
            return self._adaptive_loss
        else:
            available = ['msle', 'huber', 'mae', 'mse', 'smooth_l1', 'mape', 'quantile', 'combined', 'log_cosh', 'weighted_mae', 'adaptive']
            raise ValueError(
                f"Unknown loss type: '{self.loss_type}'. "
                f"Available: {', '.join(available)}"
            )
    
    def _get_loss_info(self) -> dict:
        """Get information about the current loss function."""
        
        loss_info = {
            'msle': {
                'name': 'Mean Squared Log Error',
                'best_for': 'Wide weight ranges (20-1500kg)',
                'robust_to_outliers': 'High',
                'scale_dependent': 'No',
                'parameters': {}
            },
            'huber': {
                'name': 'Huber Loss',
                'best_for': 'Data with outliers',
                'robust_to_outliers': 'High',
                'scale_dependent': 'Yes',
                'parameters': {'delta': self.huber_delta}
            },
            'mae': {
                'name': 'Mean Absolute Error (L1)',
                'best_for': 'Very robust to outliers',
                'robust_to_outliers': 'Very High',
                'scale_dependent': 'Yes',
                'parameters': {}
            },
            'mse': {
                'name': 'Mean Squared Error (L2)',
                'best_for': 'Well-behaved data without outliers',
                'robust_to_outliers': 'Low',
                'scale_dependent': 'Yes',
                'parameters': {}
            },
            'smooth_l1': {
                'name': 'Smooth L1 Loss',
                'best_for': 'Quick prototyping',
                'robust_to_outliers': 'Medium',
                'scale_dependent': 'Yes',
                'parameters': {}
            },
            'mape': {
                'name': 'Mean Absolute Percentage Error',
                'best_for': 'Percentage-based accuracy',
                'robust_to_outliers': 'Medium',
                'scale_dependent': 'No',
                'parameters': {}
            },
            'quantile': {
                'name': 'Quantile Loss',
                'best_for': 'Uncertainty estimation',
                'robust_to_outliers': 'Medium',
                'scale_dependent': 'Yes',
                'parameters': {'alpha': self.quantile_alpha}
            },
            'combined': {
                'name': 'Combined MSE + MAE',
                'best_for': 'Balanced approach',
                'robust_to_outliers': 'Medium',
                'scale_dependent': 'Yes',
                'parameters': {
                    'mse_weight': self.combined_mse_weight,
                    'mae_weight': self.combined_mae_weight
                }
            },
            'log_cosh': {
                'name': 'Log-Cosh Loss',
                'best_for': 'Smoother than MSE, more robust',
                'robust_to_outliers': 'Medium-High',
                'scale_dependent': 'Yes',
                'parameters': {}
            },
            'weighted_mae': {
                'name': 'Weighted MAE (by target weight)',
                'best_for': 'When small weights need more attention',
                'robust_to_outliers': 'High',
                'scale_dependent': 'No',
                'parameters': {}
            }
        }
        
        return loss_info.get(self.loss_type, {})
    
    # ========================================================================
    # CUSTOM LOSS FUNCTIONS
    # ========================================================================
    
    def _msle_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Log Error.
        
        Best for wide weight ranges (20-1500kg).
        Naturally handles exponential distributions.
        """
        # Ensure positive values (weights should already be positive)
        predictions = torch.clamp(predictions, min=0.0)
        targets = torch.clamp(targets, min=0.0)
        
        # Compute MSLE
        log_pred = torch.log1p(predictions)  # log(1 + x) for numerical stability
        log_target = torch.log1p(targets)
        
        return torch.mean((log_pred - log_target) ** 2)
    
    def _mape_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Mean Absolute Percentage Error.
        
        Best for percentage-based accuracy evaluation.
        """
        # Avoid division by zero
        return torch.mean(torch.abs((targets - predictions) / (targets + self.epsilon))) * 100
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Quantile Loss for uncertainty estimation.
        
        alpha = 0.5: median prediction
        alpha > 0.5: conservative (over-prediction penalized less)
        alpha < 0.5: optimistic (under-prediction penalized less)
        """
        errors = targets - predictions
        return torch.mean(torch.maximum(
            self.quantile_alpha * errors,
            (self.quantile_alpha - 1) * errors
        ))
    
    def _combined_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Combined MSE + MAE loss.
        
        Balances smooth gradients (MSE) with robustness (MAE).
        """
        mse = torch.mean((predictions - targets) ** 2)
        mae = torch.mean(torch.abs(predictions - targets))
        
        return self.combined_mse_weight * mse + self.combined_mae_weight * mae
    
    def _log_cosh_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Log-Cosh Loss.
        
        Smoother than MSE, more robust to outliers.
        log(cosh(x)) ≈ (x^2)/2 for small x, ≈ |x| for large x
        """
        diff = predictions - targets
        return torch.mean(torch.log(torch.cosh(diff + self.epsilon)))
    
    def _weighted_mae_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Weighted MAE - weights inversely proportional to target weight.
        
        Gives more importance to accurate prediction of lighter objects.
        """
        weights = 1.0 / (targets + self.epsilon)
        weights = weights / weights.mean()  # Normalize
        
        return torch.mean(weights * torch.abs(predictions - targets))
    
    def _adaptive_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Loss combining MSLE + MAE.
        
        MSLE handles the scale-invariance for wide weight ranges,
        while MAE provides robustness to outliers and direct error minimization.
        
        Best for weight prediction with wide ranges (50-3450kg).
        """
        # MSLE component (scale-invariant)
        predictions_clamped = torch.clamp(predictions, min=0.0)
        targets_clamped = torch.clamp(targets, min=0.0)
        log_pred = torch.log1p(predictions_clamped)
        log_target = torch.log1p(targets_clamped)
        msle = torch.mean((log_pred - log_target) ** 2)
        
        # MAE component (direct error)
        mae = torch.mean(torch.abs(predictions - targets))
        
        # Normalize MAE to be on similar scale as MSLE
        # For log-transformed values, MAE is typically much larger
        # We normalize by expected log range
        mae_normalized = mae / (torch.log1p(targets.mean()) + self.epsilon)
        
        return self.adaptive_msle_weight * msle + self.adaptive_mae_weight * mae_normalized
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        return self.criterion(predictions, targets)
    
    def print_info(self):
        """Print information about the current loss function."""
        print(f"\n{'='*70}")
        print(f"LOSS FUNCTION: {self.info.get('name', self.loss_type.upper())}")
        print(f"{'='*70}")
        print(f"Best for:            {self.info.get('best_for', 'N/A')}")
        print(f"Outlier robustness:  {self.info.get('robust_to_outliers', 'N/A')}")
        print(f"Scale dependent:     {self.info.get('scale_dependent', 'N/A')}")
        
        if self.info.get('parameters'):
            print(f"\nParameters:")
            for key, value in self.info['parameters'].items():
                print(f"  {key}: {value}")
        
        print(f"{'='*70}\n")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_msle_loss() -> WeightPredictionLoss:
    """
    Create MSLE loss - RECOMMENDED for wide weight ranges (20-1500kg).
    
    Best for:
    - Wide weight ranges
    - Log-normal distributions
    - Scale-independent predictions
    """
    return WeightPredictionLoss(loss_type='msle')


def create_huber_loss(delta: float = 10.0) -> WeightPredictionLoss:
    """
    Create Huber loss - RECOMMENDED for data with outliers.
    
    Args:
        delta: Transition point between MSE and MAE
               - For 20-1500kg range: delta=10-50
               - Smaller delta = more robust to outliers
    
    Best for:
    - Data with outliers
    - Need smooth gradients
    - Flexible robustness control
    """
    return WeightPredictionLoss(loss_type='huber', huber_delta=delta)


def create_mae_loss() -> WeightPredictionLoss:
    """
    Create MAE loss - Most robust to outliers.
    
    Best for:
    - Very noisy data
    - Many outliers
    - Simple interpretation (error in kg)
    """
    return WeightPredictionLoss(loss_type='mae')


def create_combined_loss(
    mse_weight: float = 0.7,
    mae_weight: float = 0.3
) -> WeightPredictionLoss:
    """
    Create combined MSE + MAE loss.
    
    Args:
        mse_weight: Weight for MSE component (default: 0.7)
        mae_weight: Weight for MAE component (default: 0.3)
    
    Best for:
    - Balanced approach
    - When uncertain which loss to use
    """
    return WeightPredictionLoss(
        loss_type='combined',
        combined_mse_weight=mse_weight,
        combined_mae_weight=mae_weight
    )


# ============================================================================
# RECOMMENDATION FUNCTION
# ============================================================================

def recommend_loss_function(
    weight_min: float,
    weight_max: float,
    has_outliers: bool = True,
    outlier_percentage: float = 5.0
) -> WeightPredictionLoss:
    """
    Automatically recommend the best loss function based on data characteristics.
    
    Args:
        weight_min: Minimum weight in dataset (kg)
        weight_max: Maximum weight in dataset (kg)
        has_outliers: Whether dataset has outliers
        outlier_percentage: Percentage of outliers (if known)
    
    Returns:
        WeightPredictionLoss instance with recommended configuration
    """
    weight_range = weight_max - weight_min
    weight_ratio = weight_max / max(weight_min, 0.1)
    
    print(f"\n{'='*70}")
    print("LOSS FUNCTION RECOMMENDATION")
    print(f"{'='*70}")
    print(f"Weight range: {weight_min:.1f}kg - {weight_max:.1f}kg")
    print(f"Range span:   {weight_range:.1f}kg")
    print(f"Ratio:        {weight_ratio:.1f}x")
    print(f"Outliers:     {'Yes' if has_outliers else 'No'} ({outlier_percentage:.1f}%)")
    print(f"{'-'*70}")
    
    # Decision logic
    if weight_ratio > 10 and weight_range > 500:
        # Wide range with high ratio
        print("Recommendation: MSLE (Mean Squared Log Error)")
        print("Reason: Wide weight range with high ratio")
        print("        MSLE handles exponential distributions well")
        loss = create_msle_loss()
    
    elif has_outliers and outlier_percentage > 10:
        # Many outliers
        print("Recommendation: MAE (Mean Absolute Error)")
        print("Reason: High outlier percentage")
        print("        MAE is most robust to outliers")
        loss = create_mae_loss()
    
    elif has_outliers and outlier_percentage > 5:
        # Some outliers
        suggested_delta = weight_range * 0.05
        print(f"Recommendation: Huber Loss (delta={suggested_delta:.1f})")
        print("Reason: Moderate outliers present")
        print("        Huber balances robustness and smooth gradients")
        loss = create_huber_loss(delta=suggested_delta)
    
    elif weight_range > 500:
        # Wide range but clean data
        print("Recommendation: MSLE (Mean Squared Log Error)")
        print("Reason: Wide weight range")
        print("        MSLE handles scale differences")
        loss = create_msle_loss()
    
    else:
        # Clean, narrow range
        print("Recommendation: Huber Loss (delta=1.0)")
        print("Reason: Relatively clean data with moderate range")
        print("        Huber provides good default performance")
        loss = create_huber_loss(delta=1.0)
    
    print(f"{'='*70}\n")
    
    return loss


# ============================================================================
# TESTING AND COMPARISON
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LOSS FUNCTIONS MODULE - TEST")
    print("="*70)
    
    # Create dummy data
    batch_size = 10
    predictions = torch.tensor([50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 100.0, 300.0])
    targets = torch.tensor([45.0, 95.0, 260.0, 510.0, 740.0, 1050.0, 1200.0, 1480.0, 120.0, 280.0])
    
    print(f"\nTest data:")
    print(f"Predictions: {predictions.tolist()}")
    print(f"Targets:     {targets.tolist()}")
    
    # Test all loss functions
    print(f"\n{'Loss Type':<20} {'Loss Value':<15} {'Description'}")
    print("-" * 70)
    
    loss_types = [
        ('msle', {}, 'For wide ranges'),
        ('huber', {'huber_delta': 10.0}, 'For outliers'),
        ('mae', {}, 'Most robust'),
        ('mse', {}, 'Classic'),
        ('smooth_l1', {}, 'Huber variant'),
        ('mape', {}, 'Percentage'),
        ('combined', {}, 'MSE + MAE'),
        ('log_cosh', {}, 'Smooth & robust'),
    ]
    
    for loss_type, kwargs, description in loss_types:
        loss_fn = WeightPredictionLoss(loss_type=loss_type, **kwargs)
        loss_value = loss_fn(predictions, targets).item()
        print(f"{loss_type:<20} {loss_value:<15.6f} {description}")
    
    # Test recommendation system
    print("\n" + "="*70)
    print("TESTING RECOMMENDATION SYSTEM")
    print("="*70)
    
    # Test case: Your data (20-1500kg with outliers)
    recommended_loss = recommend_loss_function(
        weight_min=20.0,
        weight_max=1500.0,
        has_outliers=True,
        outlier_percentage=8.0
    )
    
    recommended_loss.print_info()
    
    # Compute loss with recommended function
    loss_value = recommended_loss(predictions, targets)
    print(f"Loss value with recommended function: {loss_value.item():.6f}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70 + "\n")
