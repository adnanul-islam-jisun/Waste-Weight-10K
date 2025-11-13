"""
Hyperparameters (DEPRECATED - Use config.py instead)

This file is kept for backward compatibility.
All hyperparameters are now in config.py for unified configuration.
"""

# Import from unified config
from config.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    EPOCHS,
    DROPOUT_RATE,
    USE_LR_SCHEDULER,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM
)

# Legacy aliases for backward compatibility
LR_PATIENCE = LR_SCHEDULER_PATIENCE
LR_FACTOR = LR_SCHEDULER_FACTOR
MIN_DELTA = 0.001

print("⚠️  WARNING: hyperparameters.py is deprecated. Use config.py instead.")

