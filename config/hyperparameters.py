# Model Hyperparameters

# Training hyperparameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Learning rate scheduler
LR_SCHEDULER = "ReduceLROnPlateau"
LR_PATIENCE = 5
LR_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001

# Data augmentation
AUGMENTATION_ENABLED = True
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False

# Model architecture
DROPOUT_RATE = 0.5
NUM_CLASSES = None  # Will be set dynamically based on dataset

# Ensemble settings
ENSEMBLE_MODELS = ["resnet", "efficientnet", "vit"]
ENSEMBLE_WEIGHTS = [0.4, 0.4, 0.2]
