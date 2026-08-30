# Use NVIDIA PyTorch image as base (includes CUDA, cuDNN, Python, PyTorch)
# Check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags for latest tags
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set metadata
LABEL maintainer="Weight Management AI Team"
LABEL description="Docker image for Weight Prediction Model Training and Inference with GPU support"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
# The NVIDIA image is very comprehensive, but we can add more if needed.
# build-essential is usually included, but good to ensure.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: The base image already has PyTorch, torchvision, etc. installed.
# We only install what's in requirements.txt. 
# It's good practice to upgrade pip first.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Create directories for artifacts
RUN mkdir -p checkpoints logs saved_models

# Expose ports if needed (e.g., for API or Jupyter)
# EXPOSE 8000

# Default command
CMD ["python", "scripts/train.py"]
