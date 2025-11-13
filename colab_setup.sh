#!/bin/bash
# Quick Colab Setup - Copy dataset to local SSD

echo "================================================================================"
echo "COLAB FAST STORAGE SETUP"
echo "================================================================================"

# Check if running in Colab
if [ ! -d "/content" ]; then
    echo "❌ This script is for Google Colab only!"
    exit 1
fi

# Source and destination
DRIVE_PATH="/content/drive/MyDrive/KaggleData/disaster/waste_dataset"
LOCAL_PATH="/content/dataset"

# Check if Drive is mounted
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "❌ Google Drive not mounted!"
    echo "   Run this first: from google.colab import drive; drive.mount('/content/drive')"
    exit 1
fi

# Check if source exists
if [ ! -d "$DRIVE_PATH" ]; then
    echo "❌ Dataset not found at: $DRIVE_PATH"
    exit 1
fi

# Check if already copied
if [ -d "$LOCAL_PATH" ] && [ -f "$LOCAL_PATH/image.csv" ]; then
    echo "✓ Dataset already exists at $LOCAL_PATH"
    echo "✓ Skipping copy (delete /content/dataset to force re-copy)"
else
    echo "📦 Copying dataset from Google Drive to local SSD..."
    echo "   This will take 1-5 minutes depending on dataset size..."
    
    # Copy with progress
    rsync -ah --progress "$DRIVE_PATH" /content/ || cp -r "$DRIVE_PATH" "$LOCAL_PATH"
    
    if [ -f "$LOCAL_PATH/image.csv" ]; then
        echo "✅ Dataset copied successfully!"
    else
        echo "❌ Copy failed!"
        exit 1
    fi
fi

# Show paths
echo ""
echo "================================================================================"
echo "UPDATE YOUR config.py WITH:"
echo "================================================================================"
echo "CSV_PATH = '$LOCAL_PATH/image.csv'"
echo "BASE_IMAGE_PATH = '$LOCAL_PATH'"
echo "================================================================================"
echo ""
echo "💡 Expected speedup: 10-20x faster data loading!"
echo "✅ Setup complete! Run: python train.py"
