#!/bin/bash
# Script to download model checkpoints from cloud storage
# 
# Usage:
#   ./scripts/download_checkpoints.sh
#
# This script downloads pre-trained model checkpoints to the checkpoints/ directory.
# Update the DOWNLOAD_URL variable with your actual cloud storage link.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"

# Create checkpoints directory if it doesn't exist
mkdir -p "$CHECKPOINTS_DIR"

echo "📥 Downloading model checkpoints..."
echo ""

# Option 1: Direct download from URL (Google Drive, Dropbox, etc.)
# Replace with your actual download URL
DOWNLOAD_URL="https://your-cloud-storage-link.com/checkpoints.zip"

# Option 2: Using gdown for Google Drive (if using Google Drive)
# Install: pip install gdown
# gdown --folder "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" -O "$CHECKPOINTS_DIR"

# Option 3: Using wget/curl
if command -v wget &> /dev/null; then
    echo "Using wget..."
    wget -O "$CHECKPOINTS_DIR/checkpoints.zip" "$DOWNLOAD_URL"
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -L -o "$CHECKPOINTS_DIR/checkpoints.zip" "$DOWNLOAD_URL"
else
    echo "❌ Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract if it's a zip file
if [ -f "$CHECKPOINTS_DIR/checkpoints.zip" ]; then
    echo "📦 Extracting checkpoints..."
    unzip -q "$CHECKPOINTS_DIR/checkpoints.zip" -d "$CHECKPOINTS_DIR"
    rm "$CHECKPOINTS_DIR/checkpoints.zip"
    echo "✅ Checkpoints extracted successfully!"
fi

# Verify checkpoints
echo ""
echo "📋 Verifying checkpoints..."
CHECKPOINT_COUNT=$(find "$CHECKPOINTS_DIR" -name "*.pth" | wc -l)
if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
    echo "✅ Found $CHECKPOINT_COUNT checkpoint file(s):"
    find "$CHECKPOINTS_DIR" -name "*.pth" -exec basename {} \;
else
    echo "⚠️  No checkpoint files found. Please check the download URL."
fi

echo ""
echo "✨ Done! Checkpoints are ready in $CHECKPOINTS_DIR"

