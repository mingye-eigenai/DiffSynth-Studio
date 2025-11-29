#!/bin/bash

# Helper script to migrate models to the new base_models structure

set -e

OLD_MODEL_PATH="/shared/mingye/DiffSynth-Studio/examples/qwen_image/model_training/lora/models/Qwen"
NEW_BASE_PATH="/shared/base_models/Qwen"

echo "=== Setting up base models with two-level hierarchy ==="

# Create base models directory
mkdir -p /shared/base_models/Qwen

# Check if old models exist
if [ -d "${OLD_MODEL_PATH}" ]; then
    echo "Found models at old location: ${OLD_MODEL_PATH}"
    echo "Copying to new location: ${NEW_BASE_PATH}"

    # Copy each model subdirectory
    if [ -d "${OLD_MODEL_PATH}/Qwen-Image-Edit" ]; then
        echo "Copying Qwen-Image-Edit..."
        mkdir -p "${NEW_BASE_PATH}/Qwen-Image-Edit"
        cp -r "${OLD_MODEL_PATH}/Qwen-Image-Edit"/* "${NEW_BASE_PATH}/Qwen-Image-Edit/"
    fi

    if [ -d "${OLD_MODEL_PATH}/Qwen-Image" ]; then
        echo "Copying Qwen-Image..."
        mkdir -p "${NEW_BASE_PATH}/Qwen-Image"
        cp -r "${OLD_MODEL_PATH}/Qwen-Image"/* "${NEW_BASE_PATH}/Qwen-Image/"
    fi

    echo "Models copied successfully!"
    echo ""
    echo "New structure:"
    echo "/shared/base_models/Qwen/"
    ls -la "${NEW_BASE_PATH}"
else
    echo "No models found at old location: ${OLD_MODEL_PATH}"
    echo "Models will be downloaded automatically when training starts."
fi

echo ""
echo "=== Setup complete ==="
echo "Base models path: /shared/base_models/"
echo ""
echo "Directory structure:"
echo "/shared/base_models/"
echo "└── Qwen/"
echo "    ├── Qwen-Image-Edit/      # Default edit model"
echo "    ├── Qwen-Image-Edit-2509/ # Add alternative models here"
echo "    └── Qwen-Image/           # Shared components"
