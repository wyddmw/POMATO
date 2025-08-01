#!/bin/bash

# POMATO Environment Setup Script
# This script sets up the conda environment for POMATO inference

set -e  # Exit on any error

echo "🚀 Setting up POMATO environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed. Please install conda or miniconda first."
    exit 1
fi

# Environment name
ENV_NAME="dust3r"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "ℹ️  Using existing environment. You may need to install additional packages manually."
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

echo "📦 Creating conda environment '${ENV_NAME}'..."
conda create -n ${ENV_NAME} python=3.11 cmake=3.14.0 -y

echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "🚀 Installing PyTorch with CUDA support..."
# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected, installing PyTorch with CUDA 12.1"
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    echo "⚠️  No NVIDIA GPU detected, installing CPU-only PyTorch"
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

echo "📋 Installing POMATO dependencies..."
pip install roma gradio matplotlib tqdm opencv-python scipy einops gdown trimesh pyglet huggingface-hub[torch]>=0.22

echo "📋 Installing optional dependencies..."
# For camera trajectory evaluation
pip install evo

# For tensorboard logging
pip install tensorboard

echo "🔧 Installing additional POMATO-specific dependencies..."
# Add any POMATO-specific packages here
# pip install additional_package

echo "✅ Environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: conda activate ${ENV_NAME}"
echo "2. Download model weights from HuggingFace: https://huggingface.co/wyddmw/POMATO_Pairwise"
echo "3. Run the demo: python demo.py --weights /path/to/weights"
echo ""
echo "🔗 Optional: Compile CUDA kernels for RoPE (faster inference)"
echo "   cd dust3r/croco/models/curope/"
echo "   python setup.py build_ext --inplace"
echo ""
echo "🎉 Happy reconstructing with POMATO!" 