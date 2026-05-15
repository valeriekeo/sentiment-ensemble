#!/bin/bash
# setup_env.sh — Project 1: Multi-Model Sentiment Ensemble
# Creates and configures the conda environment. Safe to rerun —
# skips environment creation if it already exists.
# Usage: bash setup_env.sh

set -e  # exit immediately on error

ENV_NAME="sentiment-ensemble"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists — skipping creation."
else
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.11 -y
fi

echo "Activating environment..."
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA 12.1 (compatible with CUDA 13.x drivers)..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing remaining dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

echo "Downloading NLTK VADER lexicon..."
python -c "import nltk; nltk.download('vader_lexicon')"

echo ""
echo "Verifying GPU access..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "Done. Activate your environment with:"
echo "  conda activate $ENV_NAME"
