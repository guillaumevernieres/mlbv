#!/bin/bash
# Setup script for IceNet Python training environment

echo "Setting up IceNet Python training environment..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please run this script from the python_training directory."
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating output directories..."
mkdir -p data
mkdir -p models
mkdir -p checkpoints

echo "Setup complete!"
echo ""
echo "To get started:"
echo "  python scripts/train.py --config configs/config.yaml --create-data"
echo ""
echo "This will create sample training data and start training the model."
