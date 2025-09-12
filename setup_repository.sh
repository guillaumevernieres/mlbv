#!/bin/bash

# Repository setup script for IceNet Python Training
# Creates a new git repository with proper structure

set -e

echo "Setting up IceNet Python Training Repository..."

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "Git repository initialized."
fi

# Add all files to git
echo "Adding files to repository..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: IceNet Python Training System

- Complete PyTorch implementation of IceNet model
- Distributed training with HPC support (SLURM, PBS)
- NetCDF data processing and preparation
- YAML configuration management
- Comprehensive documentation and examples
- Mathematical equivalence to C++ MPI implementation"

echo "Repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Set up remote repository:"
echo "   git remote add origin <your-repository-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "2. Install the package:"
echo "   pip install -e ."
echo ""
echo "3. Test the installation:"
echo "   python -c 'import icenet_training; print(icenet_training.__version__)'"
echo ""
echo "4. Run tests:"
echo "   python test_hpc_setup.py"
echo "   python train_icenet.py --create-data"
