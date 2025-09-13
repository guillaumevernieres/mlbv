# IceNet Directory Structure

## New Organized Structure

The repository has been reorganized with a much cleaner and more professional structure:

```
icenet-training/
├── icenet/                      # 📦 Main Python package
│   ├── __init__.py             # Package initialization and exports
│   ├── model.py                # 🧠 Neural network model (IceNet class)
│   ├── training.py             # 🎯 Main training logic and CLI
│   ├── data.py                 # 📊 Data preparation and NetCDF processing
│   └── distributed.py         # 🔗 Distributed training utilities
├── scripts/                    # 🔧 Executable scripts and utilities
│   ├── train.py               # 🚀 Main CLI entry point for training
│   ├── launch_hpc_training.py # 🖥️  HPC cluster job launcher
│   ├── setup.sh              # ⚙️  Environment setup
│   └── setup_repository.sh   # 📦 Repository initialization
├── configs/                   # ⚙️  Configuration files
│   └── config.yaml           # 📋 Default training configuration
├── tests/                     # 🧪 Test modules
│   ├── __init__.py           # Test package init
│   └── test_hpc_setup.py     # HPC setup validation tests
├── examples/                  # 📚 Usage examples and demos
│   └── example_usage.py      # Complete usage examples
├── docs/                      # 📖 Documentation
│   ├── DISTRIBUTED_TRAINING.md # Distributed training guide
│   ├── HPC_DEPLOYMENT.md     # HPC deployment instructions
│   └── MIGRATION_GUIDE.md    # Migration and setup guide
├── .gitignore                # Git ignore patterns
├── Makefile                  # Common development tasks
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Modern Python project configuration
├── CHANGELOG.md             # Version history
└── README.md                # Main project documentation
```

## Key Improvements

### 🎯 **Clean Package Structure**
- All core functionality organized in the `icenet/` package
- Clear separation of model, training, data processing, and distributed computing
- Proper Python package with `__init__.py` and clean imports

### 🔧 **Organized Scripts**
- Executable scripts moved to `scripts/` directory
- Clear CLI entry point (`scripts/train.py`)
- HPC utilities properly organized

### ⚙️ **Separated Configurations**
- Configuration files in dedicated `configs/` directory
- Easy to manage different configurations for different environments

### 📚 **Documentation Structure**
- All documentation in `docs/` directory
- Clear separation of user guides, deployment instructions, and migration info

### 🧪 **Proper Testing**
- Test modules in dedicated `tests/` directory
- Ready for pytest and other testing frameworks

### 🛠️ **Development Tools**
- Makefile for common development tasks
- Proper .gitignore for Python projects
- Modern pyproject.toml configuration

## Usage with New Structure

### Quick Start
```bash
# Install and setup
pip install -e .
make install

# Run training
python scripts/train.py --create-data --config configs/config.yaml
# or
make train

# Run tests
python tests/test_hpc_setup.py
# or
make test
```

### Import in Code
```python
# Clean imports from the organized package
from icenet.model import IceNet, create_icenet
from icenet.training import IceNetTrainer, load_config
from icenet.data import IceDataPreparer
```

### HPC Deployment
```bash
# Launch distributed training
python scripts/launch_hpc_training.py --config configs/config.yaml --nodes 4
```

This new structure follows Python best practices and makes the project much more maintainable and professional!
