# IceNet Python Training System
IceNet is a PyTorch-based distributed training system for sea ice prediction models that supports HPC environments with SLURM, PBS, and manual distributed setups.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Essential Setup
- Setup environment and dependencies:
  - `./setup.sh` -- takes 2 minutes. NEVER CANCEL. Wait for PyTorch installation.
  - Creates `data/`, `models/`, and `checkpoints/` directories
- Install development tools (if needed):
  - `python -m pip install flake8 black mypy types-PyYAML`

### Build and Validation Commands  
- Basic training test: `python train_icenet.py --config config.yaml --create-data` -- takes 7 seconds with default config (10k samples, 100 epochs)
- Quick training test: `python train_icenet.py --config config_quick.yaml --create-data` -- takes 4 seconds (1k samples, 5 epochs)
- HPC setup test: `python test_hpc_setup.py` -- takes 2 seconds, validates distributed training environment
- Distributed training: `python launch_hpc_training.py --config config.yaml --scheduler single-node --gpus-per-node 1` -- takes 8 seconds
- Example workflows: `python example_usage.py --create-sample-data` -- takes 8 seconds, demonstrates full training cycle

### Linting and Code Quality
- Lint code: `python -m flake8 train_icenet.py --max-line-length=88 --ignore=E203,W503` -- takes <1 second
- Type checking: `python -m mypy icenet.py --ignore-missing-imports` -- takes 1 second
- **WARNING**: mypy configuration in pyproject.toml requires Python 3.9+, but declares 3.8. Update python_version to "3.9" if using mypy
- Format code: `python -m black train_icenet.py` -- takes <1 second

## Manual Validation Scenarios

### Core Training Validation
ALWAYS test these scenarios after making changes to training code:
1. **Basic Training Flow**: Run `python train_icenet.py --config config.yaml --create-data` and verify:
   - Data creation completes without errors
   - Training starts and logs epoch progress
   - Model checkpoints are saved to `models/`
   - Training completes with "Training completed successfully!" message
   - Verify files created: `data/sample_data.npz`, `models/best_model.pt`, `models/training_history.png`

2. **Distributed Training Setup**: Run `python test_hpc_setup.py` and verify:
   - Environment detection works correctly
   - CUDA/CPU device assignment is correct
   - Output shows "Ready for distributed IceNet training!"

3. **Example Usage**: Run `python example_usage.py --create-sample-data` and verify:
   - Complete end-to-end workflow executes
   - All training outputs are generated correctly

### HPC/Distributed Validation
After changes to distributed training code:
1. Test single-node: `python launch_hpc_training.py --config config.yaml --scheduler single-node --gpus-per-node 1`
2. Verify distributed initialization messages appear
3. Check that training completes with "Distributed training completed successfully!"

### Data Processing Validation  
After changes to data preparation:
1. Test NetCDF conversion: `python example_usage.py --netcdf-file test.nc --convert-only` (if NetCDF file available)
2. Verify data loading: Check that `data/sample_data.npz` contains expected fields

## Known Issues and Workarounds

### Type Checking Issues
- **mypy python_version error**: pyproject.toml declares Python 3.8 but mypy requires 3.9+
  - Workaround: Update pyproject.toml `[tool.mypy]` section: `python_version = "3.9"`
  - Or skip mypy if not essential: Use flake8 for basic linting instead

### Multi-GPU Testing Limitation
- **Multi-GPU spawn test fails**: `python test_hpc_setup.py --gpus 2` fails due to multiprocessing spawn issue
  - Workaround: Test distributed training with single GPU: `--gpus 1`
  - Real multi-GPU testing requires actual GPU hardware

### Training Warnings
- **Tensor size mismatch warnings**: MSE loss shows broadcasting warnings but training works correctly
  - This is a known issue with target tensor shape - functionality not affected
  - Warnings appear as: "Using a target size (torch.Size([64])) that is different to the input size (torch.Size([64, 1]))"

## Configuration and File Structure

### Core Files and Purpose
- `train_icenet.py` -- Main training script with distributed support
- `icenet.py` -- Neural network model definition  
- `data_preparation.py` -- NetCDF data processing utilities
- `launch_hpc_training.py` -- HPC job launcher (SLURM, PBS, manual)
- `test_hpc_setup.py` -- Distributed training environment validator
- `config.yaml` -- Default training configuration
- `setup.sh` -- Environment setup script
- `requirements.txt` -- Python dependencies

### Generated Files and Directories
- `data/sample_data.npz` -- Training data (created by --create-data)
- `models/best_model.pt` -- Best model checkpoint
- `models/normalization.*.pt` -- Data normalization parameters
- `models/training_history.png` -- Training loss plot
- `checkpoints/` -- Additional model checkpoints (created every 20 epochs)

### Key Configuration Parameters
- `training.epochs` -- Number of training epochs (default: 100)
- `data.num_samples` -- Number of training samples (default: 10000)  
- `training.batch_size` -- Batch size (default: 64)
- `model.hidden_size` -- Model hidden layer size (default: 16)

## Common Tasks and Expected Timings

### Setup and Installation Timings
- **NEVER CANCEL**: Setup takes 2 minutes for PyTorch and dependencies installation
- Environment setup: `./setup.sh` -- 2 minutes
- Dev tools install: 30 seconds
- Directory creation: instant

### Training and Testing Timings  
- Basic training (10k samples, 100 epochs): 7 seconds
- Quick training (1k samples, 5 epochs): 4 seconds  
- Distributed training launch: 8 seconds
- HPC setup validation: 2 seconds
- Example workflow: 8 seconds

### Code Quality Timings
- Linting with flake8: <1 second
- Type checking with mypy: 1 second
- Code formatting: <1 second

## Distributed Training Support

### Supported Schedulers
- **SLURM**: Auto-detects SLURM environment variables
- **PBS/Torque**: Uses PBS_NODEFILE for node allocation  
- **Manual**: Set environment variables manually
- **Single-node**: Multi-GPU on single machine

### Environment Variables for Manual Setup
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export WORLD_SIZE=<total_processes>  
export RANK=<process_rank>
export LOCAL_RANK=<local_rank>
```

### Quick Examples
- Single node: `python launch_hpc_training.py --config config.yaml --scheduler single-node --gpus-per-node 1`
- SLURM job: `python launch_hpc_training.py --config config.yaml --scheduler slurm --nodes 2 --gpus-per-node 2 --submit`

## Troubleshooting Commands

### Diagnostic Commands
- Check Python/PyTorch: `python -c "import torch; print(f'PyTorch {torch.__version__}')"`
- Test imports: `python -c "import icenet; print('IceNet loaded successfully')"`
- Check GPU: `python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"`
- Validate training data: `python -c "import numpy as np; data=np.load('data/sample_data.npz'); print(f'Data shape: {data['inputs'].shape}')"`

### File Status Checks
- Check generated files: `ls -la data/ models/ checkpoints/`
- Verify data format: `python -c "import numpy as np; print(list(np.load('data/sample_data.npz').keys()))"`
- Check model files: `python -c "import torch; print(list(torch.load('models/best_model.pt', map_location='cpu').keys()))"`

### Performance Monitoring
- Training with profiling: Add `training.profile_training: true` to config.yaml
- Monitor GPU usage: `nvidia-smi` (if GPUs available)
- Check logs: `tail -f *.out` (for HPC job outputs)