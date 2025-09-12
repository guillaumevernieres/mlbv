"""
Test configuration for pytest
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

# Set up test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'input_size': 7,
            'hidden_size': 16,
            'output_size': 1
        },
        'training': {
            'epochs': 2,
            'batch_size': 32,
            'optimizer': {
                'type': 'adam',
                'learning_rate': 0.001
            },
            'scheduler': {
                'type': 'step',
                'step_size': 1,
                'gamma': 0.9
            },
            'early_stopping_patience': 3
        },
        'data': {
            'data_path': 'test_data.npz',
            'validation_split': 0.2,
            'num_workers': 0  # Disable multiprocessing for tests
        },
        'distributed': {
            'backend': 'nccl'
        }
    }

@pytest.fixture
def sample_data():
    """Generate sample training data."""
    torch.manual_seed(42)  # For reproducible tests
    n_samples = 100
    input_size = 7

    # Generate random input data
    X = torch.randn(n_samples, input_size)

    # Generate simple target (sum of inputs with some noise)
    y = torch.sigmoid(X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1))

    return X, y

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Disable GPU for tests by default
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Set torch to use deterministic algorithms
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Configure torch backends for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
