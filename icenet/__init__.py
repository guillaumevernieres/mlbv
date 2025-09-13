"""
IceNet Python Training Package

A comprehensive PyTorch-based training system for the IceNet neural network
model, designed for sea ice concentration prediction with distributed training
capabilities.

Components:
- model.py: Neural network model implementation
- training.py: Main training application with distributed support
- data.py: NetCDF data processing utilities
- distributed.py: Distributed training utilities
"""

from .model import IceNet, create_icenet
from .training import IceNetTrainer, load_config, create_default_config
from .data import IceDataPreparer, create_training_data_from_netcdf

__version__ = "1.0.0"

__all__ = [
    "IceNet",
    "create_icenet",
    "IceNetTrainer",
    "IceDataPreparer",
    "load_config",
    "create_default_config",
    "create_training_data_from_netcdf",
]
