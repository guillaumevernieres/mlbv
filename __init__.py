"""
IceNet Python Training Package

A comprehensive PyTorch-based training system for the IceNet neural network model,
designed for sea ice concentration prediction with distributed training capabilities.

Components:
- icenet.py: Neural network model implementation
- train_icenet.py: Main training application with distributed support
- data_preparation.py: NetCDF data processing utilities
- launch_hpc_training.py: HPC cluster launcher
- distributed_train.py: Distributed training utilities

(C) Copyright 2024 NOAA/NWS/NCEP/EMC
This software is licensed under the terms of the Apache Licence Version 2.0
which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""

from .icenet import IceNet, create_icenet
from .train_icenet import IceNetTrainer, load_config, create_default_config
from .data_preparation import IceDataPreparer, create_training_data_from_netcdf

__version__ = "1.0.0"
__author__ = "NOAA/NWS/NCEP/EMC"
__email__ = "your-email@noaa.gov"
__license__ = "Apache-2.0"

__all__ = [
    "IceNet",
    "create_icenet",
    "IceNetTrainer",
    "IceDataPreparer",
    "load_config",
    "create_default_config",
    "create_training_data_from_netcdf",
]
