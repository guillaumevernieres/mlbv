"""
Training application for IceNet model
Handles data loading, training, validation, and model saving.
Uses YAML configuration files for easy parameter management.

(C) Copyright 2024 NOAA/NWS/NCEP/EMC
This software is licensed under the terms of the Apache Licence Version 2.0
which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""

import argparse
import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union
from typing import Any, List, Optional

from icenet import create_icenet, IceNet
from data_preparation import IceDataPreparer, create_training_data_from_netcdf


class IceNetTrainer:
    """Training class for IceNet model with distributed training support."""

    def __init__(self, config: Dict, rank: int = 0, world_size: int = 1):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration dictionary
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        # Setup device
        if self.is_distributed:
            # Use LOCAL_RANK for device assignment if available
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            if torch.cuda.is_available():
                # Ensure we don't exceed available GPUs
                local_rank = local_rank % torch.cuda.device_count()
                self.device = torch.device(f'cuda:{local_rank}')
                torch.cuda.set_device(local_rank)
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                'cuda' if (torch.cuda.is_available() and
                           config.get('use_cuda', True))
                else 'cpu'
            )

        if self.rank == 0:
            print(f"Using device: {self.device}")
            if self.is_distributed:
                print(f"Distributed training: {world_size} processes")

        # Initialize model
        self.model: Union[IceNet, DDP] = create_icenet(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            output_size=config['model']['output_size']
        ).to(self.device)

        # Wrap model for distributed training
        if self.is_distributed:
            device_ids = [rank] if torch.cuda.is_available() else None
            self.model = DDP(self.model, device_ids=device_ids)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize loss function
        self.criterion = self._create_loss_function()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config['training']['optimizer']

        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_config['type'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_type = self.config['training'].get('loss_function', 'mse')

        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        scheduler_config = self.config['training'].get('scheduler', None)

        if scheduler_config is None:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1000,
                gamma=1.0
            )  # No-op scheduler

        if scheduler_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        else:
            raise ValueError(
                f"Unknown scheduler type: {scheduler_config['type']}"
            )

    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare training data with distributed support.

        Args:
            data_path: Path to data file (.npz, .pt, or .nc)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.rank == 0:
            print(f"Loading data from: {data_path}")

        # Handle NetCDF files by converting them first
        if data_path.endswith('.nc'):
            if self.rank == 0:
                print("NetCDF file detected, converting to training format...")
                processed_file = str(Path(data_path).with_suffix('.npz'))

                # Use data preparation module
                create_training_data_from_netcdf(
                    data_path,
                    self.config,
                    processed_file
                )

            # Synchronize processes if distributed
            if self.is_distributed:
                dist.barrier()

            data_path = str(Path(data_path).with_suffix('.npz'))

        # Load processed data
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            inputs = torch.FloatTensor(data['inputs'])
            targets = torch.FloatTensor(data['targets'])

            # Use saved normalization stats if available
            if 'input_mean' in data and 'input_std' in data:
                input_mean = torch.tensor(data['input_mean'], dtype=torch.float32)
                input_std = torch.tensor(data['input_std'], dtype=torch.float32)
                if self.rank == 0:
                    print("Using saved normalization statistics")
            else:
                # Compute normalization statistics (distributed if needed)
                input_mean, input_std = self._compute_distributed_stats(inputs)
                if self.rank == 0:
                    print("Computing normalization statistics from data")

        elif data_path.endswith('.pt'):
            data = torch.load(data_path)
            inputs = data['inputs']
            targets = data['targets']

            # Use saved normalization stats if available
            if 'input_mean' in data and 'input_std' in data:
                input_mean = data['input_mean']
                input_std = data['input_std']
                if self.rank == 0:
                    print("Using saved normalization statistics")
            else:
                # Compute normalization statistics (distributed if needed)
                input_mean, input_std = self._compute_distributed_stats(inputs)
                if self.rank == 0:
                    print("Computing normalization statistics from data")
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        if self.rank == 0:
            print(f"Data shape - Inputs: {inputs.shape}, "
                  f"Targets: {targets.shape}")

        # Prevent division by zero
        input_std = torch.where(
            input_std > 1e-6,
            input_std,
            torch.ones_like(input_std)
        )

        # Initialize model normalization
        if hasattr(self.model, 'module'):  # DDP wrapped model
            assert hasattr(self.model.module, 'init_norm')
            self.model.module.init_norm(input_mean, input_std)  # type: ignore
        else:
            assert hasattr(self.model, 'init_norm')
            self.model.init_norm(input_mean, input_std)  # type: ignore

        # Create dataset
        dataset = TensorDataset(inputs, targets)

        # Split into train/validation
        val_size = int(len(dataset) * self.config['data']['validation_split'])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        # Create distributed samplers if needed
        if self.is_distributed:
            train_sampler: Optional[DistributedSampler] = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
            val_sampler: Optional[DistributedSampler] = DistributedSampler(
                val_dataset, num_replicas=self.world_size, rank=self.rank
            )
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.config['data'].get('num_workers', 0)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config['data'].get('num_workers', 0)
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            log_interval = self.config['training'].get('log_interval', 100)
            if batch_idx % log_interval == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("Starting training...")
        start_time = time.time()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training'].get('early_stopping_patience', 10)

        for epoch in range(self.config['training']['epochs']):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            epoch_time = time.time() - epoch_start

            print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]} '
                  f'({epoch_time:.2f}s) - '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

            # Save periodic checkpoint
            save_interval = self.config['training'].get('save_interval', 10)
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.2f} seconds')

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }

        output_dir = Path(self.config['output']['model_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, output_dir / filename)
        
        # Save normalization stats
        if hasattr(self.model, 'module'):  # DDP wrapped model
            self.model.module.save_norm(str(output_dir / filename))  # type: ignore
        else:
            self.model.save_norm(str(output_dir / filename))  # type: ignore
            
        print(f'Saved checkpoint: {output_dir / filename}')

    def plot_training_history(self) -> None:
        """Plot training history."""
        output_dir = Path(self.config['output']['model_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Learning rate plot
        axes[0, 1].plot(self.history['learning_rate'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')

        # Loss plot (log scale)
        axes[1, 0].semilogy(self.history['train_loss'], label='Train Loss')
        axes[1, 0].semilogy(self.history['val_loss'], label='Val Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].set_title('Training and Validation Loss (Log Scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Validation loss zoomed
        if len(self.history['val_loss']) > 10:
            axes[1, 1].plot(self.history['val_loss'][10:])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].set_title('Validation Loss (After Epoch 10)')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(
            output_dir / 'training_history.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    def _compute_distributed_stats(
            self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std statistics across distributed processes.
        Replicates the C++ MPI_Allreduce functionality.

        Args:
            inputs: Local input tensor

        Returns:
            Tuple of (global_mean, global_std)
        """
        if not self.is_distributed:
            # Single process case
            return inputs.mean(dim=0), inputs.std(dim=0)

        # Compute local statistics
        local_sum = torch.sum(inputs, dim=0)
        local_sq_sum = torch.sum(torch.pow(inputs, 2), dim=0)
        local_count = torch.tensor(inputs.size(0), dtype=torch.float32)

        # All-reduce across processes (equivalent to MPI_Allreduce)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        # Compute global mean and std
        global_mean = local_sum / local_count
        global_var = (local_sq_sum / local_count) - torch.pow(global_mean, 2)
        global_std = torch.sqrt(global_var)

        return global_mean, global_std


def create_sample_data(config: Dict) -> str:
    """
    Create sample training data for testing.

    Args:
        config: Configuration dictionary

    Returns:
        Path to saved data file
    """
    print("Creating sample training data...")

    input_size = config['model']['input_size']
    num_samples = config['data'].get('num_samples', 10000)

    # Generate synthetic data matching the NetCDF structure
    np.random.seed(42)

    # Create correlated input features (tair, tsfc, sst, sss, hs, hi, sice)
    inputs = np.random.randn(num_samples, input_size).astype(np.float32)

    # Create realistic ranges for each variable
    # Air temperature: -40 to +20 C
    inputs[:, 0] = inputs[:, 0] * 15 + (-10)
    # Surface temperature: -35 to +25 C
    inputs[:, 1] = inputs[:, 1] * 15 + (-5)
    # Sea surface temperature: -2 to +30 C
    inputs[:, 2] = inputs[:, 2] * 8 + 14
    # Sea surface salinity: 30 to 37 psu
    inputs[:, 3] = inputs[:, 3] * 1.5 + 33.5
    # Snow thickness: 0 to 2 m
    inputs[:, 4] = np.abs(inputs[:, 4]) * 0.5
    # Ice thickness: 0 to 5 m
    inputs[:, 5] = np.abs(inputs[:, 5]) * 1.2
    # Ice salinity: 0 to 20 psu
    inputs[:, 6] = np.abs(inputs[:, 6]) * 4 + 6

    # Create synthetic ice concentration target (0-1)
    # Based on temperature and existing ice
    temp_factor = np.maximum(0, (-inputs[:, 0] + 10) / 30)  # Colder = more ice
    ice_factor = inputs[:, 5] / 5.0  # More thickness = higher concentration
    noise = np.random.randn(num_samples) * 0.1

    targets = np.clip(temp_factor * 0.7 + ice_factor * 0.3 + noise, 0, 1)
    targets = targets.astype(np.float32)

    # Save data
    data_path = Path(config['data']['data_path'])
    data_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(data_path, inputs=inputs, targets=targets)
    print(f"Saved sample data to: {data_path}")
    feature_names = ['tair', 'tsfc', 'sst', 'sss', 'hs', 'hi', 'sice']
    print(f"Input features: {feature_names}")
    print("Target: ice concentration (aice)")

    return str(data_path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        result = yaml.safe_load(f)
        if not isinstance(result, dict):
            raise ValueError(f"Configuration file {config_path} must contain a dictionary")
        return result


def create_default_config() -> Dict:
    """Create default training configuration."""
    return {
        "model": {
            "input_size": 7,  # Updated to match C++ (7 features)
            "hidden_size": 16,
            "output_size": 1  # Updated to match C++ (1 output: aice)
        },
        "domain": {
            "pole": "north",
            "clean_data": True
        },
        "data": {
            "data_path": "data/sample_data.npz",
            "validation_split": 0.2,
            "num_workers": 0,
            "num_samples": 10000
        },
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 1e-5
            },
            "scheduler": {
                "type": "step",
                "step_size": 30,
                "gamma": 0.5
            },
            "loss_function": "mse",
            "early_stopping_patience": 15,
            "log_interval": 50,
            "save_interval": 20
        },
        "output": {
            "model_dir": "models/"
        },
        "use_cuda": True
    }


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize distributed training for HPC environments."""
    # HPC systems often set these environment variables
    if 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = int(os.environ.get(
            'SLURM_LOCALID',
            rank % torch.cuda.device_count()
        ))

        # SLURM sets node list, extract master node
        node_list = os.environ['SLURM_NODELIST']
        if '[' in node_list:
            # Handle node ranges like "node[01-04]"
            base = node_list.split('[')[0]
            first_num = node_list.split('[')[1].split('-')[0].zfill(2)
            master_node = base + first_num
        else:
            master_node = node_list.split(',')[0]

        os.environ['MASTER_ADDR'] = master_node
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)

    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Manual distributed setup
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get(
            'LOCAL_RANK',
            rank % torch.cuda.device_count()
        ))

        # Use provided MASTER_ADDR or default to localhost
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'

    else:
        # Single node setup
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        if torch.cuda.is_available():
            local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = 0

    # Initialize the process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30)  # Longer timeout for HPC
        )

        if rank == 0:
            print("Distributed training initialized:")
            print(f"  Backend: {backend}")
            print(f"  Rank: {rank}/{world_size}")
            master_addr = os.environ['MASTER_ADDR']
            master_port = os.environ['MASTER_PORT']
            print(f"  Master: {master_addr}:{master_port}")
            if torch.cuda.is_available():
                print(f"  Local rank: {local_rank}")

    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        raise


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_distributed(rank: int, world_size: int, config: Dict[str, Any],
                      data_path: str) -> None:
    """
    Distributed training function.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration
        data_path: Path to training data
    """
    # Setup distributed training
    setup_distributed(rank, world_size)

    try:
        # Initialize trainer with distributed settings
        trainer = IceNetTrainer(config, rank=rank, world_size=world_size)

        # Load data
        train_loader, val_loader = trainer.load_data(data_path)

        # Train model
        trainer.train(train_loader, val_loader)

        # Plot training history (only on rank 0)
        if rank == 0:
            trainer.plot_training_history()
            print("Distributed training completed successfully!")

    finally:
        # Clean up
        cleanup_distributed()


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train IceNet model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--create-data', action='store_true',
                        help='Create sample training data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Override data path (.npz, .pt, or .nc)')
    parser.add_argument('--netcdf-file', type=str, default=None,
                        help='NetCDF file to convert to training data')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Total processes for distributed training')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        print("Using default configuration")

    # Override data path if provided
    if args.data_path:
        config['data']['data_path'] = args.data_path
    elif args.netcdf_file:
        config['data']['data_path'] = args.netcdf_file

    # Create sample data if requested
    if args.create_data:
        data_path = create_sample_data(config)
        config['data']['data_path'] = data_path

    # Check if data exists
    data_file = config['data']['data_path']
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        if data_file.endswith('.nc'):
            print("NetCDF file will be processed during training")
        else:
            print("Use --create-data to generate sample data")
            print("Or provide a NetCDF file with --netcdf-file")
            return

    # Distributed training - check for HPC environment
    world_size = args.world_size
    rank = args.local_rank

    # Override with HPC environment variables if present
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

    if world_size > 1:
        print(f"Starting distributed training: rank {rank}/{world_size}")
        # Don't use mp.spawn for HPC - processes are already spawned
        train_distributed(rank, world_size, config, config['data']['data_path'])
    else:
        # Initialize trainer
        trainer = IceNetTrainer(config)

        # Load data
        data_path = config['data']['data_path']
        train_loader, val_loader = trainer.load_data(data_path)

        # Train model
        trainer.train(train_loader, val_loader)

        # Plot training history
        trainer.plot_training_history()

        print("Training completed successfully!")


if __name__ == "__main__":
    main()
