#!/usr/bin/env python3
"""
Distributed training launcher for IceNet.
Provides easy commands for single-node and multi-node distributed training.

Examples:
  # Single GPU
  python distributed_train.py --config config.yaml

  # Multi-GPU on single node (4 GPUs)
  python distributed_train.py --config config.yaml --gpus 4

  # Multi-node with SLURM
  srun --nodes=2 --ntasks-per-node=4 python distributed_train.py \\
       --config config.yaml --distributed
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from typing import Optional


def run_single_process(config_file: str, create_data: bool = False, data_path: Optional[str] = None) -> None:
    """Run single process training."""
    cmd = [sys.executable, "train_icenet.py", "--config", config_file]

    if create_data:
        cmd.append("--create-data")
    if data_path:
        cmd.extend(["--data-path", data_path])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_multi_gpu(config_file: str, num_gpus: int, create_data: bool = False, data_path: Optional[str] = None) -> None:
    """Run multi-GPU training on single node."""
    cmd = [
        sys.executable, "train_icenet.py",
        "--config", config_file,
        "--world-size", str(num_gpus)
    ]

    if create_data:
        cmd.append("--create-data")
    if data_path:
        cmd.extend(["--data-path", data_path])

    print(f"Running multi-GPU training with {num_gpus} processes")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_distributed_slurm(config_file: str, create_data: bool = False, data_path: Optional[str] = None) -> None:
    """Run with SLURM distributed setup."""
    # Get SLURM environment variables
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NPROCS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))

    # Set distributed environment variables
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

    cmd = [
        sys.executable, "train_icenet.py",
        "--config", config_file,
        "--world-size", str(world_size),
        "--local-rank", str(local_rank)
    ]

    if create_data:
        cmd.append("--create-data")
    if data_path:
        cmd.extend(["--data-path", data_path])

    print(f"SLURM distributed training: rank {rank}/{world_size}")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Distributed training launcher for IceNet'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='YAML configuration file')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs for single-node training')
    parser.add_argument('--distributed', action='store_true',
                        help='Use SLURM distributed training')
    parser.add_argument('--create-data', action='store_true',
                        help='Create sample training data')
    parser.add_argument('--data-path', type=str,
                        help='Override data path from config')

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file {args.config} not found")
        return

    if args.distributed:
        # SLURM distributed training
        run_distributed_slurm(args.config, args.create_data, args.data_path)
    elif args.gpus > 1:
        # Multi-GPU single node
        run_multi_gpu(args.config, args.gpus, args.create_data, args.data_path)
    else:
        # Single process
        run_single_process(args.config, args.create_data, args.data_path)


if __name__ == "__main__":
    main()
