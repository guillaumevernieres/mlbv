#!/usr/bin/env python3
"""
HPC launcher for distributed IceNet training
Supports SLURM, PBS, and manual distributed setups

Usage:
  # Single node, multiple GPUs
  python launch_hpc_training.py --config config.yaml --nodes 1 --gpus-per-node 4

  # Multi-node SLURM
  sbatch slurm_job.sh

  # Manual distributed
  MASTER_ADDR=node01 MASTER_PORT=29500 python launch_hpc_training.py --config config.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def create_slurm_script(config_file: str, nodes: int, gpus_per_node: int,
                       cpus_per_task: int, time_limit: str,
                       partition: str = None, account: str = None) -> str:
    """Create SLURM batch script for distributed training."""

    total_gpus = nodes * gpus_per_node
    script_content = f"""#!/bin/bash
#SBATCH --job-name=icenet_training
#SBATCH --nodes={nodes}
#SBATCH --ntasks={total_gpus}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --output=icenet_training_%j.out
#SBATCH --error=icenet_training_%j.err
"""

    if partition:
        script_content += f"#SBATCH --partition={partition}\n"
    if account:
        script_content += f"#SBATCH --account={account}\n"

    script_content += f"""
# Load modules (adapt for your HPC system)
# module load python/3.9
# module load cuda/11.7
# module load pytorch/1.13

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure CUDA is visible
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run distributed training
srun python ../scripts/train.py --config {config_file} \\
    --world-size $SLURM_NTASKS \\
    --local-rank $SLURM_LOCALID
"""

    script_path = "slurm_icenet_training.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"Created SLURM script: {script_path}")
    return script_path


def create_pbs_script(config_file: str, nodes: int, gpus_per_node: int,
                     cpus_per_task: int, time_limit: str,
                     queue: str = None) -> str:
    """Create PBS batch script for distributed training."""

    total_gpus = nodes * gpus_per_node
    script_content = f"""#!/bin/bash
#PBS -N icenet_training
#PBS -l nodes={nodes}:ppn={gpus_per_node}:gpus={gpus_per_node}
#PBS -l walltime={time_limit}
#PBS -o icenet_training.out
#PBS -e icenet_training.err
"""

    if queue:
        script_content += f"#PBS -q {queue}\n"

    script_content += f"""
# Load modules (adapt for your HPC system)
# module load python/3.9
# module load cuda/11.7
# module load pytorch/1.13

cd $PBS_O_WORKDIR

# Get node list
NODELIST=$(cat $PBS_NODEFILE | uniq)
MASTER_NODE=$(cat $PBS_NODEFILE | head -n 1)

export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500
export WORLD_SIZE={total_gpus}
export OMP_NUM_THREADS={cpus_per_task}

# Run distributed training using mpirun
mpirun -np {total_gpus} -hostfile $PBS_NODEFILE \\
    python ../scripts/train.py --config {config_file} \\
    --world-size {total_gpus}
"""

    script_path = "pbs_icenet_training.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"Created PBS script: {script_path}")
    return script_path


def launch_single_node(config_file: str, gpus_per_node: int):
    """Launch single-node distributed training."""
    import torch.multiprocessing as mp
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from icenet.training import train_distributed, load_config

    config = load_config(config_file)

    print(f"Launching single-node training with {gpus_per_node} processes")

    # Use multiprocessing spawn for single node
    mp.spawn(
        train_distributed,
        args=(gpus_per_node, config, config['data']['data_path']),
        nprocs=gpus_per_node,
        join=True
    )


def launch_manual_distributed(config_file: str):
    """Launch using manual environment variables."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from icenet.training import main

    # Check required environment variables
    required_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        print(f"Missing environment variables: {missing_vars}")
        print("For manual distributed training, set:")
        print("  MASTER_ADDR=<master_node_ip>")
        print("  MASTER_PORT=<port>")
        print("  WORLD_SIZE=<total_processes>")
        print("  RANK=<process_rank>")
        print("  LOCAL_RANK=<local_process_rank>  # optional")
        sys.exit(1)

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    print(f"Launching process {rank}/{world_size}")

    # Add distributed arguments and run
    sys.argv.extend([
        '--config', config_file,
        '--world-size', str(world_size),
        '--local-rank', str(rank)
    ])

    main()


def main():
    parser = argparse.ArgumentParser(
        description='Launch distributed IceNet training on HPC systems'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--scheduler', type=str,
                        choices=['slurm', 'pbs', 'manual', 'single-node'],
                        default='single-node',
                        help='Job scheduler type')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes to use')
    parser.add_argument('--gpus-per-node', type=int, default=1,
                        help='Number of GPUs per node')
    parser.add_argument('--cpus-per-task', type=int, default=4,
                        help='Number of CPU cores per task')
    parser.add_argument('--time-limit', type=str, default='02:00:00',
                        help='Job time limit (HH:MM:SS)')
    parser.add_argument('--partition', type=str, default=None,
                        help='SLURM partition name')
    parser.add_argument('--queue', type=str, default=None,
                        help='PBS queue name')
    parser.add_argument('--account', type=str, default=None,
                        help='SLURM account name')
    parser.add_argument('--submit', action='store_true',
                        help='Automatically submit the job')

    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)

    if args.scheduler == 'slurm':
        script_path = create_slurm_script(
            args.config, args.nodes, args.gpus_per_node,
            args.cpus_per_task, args.time_limit,
            args.partition, args.account
        )

        if args.submit:
            print("Submitting SLURM job...")
            result = subprocess.run(['sbatch', script_path],
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        else:
            print(f"To submit the job, run: sbatch {script_path}")

    elif args.scheduler == 'pbs':
        script_path = create_pbs_script(
            args.config, args.nodes, args.gpus_per_node,
            args.cpus_per_task, args.time_limit, args.queue
        )

        if args.submit:
            print("Submitting PBS job...")
            result = subprocess.run(['qsub', script_path],
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        else:
            print(f"To submit the job, run: qsub {script_path}")

    elif args.scheduler == 'single-node':
        launch_single_node(args.config, args.gpus_per_node)

    elif args.scheduler == 'manual':
        launch_manual_distributed(args.config)


if __name__ == "__main__":
    main()
