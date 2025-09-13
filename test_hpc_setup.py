#!/usr/bin/env python3
"""
Test script for HPC distributed training setup
Verifies that distributed training initialization works correctly

Usage:
  # Test single node
  python test_hpc_setup.py --gpus 2

  # Test with SLURM (in job)
  srun python test_hpc_setup.py

  # Test manual distributed
  MASTER_ADDR=node01 WORLD_SIZE=4 RANK=0 python test_hpc_setup.py
"""

import os
import torch
import torch.distributed as dist
from datetime import timedelta


def setup_distributed_test() -> bool:
    """Test distributed setup."""
    print("=" * 60)
    print("HPC Distributed Training Setup Test")
    print("=" * 60)

    # Detect environment
    if 'SLURM_PROCID' in os.environ:
        print("SLURM environment detected:")
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        node_list = os.environ.get('SLURM_NODELIST', 'unknown')

        print(f"  Rank: {rank}/{world_size}")
        print(f"  Local rank: {local_rank}")
        print(f"  Node list: {node_list}")

        # Extract master node
        if '[' in node_list:
            base = node_list.split('[')[0]
            first_num = node_list.split('[')[1].split('-')[0].zfill(2)
            master_node = base + first_num
        else:
            master_node = node_list.split(',')[0]

        os.environ['MASTER_ADDR'] = master_node
        os.environ['MASTER_PORT'] = '29500'

    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("Manual distributed environment detected:")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))

        print(f"  Rank: {rank}/{world_size}")
        print(f"  Local rank: {local_rank}")
        print(f"  Master: {os.environ.get('MASTER_ADDR', 'not set')}")

    else:
        print("Single process environment")
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

    print(f"\nEnvironment variables:")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")

    # Test CUDA availability
    print(f"\nCUDA information:")
    print(f"  Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        device = torch.device(f'cuda:{local_rank % torch.cuda.device_count()}')
        print(f"  Assigned device: {device}")
    else:
        device = torch.device('cpu')
        print(f"  Using CPU")

    # Test distributed initialization if multi-process
    if world_size > 1:
        print(f"\nInitializing distributed training...")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        print(f"  Backend: {backend}")

        try:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=5)
            )
            print(f"  ✓ Initialization successful")

            # Test communication
            if rank == 0:
                print(f"  Testing all-reduce operation...")

            test_tensor = torch.tensor([rank + 1.0], device=device)
            print(f"  Rank {rank}: Initial tensor = {test_tensor.item()}")

            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            expected_sum = sum(range(1, world_size + 1))

            print(f"  Rank {rank}: After all-reduce = {test_tensor.item()}")

            if abs(test_tensor.item() - expected_sum) < 1e-6:
                print(f"  ✓ All-reduce test passed")
            else:
                print(f"  ✗ All-reduce test failed: "
                      f"expected {expected_sum}, got {test_tensor.item()}")

            # Cleanup
            dist.destroy_process_group()
            print(f"  ✓ Process group destroyed")

        except Exception as e:
            print(f"  ✗ Initialization failed: {e}")
            return False

    print(f"\n" + "=" * 60)
    print(f"Test completed successfully for rank {rank}/{world_size}")
    print(f"Ready for distributed IceNet training!")
    print(f"=" * 60)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test HPC distributed setup')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs for single-node test')
    args = parser.parse_args()

    # For single-node testing with multiple processes
    if args.gpus > 1 and 'SLURM_PROCID' not in os.environ and 'RANK' not in os.environ:
        import torch.multiprocessing as mp

        def test_process(rank: int, world_size: int) -> None:
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['LOCAL_RANK'] = str(rank)
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            setup_distributed_test()

        print(f"Starting {args.gpus} processes for testing...")
        mp.spawn(test_process, args=(args.gpus,), nprocs=args.gpus, join=True)
    else:
        setup_distributed_test()
