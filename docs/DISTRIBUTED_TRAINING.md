# Distributed Training Implementation

## Overview

The Python training system now supports **distributed training with gradient averaging**, replicating your C++ MPI implementation. Here's how it compares:

## C++ vs Python Parallelization

### Your C++ Implementation (MPI-based)
```cpp
// Data distribution across MPI processes
for (size_t i = getComm().rank(); i < lat.size(); i += getComm().size()) {
    // Process data on each rank
}

// Gradient averaging with MPI_Allreduce
MPI_Allreduce(local_sum.data_ptr(), global_sum.data_ptr(),
              global_sum.numel(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
```

### Python Implementation (PyTorch Distributed)
```python
# Data distribution with DistributedSampler
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

# Automatic gradient averaging with DistributedDataParallel
model = DDP(model, device_ids=[rank])

# Manual statistics averaging (equivalent to your MPI_Allreduce)
dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
```

## Key Features Implemented

### 1. **Distributed Data Loading**
- **DistributedSampler**: Automatically splits data across processes
- **No overlap**: Each process gets unique data samples
- **Consistent splits**: Reproducible data distribution

### 2. **Gradient Averaging**
- **DistributedDataParallel (DDP)**: Automatic gradient synchronization
- **All-reduce operations**: Gradients averaged across all processes after backward pass
- **Same mathematical result**: Equivalent to your C++ MPI implementation

### 3. **Statistics Computation**
- **Distributed normalization**: Mean/std computed across all processes
- **MPI_Allreduce equivalent**: `dist.all_reduce()` for global statistics
- **Exact replication**: Same algorithm as your C++ code

### 4. **Process Coordination**
- **Barrier synchronization**: `dist.barrier()` for process coordination
- **Rank-based operations**: Only rank 0 performs I/O operations
- **Clean initialization/cleanup**: Proper distributed setup/teardown

## Usage Examples

### Single Process (Current Default)
```bash
python train_icenet.py --config config.yaml --create-data
```

### Multi-GPU Single Node
```bash
python distributed_train.py --config config.yaml --gpus 4
```

### Multi-Node with SLURM
```bash
srun --nodes=2 --ntasks-per-node=4 python distributed_train.py \
     --config config.yaml --distributed
```

### Direct Distributed Launch
```bash
python train_icenet.py --config config.yaml --world-size 4 --create-data
```

## Performance Benefits

### Gradient Averaging (Same as C++)
- **Larger effective batch size**: Sum of all local batch sizes
- **Better gradient estimates**: More stable training
- **Linear scaling**: Training time decreases with more processes

### Data Parallelism
- **Automatic load balancing**: PyTorch handles data distribution
- **No communication overhead**: Only gradients are synchronized
- **Memory efficiency**: Each process loads only its data subset

## Mathematical Equivalence

Your C++ implementation:
```
global_mean = Σ(local_sums) / Σ(local_counts)
global_std = sqrt(Σ(local_sq_sums) / Σ(local_counts) - global_mean²)
```

Python implementation:
```python
dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
global_mean = local_sum / local_count
# Same mathematical operations
```

## When to Use Distributed Training

### **Use Distributed Training When:**
- Large datasets that don't fit in single GPU memory
- Want faster training with multiple GPUs/nodes
- Need exactly the same gradient averaging as your C++ code
- Training on HPC systems with SLURM

### **Use Single Process When:**
- Small datasets (< 1M samples)
- Single GPU is sufficient
- Debugging or development
- Simple experimentation

## Configuration

Add to your `config.yaml`:
```yaml
training:
  # Distributed settings
  distributed: true
  backend: "nccl"  # or "gloo" for CPU
  # ... other training settings
```

The distributed implementation provides **exactly the same mathematical behavior** as your C++ MPI code, but with PyTorch's optimized communication libraries and automatic gradient synchronization.
