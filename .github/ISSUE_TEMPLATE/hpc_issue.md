---
name: HPC/Distributed Training Issue
about: Report issues with HPC clusters or distributed training
title: '[HPC] '
labels: hpc, distributed
assignees: ''
---

**HPC Environment**
- Scheduler: [e.g. SLURM, PBS, manual]
- Number of nodes: [e.g. 4]
- GPUs per node: [e.g. 2]
- Network: [e.g. InfiniBand FDR, Ethernet]
- Cluster name/type: [e.g. institutional HPC, cloud cluster]

**Issue Description**
A clear and concise description of the issue.

**Command/Configuration**
```bash
# Command used
python scripts/launch_hpc_training.py --config configs/config.yaml --nodes 4 --gpus-per-node 2

# Or manual setup
MASTER_ADDR=node01 MASTER_PORT=29500 python scripts/train.py ...
```

**Configuration file**
```yaml
# Your config.yaml content
```

**Job script (if applicable)**
```bash
# SLURM/PBS job script content
```

**Error logs**
```
# Error messages from job output
```

**Expected behavior**
What should happen in a successful distributed training run.

**Additional HPC details**
- Module system: [e.g. Environment Modules, Lmod]
- MPI implementation: [e.g. OpenMPI, Intel MPI]
- NCCL version: [if known]
- Network topology: [if relevant]

**Troubleshooting attempted**
What steps have you already tried to resolve the issue?
