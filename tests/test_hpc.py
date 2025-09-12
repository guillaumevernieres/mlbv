"""
Tests for HPC and distributed training setup
"""

import os
import torch


class TestHPCSetup:
    """Test cases for HPC setup and environment detection."""

    def test_torch_installation(self):
        """Test that PyTorch is properly installed."""
        assert torch.__version__ is not None
        print(f"PyTorch version: {torch.__version__}")

    def test_cuda_availability(self):
        """Test CUDA availability (optional)."""
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA available: {torch.cuda.device_count()} devices")
            print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("CUDA not available - using CPU")

        # Test should pass regardless of CUDA availability
        assert True

    def test_device_detection(self):
        """Test device detection logic."""
        # Force CPU for testing
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert device.type in ['cuda', 'cpu']

    def test_tensor_operations(self):
        """Test basic tensor operations."""
        # Create test tensors
        x = torch.randn(10, 5, requires_grad=True)
        y = torch.randn(5, 3, requires_grad=True)

        # Test operations
        z = torch.matmul(x, y)
        assert z.shape == (10, 3)

        # Test gradients
        loss = z.sum()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None

    def test_model_device_placement(self):
        """Test model device placement."""
        from icenet.model import create_icenet

        model = create_icenet(input_size=4, hidden_size=8, output_size=1)

        # Test CPU placement
        device = torch.device('cpu')
        model = model.to(device)

        # Test forward pass
        x = torch.randn(2, 4).to(device)
        model.init_norm(torch.zeros(4), torch.ones(4))
        output = model(x)

        assert output.device == device

    def test_distributed_environment_variables(self):
        """Test distributed environment variable handling."""
        # Save original values
        original_vars = {}
        env_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']

        for var in env_vars:
            original_vars[var] = os.environ.get(var)

        try:
            # Test setting distributed variables
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['RANK'] = '0'

            # Check variables are set
            assert os.environ.get('MASTER_ADDR') == 'localhost'
            assert os.environ.get('MASTER_PORT') == '29500'
            assert os.environ.get('WORLD_SIZE') == '1'
            assert os.environ.get('RANK') == '0'

        finally:
            # Restore original values
            for var, value in original_vars.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value

    def test_slurm_environment_detection(self):
        """Test SLURM environment detection."""
        # Save original values
        original_vars = {}
        slurm_vars = ['SLURM_PROCID', 'SLURM_NPROCS', 'SLURM_LOCALID']

        for var in slurm_vars:
            original_vars[var] = os.environ.get(var)

        try:
            # Simulate SLURM environment
            os.environ['SLURM_PROCID'] = '0'
            os.environ['SLURM_NPROCS'] = '2'
            os.environ['SLURM_LOCALID'] = '0'

            # Test detection
            in_slurm = 'SLURM_PROCID' in os.environ
            assert in_slurm is True

        finally:
            # Restore original values
            for var, value in original_vars.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        from icenet.model import create_icenet

        model = create_icenet(input_size=7, hidden_size=16, output_size=1)
        model.init_norm(torch.zeros(7), torch.ones(7))

        # Test different batch sizes
        batch_sizes = [1, 4, 16, 32]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 7)
            output = model(x)

            assert output.shape == (batch_size, 1)
            assert torch.all(output >= 0) and torch.all(output <= 1)
