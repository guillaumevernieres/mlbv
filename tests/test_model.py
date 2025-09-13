"""
Unit tests for IceNet model functionality
"""

import torch

from icenet.model import IceNet, create_icenet


class TestIceNetModel:
    """Test cases for the IceNet model class."""

    def test_model_creation(self):
        """Test basic model creation."""
        model = create_icenet(input_size=7, hidden_size=16, output_size=1)

        assert isinstance(model, IceNet)
        assert model.fc1.in_features == 7
        assert model.fc1.out_features == 16
        assert model.fc2.in_features == 16
        assert model.fc2.out_features == 1

    def test_model_forward_pass(self, sample_data):
        """Test forward pass with sample data."""
        X, y = sample_data
        model = create_icenet(input_size=7, hidden_size=16, output_size=1)

        # Initialize normalization
        mean = X.mean(dim=0)
        std = X.std(dim=0)
        model.init_norm(mean, std)

        # Forward pass
        output = model(X[:10])  # Test with first 10 samples

        assert output.shape == (10, 1)
        # Check sigmoid output is in valid range
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_normalization(self):
        """Test input normalization functionality."""
        model = create_icenet(input_size=4, hidden_size=8, output_size=1)

        # Test normalization initialization
        mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
        std = torch.tensor([0.5, 1.0, 1.5, 2.0])
        model.init_norm(mean, std)

        assert torch.allclose(model.input_mean, mean)
        assert torch.allclose(model.input_std, std)

    def test_jacobian_computation(self):
        """Test Jacobian computation."""
        model = create_icenet(input_size=3, hidden_size=5, output_size=1)

        # Initialize normalization
        model.init_norm(torch.zeros(3), torch.ones(3))

        x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        jac = model.jac(x)

        assert jac is not None
        assert jac.shape == x.shape

    def test_model_save_load(self, temp_dir):
        """Test model saving and loading."""
        model = create_icenet(input_size=4, hidden_size=8, output_size=1)

        # Save model
        model_path = temp_dir / "test_model.pth"
        model.save_model(str(model_path))

        assert model_path.exists()

        # Create new model and load
        new_model = create_icenet(input_size=4, hidden_size=8, output_size=1)
        new_model.load_model(str(model_path))

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_normalization_save_load(self, temp_dir):
        """Test normalization parameter saving and loading."""
        model = create_icenet(input_size=3, hidden_size=5, output_size=1)

        # Set normalization parameters
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([0.5, 1.0, 1.5])
        model.init_norm(mean, std)

        # Save normalization
        model_path = temp_dir / "test_norm.pth"
        model.save_norm(str(model_path))

        # Create new model and load normalization
        new_model = create_icenet(input_size=3, hidden_size=5, output_size=1)
        new_model.load_norm(str(model_path))

        assert torch.allclose(new_model.input_mean, mean)
        assert torch.allclose(new_model.input_std, std)

    def test_different_architectures(self):
        """Test model creation with different architectures."""
        architectures = [
            (5, 10, 1),
            (7, 16, 1),
            (10, 32, 2),
            (3, 64, 3)
        ]

        for input_size, hidden_size, output_size in architectures:
            model = create_icenet(input_size, hidden_size, output_size)

            # Test forward pass
            x = torch.randn(2, input_size)
            model.init_norm(torch.zeros(input_size), torch.ones(input_size))
            output = model(x)

            assert output.shape == (2, output_size)

    def test_model_training_mode(self):
        """Test model training/eval mode switching."""
        model = create_icenet(input_size=4, hidden_size=8, output_size=1)

        # Test training mode
        model.train()
        assert model.training

        # Test eval mode
        model.eval()
        assert not model.training

    def test_model_parameters_count(self):
        """Test parameter counting for different architectures."""
        model = create_icenet(input_size=7, hidden_size=16, output_size=1)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Expected: (7*16 + 16) + (16*1 + 1) = 112 + 16 + 16 + 1 = 145
        expected_params = (7 * 16 + 16) + (16 * 1 + 1)
        assert total_params == expected_params

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = create_icenet(input_size=3, hidden_size=5, output_size=1)
        model.init_norm(torch.zeros(3), torch.ones(3))

        x = torch.randn(1, 3, requires_grad=True)
        y_target = torch.randn(1, 1)

        # Forward pass
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y_target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_model_reproducibility(self):
        """Test that model outputs are reproducible with same inputs."""
        torch.manual_seed(42)
        model1 = create_icenet(input_size=4, hidden_size=8, output_size=1)
        model1.init_norm(torch.zeros(4), torch.ones(4))

        torch.manual_seed(42)
        model2 = create_icenet(input_size=4, hidden_size=8, output_size=1)
        model2.init_norm(torch.zeros(4), torch.ones(4))

        x = torch.randn(5, 4)
        output1 = model1(x)
        output2 = model2(x)

        assert torch.allclose(output1, output2, atol=1e-6)
