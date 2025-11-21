"""
IceNet PyTorch Model in Python
Equivalent implementation of the C++ IceNet model for training purposes.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path


class IceNet(nn.Module):
    """
    Feed Forward Neural Network for ice modeling.
    Equivalent to the C++ IceNet implementation.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 hidden_layers: int = 2, kernel_size: int = 1,
                 stride: int = 1):
        """
        Initialize the IceNet model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units per layer
            output_size: Number of output features
            hidden_layers: Number of hidden layers (default: 2)
            kernel_size: Kernel size (currently unused, for compatibility)
            stride: Stride (currently unused, for compatibility)
        """
        super(IceNet, self).__init__()

        print(f"Starting IceNet constructor: {input_size} -> "
              f"{hidden_layers}x{hidden_size} -> {output_size}")

        # Build dynamic network based on hidden_layers
        layers = []

        # First layer: input -> hidden
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())  # Fixed: Added ReLU activation

        # Additional hidden layers: hidden -> hidden
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Final layer: hidden -> output (no activation - linear output)
        layers.append(nn.Linear(hidden_size, output_size))
        # Removed sigmoid activation to allow unbounded output

        self.network = nn.Sequential(*layers)

        # Register mean and std as buffers (non-trainable parameters)
        self.register_buffer('input_mean', torch.full((input_size,), 0.0))
        self.register_buffer('input_std', torch.full((input_size,), 1.0))

        # Type annotations for buffers (for mypy)
        self.input_mean: torch.Tensor
        self.input_std: torch.Tensor

        # Compute total number of parameters (degrees of freedom)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total degrees of freedom (parameters): {total_params}")

        print("End IceNet constructor")

    def init_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Initialize normalization parameters.

        Args:
            mean: Mean values for input normalization
            std: Standard deviation values for input normalization
        """
        self.input_mean.data = mean.clone()
        self.input_std.data = std.clone()

    def save_norm(self, model_filename: str) -> None:
        """
        Save normalization parameters to file.

        Args:
            model_filename: Path to the model file
        """
        file_path = Path(model_filename)
        path = file_path.parent
        filename = file_path.name

        # Save 1st and 2nd moments
        moments = [self.input_mean, self.input_std]
        norm_path = path / f"normalization.{filename}"
        torch.save(moments, norm_path)
        print(f"Saved normalization to: {norm_path}")

    def load_norm(self, model_filename: str) -> None:
        """
        Load normalization parameters from file.

        Args:
            model_filename: Path to the model file
        """
        file_path = Path(model_filename)
        path = file_path.parent

        # Try the new single normalization file first
        norm_path = path / "normalization.pt"
        if norm_path.exists():
            moments = torch.load(norm_path)
            self.input_mean.data = moments[0]
            self.input_std.data = moments[1]
            print(f"Loaded normalization from: {norm_path}")
        else:
            # Fallback to old per-model normalization files
            filename = file_path.name
            old_norm_path = path / f"normalization.{filename}"
            moments = torch.load(old_norm_path)
            self.input_mean.data = moments[0]
            self.input_std.data = moments[1]
            print(f"Loaded normalization from: {old_norm_path}")

    def init_weights(self) -> None:
        """
        Initialize weights using Xavier normal initialization.
        """
        for module in self.network:
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
        print("Initialized weights with Xavier normal distribution")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor after forward pass
        """
        # Normalize the input
        x = (x - self.input_mean) / self.input_std

        # Forward through dynamic network
        x = self.network(x)

        return x

    def jac(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian (dout/dx) using automatic differentiation.
        Updated to properly handle sigmoid activation and multi-output cases.

        Args:
            x: Input tensor

        Returns:
            Jacobian matrix [batch_size, output_size, input_size]
        """
        # Create input tensor that requires gradients
        x_input = x.clone().detach().requires_grad_(True)

        # Forward pass
        y = self.forward(x_input)
        output_size = y.shape[1]

        # Compute Jacobian for each output
        jacobians = []
        for i in range(output_size):
            # Zero gradients from previous computation
            if x_input.grad is not None:
                x_input.grad.zero_()

            # Backward pass for output i
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            y.backward(grad_outputs, retain_graph=True)

            # Store gradients for this output
            if x_input.grad is None:
                raise RuntimeError("Gradients not computed properly")
            jacobians.append(x_input.grad.clone())

        # Stack jacobians: [batch_size, output_size, input_size]
        jacobian_matrix = torch.stack(jacobians, dim=1)
        return jacobian_matrix

    def jac_norm(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute Frobenius norm of Jacobian (placeholder implementation).

        Args:
            input_tensor: Input tensor

        Returns:
            Frobenius norm (currently returns 0.0)
        """
        return torch.tensor(0.0)

    def save_model(self, model_filename: str) -> None:
        """
        Save the entire model to file.

        Args:
            model_filename: Path to save the model
        """
        torch.save(self.state_dict(), model_filename)
        print(f"Saved model to: {model_filename}")

    def load_model(self, model_filename: str) -> None:
        """
        Load the model from file.

        Args:
            model_filename: Path to the model file
        """
        self.load_state_dict(torch.load(model_filename))
        print(f"Loaded model from: {model_filename}")

        # Print model parameters for debugging (equivalent to C++ code)
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, Size: {param.size()}")

        for name, buffer in self.named_buffers():
            print(f"Buffer name: {name}, Size: {buffer.size()}")
            print(f"       values: {buffer}")


def create_icenet(
        input_size: int, hidden_size: int, output_size: int,
        hidden_layers: int = 2
) -> IceNet:
    """
    Factory function to create and initialize an IceNet model.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units per layer
        output_size: Number of output features
        hidden_layers: Number of hidden layers

    Returns:
        Initialized IceNet model
    """
    model = IceNet(input_size, hidden_size, output_size, hidden_layers)
    model.init_weights()
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing IceNet model...")

    # Create model
    model = create_icenet(input_size=4, hidden_size=10, output_size=2)

    # Test forward pass
    x = torch.randn(1, 4)
    print(f"Input: {x}")

    # Initialize normalization
    mean = torch.zeros(4)
    std = torch.ones(4)
    model.init_norm(mean, std)

    # Forward pass
    output = model.forward(x)
    print(f"Output: {output}")

    # Test Jacobian
    jac = model.jac(x)
    print(f"Jacobian: {jac}")

    print("IceNet model test completed successfully!")
