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
                 kernel_size: int = 1, stride: int = 1):
        """
        Initialize the IceNet model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in the first layer
            output_size: Number of output features
            kernel_size: Kernel size (currently unused, for compatibility)
            stride: Stride (currently unused, for compatibility)
        """
        super(IceNet, self).__init__()

        print(f"Starting IceNet constructor: {input_size} {output_size} "
              f"{hidden_size}")

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Register mean and std as buffers (non-trainable parameters)
        self.register_buffer('input_mean', torch.full((input_size,), 0.0))
        self.register_buffer('input_std', torch.full((input_size,), 1.0))

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
        filename = file_path.name

        # Load 1st and 2nd moments
        norm_path = path / f"normalization.{filename}"
        moments = torch.load(norm_path)
        self.input_mean.data = moments[0]
        self.input_std.data = moments[1]
        print(f"Loaded normalization from: {norm_path}")

    def init_weights(self) -> None:
        """
        Initialize weights using Xavier normal initialization.
        """
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
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

        # Pass through layers
        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))

        return x

    def jac(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian (dout/dx) using automatic differentiation.

        Args:
            x: Input tensor

        Returns:
            Jacobian matrix
        """
        # Create input tensor that requires gradients
        x_input = x.clone().detach().requires_grad_(True)

        # Forward pass
        y = self.forward(x_input)

        # Compute gradients
        y.backward(torch.ones_like(y))

        return x_input.grad

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

        # Print model parameters for debugging (equivalent to C++ commented code)
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, Size: {param.size()}")

        for name, buffer in self.named_buffers():
            print(f"Buffer name: {name}, Size: {buffer.size()}")
            print(f"       values: {buffer}")


def create_icenet(input_size: int, hidden_size: int, output_size: int) -> IceNet:
    """
    Factory function to create and initialize an IceNet model.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output features

    Returns:
        Initialized IceNet model
    """
    model = IceNet(input_size, hidden_size, output_size)
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
