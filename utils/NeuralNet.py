import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.

    Attributes:
        layer1 (torch.nn.Linear): Fully connected layer with input_size input units and hidden_size output units.
        relu (torch.nn.ReLU): ReLU activation function for introducing non-linearity.
        layer2 (torch.nn.Linear): Fully connected layer with hidden_size input units and output_size output units.

    Methods:
        forward(x): Defines the forward pass of the neural network.
    """

    def __init__(self, input_size=3, hidden_size=10, output_size=3):
        """
        Initialize the neural network components.
        """
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()           # ReLU activation function.
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Perform the forward pass with the input tensor 'x'.

        Parameters:
            x (torch.Tensor): The input tensor to the neural network.

        Returns:
            torch.Tensor: The output tensor after passing through two layers and ReLU activation.
        """
        x = self.layer1(x)  # First layer transformation.
        x = self.relu(x)    # Activation function.
        x = self.layer2(x)  # Output layer transformation.
        return x
