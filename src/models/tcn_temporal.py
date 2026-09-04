import torch
import torch.nn as nn
from typing import List


class Chomp1d(nn.Module):
    """Removes the extra padding added for causal convolution."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A residual block for Temporal Convolutional Network.
    
    Implements dilated causal convolution.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # First causal conv
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal conv
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalTCN(nn.Module):
    """Temporal Convolutional Network for 90-day lookback per-node processing.
    
    Attributes:
        num_inputs (int): Number of input features.
        num_channels (List[int]): Number of channels for each layer.
        kernel_size (int): Size of the convolutional kernel.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ) -> None:
        """Initializes the TCN.
        
        Args:
            num_inputs: Dimensionality of input features.
            num_channels: List indicating the output channels of each residual block.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TCN.
        
        Args:
            x (torch.Tensor): Input temporal data.
                Shape: (batch_size * num_nodes, feature_dim, lookback_days)
                Wait, standard 1D conv expects (B, C, L).
                If input is (batch_size, num_nodes, lookback_days, feature_dim),
                it should be reshaped before calling this.
                
        Returns:
            torch.Tensor: Temporal embeddings. 
                Shape: (batch_size * num_nodes, temporal_embed_dim, lookback_days)
                Typically we take the last timestep as the embedding.
        """
        # x is (B, C, L)
        out = self.network(x)
        # Take the embedding from the last time step: out[:, :, -1]
        return out[:, :, -1]
