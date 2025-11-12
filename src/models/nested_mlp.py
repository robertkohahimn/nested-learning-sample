"""
NestedMLP: Multi-layer perceptron with multi-frequency parameter updates.

A simple baseline model demonstrating the Nested Learning paradigm.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from ..layers import NestedLinear, NestedLayerNorm, FrequencyLevel


class NestedMLP(nn.Module):
    """
    Multi-layer perceptron with nested parameter updates.

    This model demonstrates the Nested Learning paradigm with different
    layers updating at different frequencies:
    - Input layer: Slow (stable feature extraction)
    - Hidden layers: Medium (general representations)
    - Output layer: Fast (task-specific adaptation)
    - Normalization: Fast (distribution adaptation)

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'silu')
        use_norm: If True, uses layer normalization
        custom_frequencies: Optional dict to override default frequency assignments
            Example: {'input': FrequencyLevel.MEDIUM, 'hidden': FrequencyLevel.FAST}

    Example:
        >>> # Simple 3-layer MLP with nested updates
        >>> model = NestedMLP(
        ...     input_dim=784,
        ...     hidden_dims=[256, 128],
        ...     output_dim=10,
        ...     dropout=0.1
        ... )
        >>> output = model(input_tensor)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_norm: bool = True,
        custom_frequencies: Optional[Dict[str, int]] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_norm = use_norm

        # Default frequency assignments
        default_freqs = {
            'input': FrequencyLevel.SLOW,     # Input features change slowly
            'hidden': FrequencyLevel.MEDIUM,  # Hidden reps change moderately
            'output': FrequencyLevel.FAST,    # Output adapts quickly
            'norm': FrequencyLevel.FAST       # Normalization adapts quickly
        }

        # Override with custom frequencies if provided
        if custom_frequencies is not None:
            default_freqs.update(custom_frequencies)

        self.frequencies = default_freqs

        # Build layers
        layers = []
        prev_dim = input_dim

        # Input layer (slow updates for stable features)
        layers.append(NestedLinear(
            prev_dim,
            hidden_dims[0] if hidden_dims else output_dim,
            parameter_frequencies={
                'weight': self.frequencies['input'],
                'bias': self.frequencies['input']
            }
        ))

        if use_norm:
            layers.append(NestedLayerNorm(
                hidden_dims[0] if hidden_dims else output_dim,
                frequency_level=self.frequencies['norm']
            ))

        layers.append(self._get_activation(activation))
        layers.append(nn.Dropout(dropout))

        # Hidden layers (medium updates for general representations)
        for i, hidden_dim in enumerate(hidden_dims):
            if i > 0:  # Skip first hidden layer (already added as input layer)
                layers.append(NestedLinear(
                    prev_dim,
                    hidden_dim,
                    parameter_frequencies={
                        'weight': self.frequencies['hidden'],
                        'bias': self.frequencies['hidden']
                    }
                ))

                if use_norm:
                    layers.append(NestedLayerNorm(
                        hidden_dim,
                        frequency_level=self.frequencies['norm']
                    ))

                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (fast updates for quick task adaptation)
        if hidden_dims:  # Only add if we have hidden layers
            layers.append(NestedLinear(
                prev_dim,
                output_dim,
                parameter_frequencies={
                    'weight': self.frequencies['output'],
                    'bias': self.frequencies['output']
                }
            ))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation.lower(), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def get_parameters_by_frequency(self) -> Dict[int, List[nn.Parameter]]:
        """
        Get all parameters grouped by their update frequency.

        Returns:
            Dict mapping frequency level to list of parameters
        """
        from ..layers import get_frequency_aware_param_groups
        return get_frequency_aware_param_groups(self)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f'input_dim={self.input_dim}, '
            f'hidden_dims={self.hidden_dims}, '
            f'output_dim={self.output_dim}, '
            f'frequencies={self.frequencies}'
        )


class NestedSequential(nn.Module):
    """
    Sequential model for sequence processing with nested updates.

    Useful for tasks like time series prediction or sequence classification.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
        num_layers: Number of recurrent/sequential layers
        dropout: Dropout probability
        use_norm: If True, uses layer normalization

    Example:
        >>> model = NestedSequential(
        ...     input_dim=64,
        ...     hidden_dim=128,
        ...     output_dim=10,
        ...     num_layers=3
        ... )
        >>> output = model(sequence_tensor)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection (slow updates)
        self.input_proj = NestedLinear(
            input_dim,
            hidden_dim,
            parameter_frequencies={
                'weight': FrequencyLevel.SLOW,
                'bias': FrequencyLevel.FAST
            }
        )

        # Processing layers (medium updates)
        self.layers = nn.ModuleList([
            nn.Sequential(
                NestedLinear(
                    hidden_dim,
                    hidden_dim,
                    parameter_frequencies={
                        'weight': FrequencyLevel.MEDIUM,
                        'bias': FrequencyLevel.FAST
                    }
                ),
                NestedLayerNorm(hidden_dim) if use_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # Output projection (fast updates)
        self.output_proj = NestedLinear(
            hidden_dim,
            output_dim,
            parameter_frequencies={
                'weight': FrequencyLevel.FAST,
                'bias': FrequencyLevel.FAST
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               or (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
            or (batch_size, output_dim)
        """
        # Handle both 2D and 3D inputs
        is_sequential = x.dim() == 3

        # Input projection
        h = self.input_proj(x)

        # Processing layers with residual connections
        for layer in self.layers:
            h = h + layer(h)

        # Output projection
        output = self.output_proj(h)

        return output

    def get_parameters_by_frequency(self) -> Dict[int, List[nn.Parameter]]:
        """Get parameters grouped by frequency."""
        from ..layers import get_frequency_aware_param_groups
        return get_frequency_aware_param_groups(self)
