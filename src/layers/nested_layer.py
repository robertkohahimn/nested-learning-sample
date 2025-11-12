"""
Nested Layer: Neural layers with multi-frequency parameter updates.

This module provides layer implementations where different parameters
update at different frequencies, mimicking neuroplasticity in biological systems.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


class FrequencyLevel(IntEnum):
    """Parameter update frequency levels."""
    FAST = 0      # Updates every step (C=1)
    MEDIUM = 1    # Updates every ~10-100 steps
    SLOW = 2      # Updates every ~1000+ steps


class NestedLayer(nn.Module):
    """
    Base class for neural layers with multi-frequency parameter updates.

    Different parameters can be assigned to different frequency levels,
    enabling adaptive learning rates at different time scales.

    Args:
        parameter_frequencies: Dict mapping parameter names to frequency levels

    Example:
        >>> layer = NestedLinear(10, 20, parameter_frequencies={
        ...     'weight': FrequencyLevel.MEDIUM,
        ...     'bias': FrequencyLevel.FAST
        ... })
    """

    def __init__(self):
        super().__init__()
        # Maps parameter name to frequency level
        self._parameter_frequencies: Dict[str, int] = {}

    def set_parameter_frequency(self, param_name: str, frequency_level: int):
        """
        Assign a frequency level to a parameter.

        Args:
            param_name: Name of the parameter (e.g., 'weight', 'bias')
            frequency_level: Frequency level (0=fast, 1=medium, 2=slow)
        """
        self._parameter_frequencies[param_name] = frequency_level

    def get_parameter_frequency(self, param_name: str) -> int:
        """
        Get frequency level of a parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Frequency level (default: 0 if not set)
        """
        return self._parameter_frequencies.get(param_name, 0)

    def get_parameters_by_frequency(self) -> Dict[int, List[nn.Parameter]]:
        """
        Group parameters by their frequency level.

        Returns:
            Dict mapping frequency level to list of parameters
        """
        params_by_freq: Dict[int, List[nn.Parameter]] = {}

        for name, param in self.named_parameters():
            # Extract base parameter name (remove any prefixes)
            base_name = name.split('.')[-1]
            freq_level = self.get_parameter_frequency(base_name)

            if freq_level not in params_by_freq:
                params_by_freq[freq_level] = []
            params_by_freq[freq_level].append(param)

        return params_by_freq


class NestedLinear(NestedLayer):
    """
    Linear layer with multi-frequency parameter updates.

    This is a drop-in replacement for nn.Linear with added support
    for frequency-based parameter grouping.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds a learnable bias
        parameter_frequencies: Dict mapping parameter names to frequency levels
            Example: {'weight': FrequencyLevel.MEDIUM, 'bias': FrequencyLevel.FAST}

    Example:
        >>> # Create layer with weight updating slowly, bias updating fast
        >>> layer = NestedLinear(
        ...     128, 64,
        ...     parameter_frequencies={
        ...         'weight': FrequencyLevel.SLOW,
        ...         'bias': FrequencyLevel.FAST
        ...     }
        ... )
        >>> output = layer(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parameter_frequencies: Optional[Dict[str, int]] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Create parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

        # Set frequency levels
        if parameter_frequencies is not None:
            for param_name, freq_level in parameter_frequencies.items():
                self.set_parameter_frequency(param_name, freq_level)

    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        freq_info = []
        for name, freq in self._parameter_frequencies.items():
            freq_info.append(f"{name}=Freq{freq}")

        base_repr = f'in_features={self.in_features}, out_features={self.out_features}'
        if self.bias is None:
            base_repr += ', bias=False'

        if freq_info:
            base_repr += f', frequencies={{' + ', '.join(freq_info) + '}}'

        return base_repr


class NestedEmbedding(NestedLayer):
    """
    Embedding layer with multi-frequency updates.

    Typically, embeddings are updated slowly to maintain stable representations.

    Args:
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        padding_idx: If specified, entries at padding_idx do not contribute to gradient
        frequency_level: Frequency level for the embedding weights (default: SLOW)

    Example:
        >>> # Embedding that updates slowly (stable representations)
        >>> embed = NestedEmbedding(
        ...     num_embeddings=10000,
        ...     embedding_dim=128,
        ...     frequency_level=FrequencyLevel.SLOW
        ... )
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        frequency_level: int = FrequencyLevel.SLOW
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Create embedding
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        # Set frequency
        self.set_parameter_frequency('weight', frequency_level)

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Tensor containing indices of shape (*)

        Returns:
            Embedded vectors of shape (*, embedding_dim)
        """
        return torch.nn.functional.embedding(
            input, self.weight, self.padding_idx
        )

    def extra_repr(self) -> str:
        """Extra representation."""
        freq = self.get_parameter_frequency('weight')
        s = f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        s += f', frequency=Freq{freq}'
        return s


class NestedLayerNorm(NestedLayer):
    """
    Layer normalization with multi-frequency updates.

    Typically, normalization parameters (scale/shift) are updated quickly
    to adapt to changing activation distributions.

    Args:
        normalized_shape: Input shape from an expected input
        eps: Value added to denominator for numerical stability
        elementwise_affine: Whether to learn affine parameters
        frequency_level: Frequency level for scale/shift (default: FAST)

    Example:
        >>> # LayerNorm that adapts quickly
        >>> norm = NestedLayerNorm(
        ...     normalized_shape=128,
        ...     frequency_level=FrequencyLevel.FAST
        ... )
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        frequency_level: int = FrequencyLevel.FAST
    ):
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

            # LayerNorm params typically update fast
            self.set_parameter_frequency('weight', frequency_level)
            self.set_parameter_frequency('bias', frequency_level)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, ..., normalized_shape)

        Returns:
            Normalized tensor of same shape
        """
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        """Extra representation."""
        s = f'normalized_shape={self.normalized_shape}, eps={self.eps}'
        if self.elementwise_affine:
            freq = self.get_parameter_frequency('weight')
            s += f', frequency=Freq{freq}'
        else:
            s += ', elementwise_affine=False'
        return s


def get_frequency_aware_param_groups(
    model: nn.Module
) -> Dict[int, List[nn.Parameter]]:
    """
    Extract parameter groups organized by frequency level from a model.

    This function walks through all NestedLayer modules in the model
    and organizes their parameters by frequency level.

    Args:
        model: PyTorch model (potentially containing NestedLayer modules)

    Returns:
        Dict mapping frequency level to list of parameters

    Example:
        >>> model = nn.Sequential(
        ...     NestedLinear(10, 20, parameter_frequencies={'weight': FrequencyLevel.SLOW}),
        ...     NestedLinear(20, 5, parameter_frequencies={'weight': FrequencyLevel.FAST})
        ... )
        >>> param_groups = get_frequency_aware_param_groups(model)
        >>> # param_groups[0] contains fast parameters
        >>> # param_groups[2] contains slow parameters
    """
    params_by_freq: Dict[int, List[nn.Parameter]] = {}

    for module in model.modules():
        if isinstance(module, NestedLayer):
            module_params = module.get_parameters_by_frequency()
            for freq_level, params in module_params.items():
                if freq_level not in params_by_freq:
                    params_by_freq[freq_level] = []
                params_by_freq[freq_level].extend(params)

    return params_by_freq
