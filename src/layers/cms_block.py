"""
CMS Block: Memory-augmented neural network block.

Integrates Continuum Memory System with standard neural network computations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from ..memory.cms import ContinuumMemorySystem
from .nested_layer import NestedLinear, NestedLayerNorm, FrequencyLevel


class CMSBlock(nn.Module):
    """
    Memory-augmented neural network block with Continuum Memory System.

    This block combines standard feedforward computation with CMS for
    multi-scale context processing. It can be used as a building block
    for memory-augmented architectures.

    Architecture:
        Input → [CMS Retrieval] → FFN → [CMS Update] → Output
                 ↓                          ↑
                 └── Memory Banks (multi-frequency) ──┘

    Args:
        hidden_dim: Dimension of hidden states
        ffn_dim: Dimension of feedforward network (default: 4 * hidden_dim)
        memory_config: Configuration for CMS memory banks
            Example: {0: {'capacity': 100}, 1: {'capacity': 50}}
        use_memory: If True, uses CMS for memory augmentation
        dropout: Dropout probability
        activation: Activation function (default: 'gelu')

    Example:
        >>> block = CMSBlock(
        ...     hidden_dim=128,
        ...     ffn_dim=512,
        ...     memory_config={
        ...         0: {'capacity': 100},  # Short-term
        ...         1: {'capacity': 50}    # Long-term
        ...     }
        ... )
        >>> output, _ = block(input_tensor, step=0)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        memory_config: Optional[Dict[int, Dict]] = None,
        use_memory: bool = True,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim or (4 * hidden_dim)
        self.use_memory = use_memory

        # Pre-norm layer (updates fast to adapt to distribution changes)
        self.norm1 = NestedLayerNorm(
            hidden_dim,
            frequency_level=FrequencyLevel.FAST
        )

        # Feedforward network
        # First layer: medium frequency (general features)
        # Second layer: fast frequency (task-specific adaptation)
        self.ffn = nn.Sequential(
            NestedLinear(
                hidden_dim,
                self.ffn_dim,
                parameter_frequencies={
                    'weight': FrequencyLevel.MEDIUM,
                    'bias': FrequencyLevel.FAST
                }
            ),
            self._get_activation(activation),
            nn.Dropout(dropout),
            NestedLinear(
                self.ffn_dim,
                hidden_dim,
                parameter_frequencies={
                    'weight': FrequencyLevel.FAST,
                    'bias': FrequencyLevel.FAST
                }
            ),
            nn.Dropout(dropout)
        )

        # Post-norm layer
        self.norm2 = NestedLayerNorm(
            hidden_dim,
            frequency_level=FrequencyLevel.FAST
        )

        # Continuum Memory System
        if use_memory and memory_config is not None:
            # Convert memory_config dict to memory_sizes and frequencies
            memory_sizes = []
            frequencies = []
            for level in sorted(memory_config.keys()):
                memory_sizes.append(memory_config[level]['capacity'])
                # Frequencies: level 0 = 1, level 1 = 10, level 2 = 100, etc.
                frequencies.append(10 ** level if level > 0 else 1)

            self.cms = ContinuumMemorySystem(
                memory_sizes=memory_sizes,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_levels=len(memory_sizes),
                frequencies=frequencies
            )

            # Projection for memory retrieval
            self.memory_proj = NestedLinear(
                hidden_dim * 2,  # Concat [hidden, retrieved]
                hidden_dim,
                parameter_frequencies={
                    'weight': FrequencyLevel.MEDIUM,
                    'bias': FrequencyLevel.FAST
                }
            )
        else:
            self.cms = None
            self.memory_proj = None

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation.lower(), nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        step: Optional[int] = None,
        return_memory_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with optional memory augmentation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            step: Current training step (for CMS updates)
            return_memory_info: If True, returns memory retrieval information

        Returns:
            output: Output tensor of same shape as input
            memory_info: Optional dict with memory retrieval information
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Pre-norm
        normed = self.norm1(x)

        # Memory-augmented computation
        if self.use_memory and self.cms is not None and step is not None:
            # Use hidden states as keys/values
            # Flatten sequence dimension for memory operations
            keys = normed.reshape(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)

            # Retrieve from memory (aggregates across all levels)
            retrieved_flat = self.cms.retrieve(keys, aggregate=True)

            # Reshape back to sequence format
            retrieved = retrieved_flat.reshape(batch_size, seq_len, hidden_dim)

            # Combine input with retrieved memory
            combined = torch.cat([normed, retrieved], dim=-1)
            memory_augmented = self.memory_proj(combined)

            # Residual connection with memory-augmented features
            ffn_input = normed + memory_augmented

            memory_info = {
                'retrieved': retrieved
            } if return_memory_info else None
        else:
            ffn_input = normed
            memory_info = None

        # Feedforward with residual
        ffn_out = self.ffn(ffn_input)
        output = x + ffn_out

        # Post-norm
        output = self.norm2(output)

        # Update memory
        if self.use_memory and self.cms is not None and step is not None:
            # Store current representations in memory
            keys = output.reshape(-1, hidden_dim)
            values = output.reshape(-1, hidden_dim)
            self.cms.store(keys, values, step=step)

        return output, memory_info

    def get_memory_stats(self) -> Optional[Dict]:
        """
        Get statistics about memory usage.

        Returns:
            Dict with memory statistics or None if no CMS
        """
        if self.cms is None:
            return None

        stats = {}
        for level, bank in enumerate(self.cms.memory_banks):
            capacity = bank.memory_size
            current_size = bank.memory_count.item() if hasattr(bank.memory_count, 'item') else bank.memory_count
            stats[f'level_{level}'] = {
                'capacity': capacity,
                'current_size': current_size,
                'utilization': current_size / capacity if capacity > 0 else 0
            }

        return stats

    def reset_memory(self):
        """Reset all memory banks."""
        if self.cms is not None:
            for bank in self.cms.memory_banks:
                # Reset by setting memory_count to 0
                bank.memory_count.fill_(0)


class CMSAttentionBlock(nn.Module):
    """
    Attention block with CMS integration.

    Combines self-attention with memory-augmented computation.

    Args:
        hidden_dim: Dimension of hidden states
        num_heads: Number of attention heads
        ffn_dim: Dimension of feedforward network
        memory_config: Configuration for CMS
        dropout: Dropout probability

    Example:
        >>> block = CMSAttentionBlock(
        ...     hidden_dim=128,
        ...     num_heads=8,
        ...     memory_config={0: {'capacity': 100}}
        ... )
        >>> output, _ = block(input_tensor, step=0)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        memory_config: Optional[Dict[int, Dict]] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or (4 * hidden_dim)

        # Multi-head attention
        # Attention weights update fast for quick adaptation
        self.attn_norm = NestedLayerNorm(
            hidden_dim,
            frequency_level=FrequencyLevel.FAST
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # CMS block for feedforward with memory
        self.cms_block = CMSBlock(
            hidden_dim=hidden_dim,
            ffn_dim=self.ffn_dim,
            memory_config=memory_config,
            use_memory=(memory_config is not None),
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        step: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_memory_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with attention and memory.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            step: Current training step
            attn_mask: Optional attention mask
            return_memory_info: If True, returns memory info

        Returns:
            output: Output tensor
            memory_info: Optional memory information
        """
        # Self-attention with residual
        normed = self.attn_norm(x)
        attn_out, _ = self.attention(
            normed, normed, normed,
            attn_mask=attn_mask
        )
        x = x + attn_out

        # CMS feedforward block
        output, memory_info = self.cms_block(
            x,
            step=step,
            return_memory_info=return_memory_info
        )

        return output, memory_info

    def get_memory_stats(self) -> Optional[Dict]:
        """Get memory statistics from CMS block."""
        return self.cms_block.get_memory_stats()

    def reset_memory(self):
        """Reset memory in CMS block."""
        self.cms_block.reset_memory()
