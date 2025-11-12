"""
Hope: Simplified Hope architecture for Nested Learning.

Based on concepts from the Nested Learning paper with:
- CMS blocks for memory
- Multi-frequency parameter updates
- Recursive optimization
- Extended context processing

Note: This is a simplified demonstration, not the full architecture from the paper.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from ..layers import (
    CMSBlock,
    CMSAttentionBlock,
    NestedLinear,
    NestedEmbedding,
    NestedLayerNorm,
    FrequencyLevel
)


class HopeModel(nn.Module):
    """
    Simplified Hope architecture demonstrating Nested Learning principles.

    This model combines:
    - Multi-level memory (CMS) for different time scales
    - Multi-frequency parameter updates
    - Recurrent processing for extended context
    - Self-attention for long-range dependencies

    The architecture is designed to handle long-context tasks and
    continual learning scenarios.

    Args:
        vocab_size: Size of vocabulary (for embeddings)
        hidden_dim: Dimension of hidden states
        num_layers: Number of CMS blocks
        num_heads: Number of attention heads
        ffn_dim: Dimension of feedforward networks
        memory_levels: Number of memory levels in CMS
        memory_capacities: List of capacities for each memory level
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_attention: If True, uses attention blocks instead of simple CMS blocks

    Example:
        >>> model = HopeModel(
        ...     vocab_size=10000,
        ...     hidden_dim=256,
        ...     num_layers=6,
        ...     memory_levels=3,
        ...     memory_capacities=[100, 50, 25]
        ... )
        >>> output = model(input_ids, step=0)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        memory_levels: int = 3,
        memory_capacities: Optional[List[int]] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.memory_levels = memory_levels
        self.max_seq_len = max_seq_len
        self.use_attention = use_attention

        # Default memory capacities (decreasing with level)
        if memory_capacities is None:
            memory_capacities = [max(100 // (2 ** i), 10) for i in range(memory_levels)]

        # Create memory configuration
        memory_config = {
            i: {'capacity': capacity}
            for i, capacity in enumerate(memory_capacities[:memory_levels])
        }

        # Token embeddings (slow updates for stable representations)
        self.token_embedding = NestedEmbedding(
            vocab_size,
            hidden_dim,
            frequency_level=FrequencyLevel.SLOW
        )

        # Positional embeddings (slow updates)
        self.pos_embedding = NestedEmbedding(
            max_seq_len,
            hidden_dim,
            frequency_level=FrequencyLevel.SLOW
        )

        # Input projection and normalization
        self.input_norm = NestedLayerNorm(
            hidden_dim,
            frequency_level=FrequencyLevel.FAST
        )

        # CMS/Attention blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if use_attention:
                block = CMSAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    memory_config=memory_config,
                    dropout=dropout
                )
            else:
                block = CMSBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    memory_config=memory_config,
                    use_memory=True,
                    dropout=dropout
                )
            self.blocks.append(block)

        # Output layer
        self.output_norm = NestedLayerNorm(
            hidden_dim,
            frequency_level=FrequencyLevel.FAST
        )

        self.output_proj = NestedLinear(
            hidden_dim,
            vocab_size,
            parameter_frequencies={
                'weight': FrequencyLevel.FAST,  # Quick task adaptation
                'bias': FrequencyLevel.FAST
            }
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        step: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_memory_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """
        Forward pass through Hope model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            step: Current training step (for CMS updates)
            attention_mask: Optional attention mask
            return_memory_info: If True, returns memory information from all blocks

        Returns:
            output: Logits of shape (batch_size, seq_len, vocab_size)
            memory_info: Optional list of memory info dicts from each block
        """
        batch_size, seq_len = input_ids.shape

        # Ensure sequence length doesn't exceed maximum
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        # Create position IDs
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch_size, -1)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(pos_ids)

        # Combine embeddings
        h = token_emb + pos_emb
        h = self.dropout(h)
        h = self.input_norm(h)

        # Process through blocks
        memory_infos = [] if return_memory_info else None

        for block in self.blocks:
            if self.use_attention:
                h, mem_info = block(
                    h,
                    step=step,
                    attn_mask=attention_mask,
                    return_memory_info=return_memory_info
                )
            else:
                h, mem_info = block(
                    h,
                    step=step,
                    return_memory_info=return_memory_info
                )

            if return_memory_info:
                memory_infos.append(mem_info)

        # Output projection
        h = self.output_norm(h)
        logits = self.output_proj(h)

        return logits, memory_infos

    def get_parameters_by_frequency(self) -> Dict[int, List[nn.Parameter]]:
        """
        Get all parameters grouped by their update frequency.

        Returns:
            Dict mapping frequency level to list of parameters
        """
        from ..layers import get_frequency_aware_param_groups
        return get_frequency_aware_param_groups(self)

    def get_memory_stats(self) -> List[Dict]:
        """
        Get memory statistics from all CMS blocks.

        Returns:
            List of dicts containing memory stats for each block
        """
        stats = []
        for i, block in enumerate(self.blocks):
            block_stats = block.get_memory_stats()
            if block_stats is not None:
                stats.append({
                    'block_idx': i,
                    'memory': block_stats
                })
        return stats

    def reset_memory(self):
        """Reset memory in all CMS blocks."""
        for block in self.blocks:
            block.reset_memory()

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from top k tokens
            step: Current step (for CMS)

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = self.forward(generated, step=step)

                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature

                # Optional top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check if we exceed max length
                if generated.shape[1] >= self.max_seq_len:
                    break

        return generated


class HopeForClassification(nn.Module):
    """
    Hope model adapted for classification tasks.

    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of output classes
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        memory_levels: Number of memory levels
        memory_capacities: Capacities for memory levels
        max_seq_len: Maximum sequence length
        pooling: Pooling strategy ('mean', 'max', 'first', 'last')

    Example:
        >>> model = HopeForClassification(
        ...     vocab_size=10000,
        ...     num_classes=10,
        ...     hidden_dim=256,
        ...     num_layers=4
        ... )
        >>> logits = model(input_ids, step=0)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        memory_levels: int = 2,
        memory_capacities: Optional[List[int]] = None,
        max_seq_len: int = 512,
        pooling: str = 'mean',
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pooling = pooling

        # Base Hope model
        self.hope = HopeModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            memory_levels=memory_levels,
            memory_capacities=memory_capacities,
            max_seq_len=max_seq_len,
            **kwargs
        )

        # Classification head (fast updates for task-specific adaptation)
        self.classifier = nn.Sequential(
            NestedLayerNorm(hidden_dim, frequency_level=FrequencyLevel.FAST),
            NestedLinear(
                hidden_dim,
                num_classes,
                parameter_frequencies={
                    'weight': FrequencyLevel.FAST,
                    'bias': FrequencyLevel.FAST
                }
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        step: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            input_ids: Input token IDs
            step: Current training step
            attention_mask: Optional attention mask

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Get hidden states (logits from base model)
        logits, _ = self.hope(input_ids, step=step, attention_mask=attention_mask)

        # Pool sequence representations
        if self.pooling == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(logits.size())
                sum_embeddings = torch.sum(logits * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = logits.mean(dim=1)
        elif self.pooling == 'max':
            pooled = logits.max(dim=1)[0]
        elif self.pooling == 'first':
            pooled = logits[:, 0, :]
        elif self.pooling == 'last':
            if attention_mask is not None:
                # Get last non-padded position
                seq_lengths = attention_mask.sum(dim=1) - 1
                pooled = logits[torch.arange(logits.size(0)), seq_lengths, :]
            else:
                pooled = logits[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Classification
        class_logits = self.classifier(pooled)

        return class_logits

    def get_parameters_by_frequency(self) -> Dict[int, List[nn.Parameter]]:
        """Get parameters by frequency."""
        from ..layers import get_frequency_aware_param_groups
        return get_frequency_aware_param_groups(self)

    def get_memory_stats(self) -> List[Dict]:
        """Get memory statistics."""
        return self.hope.get_memory_stats()

    def reset_memory(self):
        """Reset memory."""
        self.hope.reset_memory()
