"""
Associative Memory for Nested Learning

Implements memory storage and retrieval using L2 regression loss
instead of traditional dot-product similarity.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AssociativeMemory(nn.Module):
    """
    Associative memory module using L2 regression for key-value storage.

    Traditional attention uses dot-product similarity: attention(Q, K) = softmax(Q @ K^T)
    This module uses L2 regression loss for more robust retrieval.

    The memory stores key-value pairs and retrieves values based on query similarity
    to keys, computed via L2 distance rather than dot products.

    Args:
        memory_size: Maximum number of memory slots
        key_dim: Dimension of keys
        value_dim: Dimension of values
        temperature: Temperature for softmax attention (default: 1.0)
        l2_weight: Weight for L2 regularization (default: 0.01)

    Example:
        >>> memory = AssociativeMemory(memory_size=100, key_dim=64, value_dim=128)
        >>> # Store key-value pairs
        >>> keys = torch.randn(10, 64)
        >>> values = torch.randn(10, 128)
        >>> memory.write(keys, values)
        >>> # Retrieve based on query
        >>> query = torch.randn(5, 64)
        >>> retrieved_values = memory.read(query)  # Shape: (5, 128)
    """

    def __init__(
        self,
        memory_size: int,
        key_dim: int,
        value_dim: int,
        temperature: float = 1.0,
        l2_weight: float = 0.01
    ):
        super().__init__()

        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.temperature = temperature
        self.l2_weight = l2_weight

        # Initialize memory banks
        # Using buffers so they're saved with model state but not trained as parameters
        self.register_buffer('keys', torch.zeros(memory_size, key_dim))
        self.register_buffer('values', torch.zeros(memory_size, value_dim))
        self.register_buffer('memory_count', torch.tensor(0, dtype=torch.long))

    def _compute_similarity(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between query and keys using L2 distance.

        Args:
            query: Query tensor of shape (batch_size, key_dim)
            keys: Key tensor of shape (memory_size, key_dim)

        Returns:
            Similarity scores of shape (batch_size, memory_size)
        """
        # Compute L2 distance: ||q - k||^2 = ||q||^2 + ||k||^2 - 2 * q @ k^T
        query_norm = (query ** 2).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        keys_norm = (keys ** 2).sum(dim=-1, keepdim=False)   # (memory_size,)
        dot_product = query @ keys.T                          # (batch_size, memory_size)

        l2_distance = query_norm + keys_norm - 2 * dot_product  # (batch_size, memory_size)

        # Convert distance to similarity (negative distance, scaled by temperature)
        similarity = -l2_distance / self.temperature

        return similarity

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from memory based on query.

        Uses L2-based attention to retrieve weighted combination of values.

        Args:
            query: Query tensor of shape (batch_size, key_dim)

        Returns:
            Retrieved values of shape (batch_size, value_dim)
        """
        batch_size = query.shape[0]

        if self.memory_count == 0:
            # No memory stored yet, return zeros
            return torch.zeros(batch_size, self.value_dim, device=query.device)

        # Get active memory (only filled slots)
        active_keys = self.keys[:self.memory_count]
        active_values = self.values[:self.memory_count]

        # Compute attention weights using L2 similarity
        similarity = self._compute_similarity(query, active_keys)
        attention_weights = torch.softmax(similarity, dim=-1)  # (batch_size, memory_count)

        # Weighted sum of values
        retrieved = attention_weights @ active_values  # (batch_size, value_dim)

        return retrieved

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        mode: str = 'append'
    ) -> None:
        """
        Write key-value pairs to memory.

        Args:
            keys: Key tensor of shape (num_items, key_dim)
            values: Value tensor of shape (num_items, value_dim)
            mode: Writing mode - 'append' (add new) or 'replace' (replace oldest)

        Raises:
            ValueError: If mode is invalid
        """
        num_items = keys.shape[0]
        assert keys.shape == (num_items, self.key_dim), \
            f"Expected keys shape ({num_items}, {self.key_dim}), got {keys.shape}"
        assert values.shape == (num_items, self.value_dim), \
            f"Expected values shape ({num_items}, {self.value_dim}), got {values.shape}"
        assert mode in ['append', 'replace'], \
            f"mode must be 'append' or 'replace', got {mode}"

        if mode == 'append':
            # Add new items to memory
            start_idx = self.memory_count.item()
            end_idx = min(start_idx + num_items, self.memory_size)
            num_to_write = end_idx - start_idx

            self.keys[start_idx:end_idx] = keys[:num_to_write]
            self.values[start_idx:end_idx] = values[:num_to_write]
            self.memory_count = torch.tensor(end_idx, dtype=torch.long)

        elif mode == 'replace':
            # Replace oldest items (circular buffer)
            for i in range(num_items):
                idx = (self.memory_count.item() + i) % self.memory_size
                self.keys[idx] = keys[i]
                self.values[idx] = values[i]

            self.memory_count = torch.tensor(
                min(self.memory_count.item() + num_items, self.memory_size),
                dtype=torch.long
            )

    def update(
        self,
        query: torch.Tensor,
        target_values: torch.Tensor,
        learning_rate: float = 0.1
    ) -> torch.Tensor:
        """
        Update memory based on L2 regression loss.

        This implements the associative memory update rule from the paper:
        M_{t+1} = M_t + η * (V - M_t @ Q^T) @ Q

        Args:
            query: Query tensor of shape (batch_size, key_dim)
            target_values: Target values to store of shape (batch_size, value_dim)
            learning_rate: Update learning rate (η)

        Returns:
            L2 regression loss
        """
        batch_size = query.shape[0]

        if self.memory_count == 0:
            # Initialize memory with first batch
            self.write(query, target_values, mode='append')
            return torch.tensor(0.0, device=query.device)

        # Get active memory
        active_keys = self.keys[:self.memory_count]
        active_values = self.values[:self.memory_count]

        # Compute attention weights
        similarity = self._compute_similarity(query, active_keys)
        attention_weights = torch.softmax(similarity, dim=-1)  # (batch_size, memory_count)

        # Retrieved values
        retrieved = attention_weights @ active_values  # (batch_size, value_dim)

        # L2 regression loss
        loss = torch.mean((retrieved - target_values) ** 2)

        # Update memory: M_{t+1} = M_t + η * (V - retrieved) @ attention_weights^T
        with torch.no_grad():
            residual = target_values - retrieved  # (batch_size, value_dim)
            update = residual.T @ attention_weights  # (value_dim, memory_count)
            self.values[:self.memory_count] += learning_rate * update.T

        return loss

    def clear(self) -> None:
        """Clear all memory."""
        self.keys.zero_()
        self.values.zero_()
        self.memory_count.zero_()

    def is_full(self) -> bool:
        """Check if memory is full."""
        return self.memory_count >= self.memory_size

    def get_memory_usage(self) -> float:
        """Get memory usage as fraction of capacity."""
        return self.memory_count.item() / self.memory_size

    def forward(
        self,
        query: torch.Tensor,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        mode: str = 'read'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass supporting both read and write operations.

        Args:
            query: Query tensor for reading/writing
            keys: Keys to write (only for 'write' mode)
            values: Values to write (only for 'write' mode)
            mode: 'read', 'write', or 'update'

        Returns:
            Tuple of (retrieved_values, loss)
            - For 'read': (retrieved_values, None)
            - For 'write': (None, None)
            - For 'update': (retrieved_values, l2_loss)
        """
        if mode == 'read':
            retrieved = self.read(query)
            return retrieved, None

        elif mode == 'write':
            assert keys is not None and values is not None, \
                "keys and values must be provided for write mode"
            self.write(keys, values)
            return None, None

        elif mode == 'update':
            assert values is not None, \
                "values must be provided for update mode"
            loss = self.update(query, values)
            retrieved = self.read(query)
            return retrieved, loss

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'read', 'write', or 'update'")

    def __repr__(self) -> str:
        """String representation."""
        return (f"AssociativeMemory(memory_size={self.memory_size}, "
                f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
                f"usage={self.memory_count}/{self.memory_size})")
