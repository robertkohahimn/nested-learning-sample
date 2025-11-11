"""
Continuum Memory System (CMS) for Nested Learning

Implements a spectrum of memory modules, each updating at different frequencies.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from .associative_memory import AssociativeMemory
from ..utils.frequency_scheduler import FrequencyScheduler


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System with multi-frequency memory banks.

    Instead of binary short/long-term memory, CMS treats memory as a spectrum
    where different banks update at different frequencies:
    - Short-term (high frequency): C=1, updates every step
    - Mid-term (medium frequency): C=10-100, updates every 10-100 steps
    - Long-term (low frequency): C=1000+, updates every 1000+ steps

    This mimics neuroplasticity where different memory systems consolidate
    information at different time scales.

    Args:
        memory_sizes: List of memory sizes for each level (or single int for all)
        key_dim: Dimension of memory keys
        value_dim: Dimension of memory values
        num_levels: Number of frequency levels (default: 3)
        frequencies: List of update frequencies for each level (or use scheduler)
        frequency_scheduler: Optional FrequencyScheduler instance
        temperature: Temperature for attention (default: 1.0)
        consolidation_rate: Rate for memory consolidation between levels (default: 0.1)

    Example:
        >>> # Create CMS with 3 levels: fast (C=1), medium (C=10), slow (C=100)
        >>> cms = ContinuumMemorySystem(
        ...     memory_sizes=[100, 200, 500],
        ...     key_dim=64,
        ...     value_dim=128,
        ...     frequencies=[1, 10, 100]
        ... )
        >>> # Store information
        >>> keys = torch.randn(10, 64)
        >>> values = torch.randn(10, 128)
        >>> cms.store(keys, values, step=50)
        >>> # Retrieve information
        >>> query = torch.randn(5, 64)
        >>> retrieved = cms.retrieve(query)  # Aggregates across all levels
    """

    def __init__(
        self,
        memory_sizes: List[int] | int,
        key_dim: int,
        value_dim: int,
        num_levels: int = 3,
        frequencies: Optional[List[int]] = None,
        frequency_scheduler: Optional[FrequencyScheduler] = None,
        temperature: float = 1.0,
        consolidation_rate: float = 0.1
    ):
        super().__init__()

        # Handle memory_sizes
        if isinstance(memory_sizes, int):
            memory_sizes = [memory_sizes] * num_levels
        assert len(memory_sizes) == num_levels, \
            f"memory_sizes length ({len(memory_sizes)}) must match num_levels ({num_levels})"

        self.num_levels = num_levels
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.memory_sizes = memory_sizes
        self.consolidation_rate = consolidation_rate

        # Setup frequency scheduler
        if frequency_scheduler is not None:
            self.frequency_scheduler = frequency_scheduler
        elif frequencies is not None:
            self.frequency_scheduler = FrequencyScheduler(
                num_levels=num_levels,
                scaling='manual',
                manual_frequencies=frequencies
            )
        else:
            # Default: exponential scaling with ratio=10
            self.frequency_scheduler = FrequencyScheduler(
                num_levels=num_levels,
                scaling='exponential',
                ratio=10.0
            )

        # Create memory banks for each level
        self.memory_banks = nn.ModuleList([
            AssociativeMemory(
                memory_size=memory_sizes[i],
                key_dim=key_dim,
                value_dim=value_dim,
                temperature=temperature
            )
            for i in range(num_levels)
        ])

        # Learnable weights for aggregating across memory levels
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

        # Track current step for frequency checking
        self.register_buffer('current_step', torch.tensor(0, dtype=torch.long))

    def store(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        step: Optional[int] = None,
        force_levels: Optional[List[int]] = None
    ) -> Dict[int, bool]:
        """
        Store key-value pairs in appropriate memory banks based on frequency.

        Args:
            keys: Keys to store of shape (batch_size, key_dim)
            values: Values to store of shape (batch_size, value_dim)
            step: Current training step (if None, uses internal counter)
            force_levels: Force storage in specific levels (overrides frequency check)

        Returns:
            Dictionary mapping level -> whether it was updated
        """
        if step is None:
            step = self.current_step.item()

        updated = {}

        if force_levels is not None:
            levels_to_update = force_levels
        else:
            # Check which levels should update at this step
            levels_to_update = self.frequency_scheduler.get_active_levels(step)

        for level in range(self.num_levels):
            if level in levels_to_update:
                # Update this memory bank
                self.memory_banks[level].write(keys, values, mode='replace')
                updated[level] = True
            else:
                updated[level] = False

        return updated

    def retrieve(
        self,
        query: torch.Tensor,
        levels: Optional[List[int]] = None,
        aggregate: bool = True
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Retrieve values from memory based on query.

        Args:
            query: Query tensor of shape (batch_size, key_dim)
            levels: Specific levels to retrieve from (None = all levels)
            aggregate: If True, aggregate across levels; else return list

        Returns:
            Retrieved values of shape (batch_size, value_dim) if aggregate=True,
            otherwise list of retrieved values from each level
        """
        if levels is None:
            levels = list(range(self.num_levels))

        # Retrieve from each level
        retrieved_per_level = []
        for level in levels:
            retrieved = self.memory_banks[level].read(query)
            retrieved_per_level.append(retrieved)

        if not aggregate:
            return retrieved_per_level

        # Aggregate across levels using learned weights
        # Only use weights for the requested levels
        active_weights = self.level_weights[levels]
        active_weights = torch.softmax(active_weights, dim=0)

        # Weighted sum: (num_levels, batch_size, value_dim) -> (batch_size, value_dim)
        stacked = torch.stack(retrieved_per_level, dim=0)  # (num_levels, batch_size, value_dim)
        aggregated = torch.einsum('l,lbv->bv', active_weights, stacked)

        return aggregated

    def consolidate(
        self,
        from_level: int,
        to_level: int,
        num_samples: Optional[int] = None
    ) -> None:
        """
        Consolidate memories from one level to another.

        This mimics memory consolidation in biological systems where
        short-term memories are gradually transferred to long-term storage.

        Args:
            from_level: Source memory level (typically fast/short-term)
            to_level: Target memory level (typically slow/long-term)
            num_samples: Number of samples to consolidate (None = all)
        """
        assert 0 <= from_level < self.num_levels, f"Invalid from_level: {from_level}"
        assert 0 <= to_level < self.num_levels, f"Invalid to_level: {to_level}"

        source_bank = self.memory_banks[from_level]
        target_bank = self.memory_banks[to_level]

        # Get memories from source
        source_count = source_bank.memory_count.item()
        if source_count == 0:
            return  # Nothing to consolidate

        if num_samples is None or num_samples > source_count:
            num_samples = source_count

        # Sample keys and values from source
        source_keys = source_bank.keys[:num_samples]
        source_values = source_bank.values[:num_samples]

        # Write to target with consolidation rate
        # Gradual transfer: blend with existing memories
        if target_bank.memory_count > 0:
            # Update existing memories
            target_bank.update(
                source_keys,
                source_values,
                learning_rate=self.consolidation_rate
            )
        else:
            # First consolidation, just write
            target_bank.write(source_keys, source_values, mode='append')

    def auto_consolidate(self, step: Optional[int] = None) -> None:
        """
        Automatically consolidate memories from fast to slow levels.

        Consolidation happens when slower levels are about to update.

        Args:
            step: Current training step
        """
        if step is None:
            step = self.current_step.item()

        # Consolidate from level i to level i+1 when level i+1 updates
        for level in range(self.num_levels - 1):
            next_level = level + 1
            if self.frequency_scheduler.should_update(next_level, step):
                # Consolidate from current level to next (slower) level
                self.consolidate(from_level=level, to_level=next_level)

    def clear(self, levels: Optional[List[int]] = None) -> None:
        """
        Clear memory banks.

        Args:
            levels: Specific levels to clear (None = all)
        """
        if levels is None:
            levels = list(range(self.num_levels))

        for level in levels:
            self.memory_banks[level].clear()

    def step(self) -> None:
        """Increment internal step counter."""
        self.current_step += 1

    def get_memory_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get statistics for all memory banks.

        Returns:
            Dictionary mapping level -> stats dict
        """
        stats = {}
        for level in range(self.num_levels):
            bank = self.memory_banks[level]
            stats[level] = {
                'usage': bank.get_memory_usage(),
                'count': bank.memory_count.item(),
                'capacity': bank.memory_size,
                'frequency': self.frequency_scheduler.get_frequency(level)
            }
        return stats

    def forward(
        self,
        query: torch.Tensor,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        mode: str = 'retrieve',
        step: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass supporting both storage and retrieval.

        Args:
            query: Query tensor for retrieval
            keys: Keys to store (for 'store' mode)
            values: Values to store (for 'store' mode)
            mode: 'retrieve', 'store', or 'consolidate'
            step: Current training step

        Returns:
            Tuple of (retrieved_values, metadata)
        """
        if mode == 'retrieve':
            retrieved = self.retrieve(query)
            return retrieved, None

        elif mode == 'store':
            assert keys is not None and values is not None, \
                "keys and values required for store mode"
            updated = self.store(keys, values, step=step)
            # Auto-consolidate if needed
            self.auto_consolidate(step=step)
            self.step()
            return None, {'updated_levels': updated}

        elif mode == 'consolidate':
            self.auto_consolidate(step=step)
            return None, None

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __repr__(self) -> str:
        """String representation."""
        freq_str = ", ".join([
            f"L{i}(f={self.frequency_scheduler.get_frequency(i)}, "
            f"size={self.memory_sizes[i]})"
            for i in range(self.num_levels)
        ])
        return f"ContinuumMemorySystem(num_levels={self.num_levels}, [{freq_str}])"
