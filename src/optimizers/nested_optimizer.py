"""
Nested Optimizer Wrapper

Coordinates multiple optimizers operating at different frequencies for
hierarchical nested optimization.
"""

import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional, Any, Union
from ..utils.frequency_scheduler import FrequencyScheduler


class NestedOptimizer:
    """
    Wrapper that coordinates multiple optimizers at different update frequencies.

    This implements the nested optimization hierarchy where different parameter
    groups are updated at different rates:
    - Fast parameters (level 0): Update every step
    - Medium parameters (level 1): Update every N steps
    - Slow parameters (level 2): Update every M steps

    Args:
        optimizers: List of optimizers, one per level
        frequency_scheduler: FrequencyScheduler for update coordination
        param_groups: Optional list of parameter group assignments

    Example:
        >>> # Create model with different parameter groups
        >>> model = MyModel()
        >>> fast_params = model.attention.parameters()
        >>> slow_params = model.embeddings.parameters()
        >>>
        >>> # Create optimizers for each level
        >>> opt_fast = torch.optim.Adam(fast_params, lr=0.001)
        >>> opt_slow = torch.optim.SGD(slow_params, lr=0.0001)
        >>>
        >>> # Create frequency scheduler
        >>> scheduler = FrequencyScheduler(num_levels=2, frequencies=[1, 10])
        >>>
        >>> # Wrap in NestedOptimizer
        >>> nested_opt = NestedOptimizer(
        ...     optimizers=[opt_fast, opt_slow],
        ...     frequency_scheduler=scheduler
        ... )
        >>>
        >>> # Training loop
        >>> for step in range(1000):
        ...     loss = model(x).sum()
        ...     loss.backward()
        ...     nested_opt.step(step)  # Automatically updates appropriate levels
        ...     nested_opt.zero_grad()
    """

    def __init__(
        self,
        optimizers: List[Optimizer],
        frequency_scheduler: FrequencyScheduler,
        param_groups: Optional[List[Dict]] = None
    ):
        assert len(optimizers) == frequency_scheduler.num_levels, \
            f"Number of optimizers ({len(optimizers)}) must match number of levels ({frequency_scheduler.num_levels})"

        self.optimizers = optimizers
        self.frequency_scheduler = frequency_scheduler
        self.param_groups = param_groups or []
        self.num_levels = len(optimizers)

        # Track which levels updated
        self.update_history: Dict[int, List[int]] = {i: [] for i in range(self.num_levels)}
        self.current_step = 0

    def step(self, step: Optional[int] = None, closure=None) -> Dict[int, bool]:
        """
        Perform optimization step for levels that should update.

        Args:
            step: Current training step (if None, uses internal counter)
            closure: Optional closure for evaluating loss

        Returns:
            Dictionary mapping level -> whether it was updated

        Example:
            >>> updated = nested_opt.step(step=100)
            >>> # updated = {0: True, 1: True, 2: False}
            >>> # Levels 0 and 1 were updated, level 2 was not
        """
        if step is None:
            step = self.current_step

        # Determine which levels should update
        active_levels = self.frequency_scheduler.get_active_levels(step)

        updated = {}
        for level in range(self.num_levels):
            if level in active_levels:
                # Update this level
                self.optimizers[level].step(closure=closure)
                self.update_history[level].append(step)
                updated[level] = True
            else:
                updated[level] = False

        self.current_step += 1
        return updated

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Zero gradients for all optimizers.

        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def get_lr(self, level: Optional[int] = None) -> Union[float, List[float]]:
        """
        Get learning rate(s).

        Args:
            level: Specific level (None = all levels)

        Returns:
            Learning rate or list of learning rates
        """
        if level is not None:
            return self.optimizers[level].param_groups[0]['lr']
        else:
            return [opt.param_groups[0]['lr'] for opt in self.optimizers]

    def set_lr(self, lr: Union[float, List[float]], level: Optional[int] = None) -> None:
        """
        Set learning rate(s).

        Args:
            lr: Learning rate or list of learning rates
            level: Specific level (None = all levels)
        """
        if level is not None:
            for param_group in self.optimizers[level].param_groups:
                param_group['lr'] = lr
        else:
            if isinstance(lr, (int, float)):
                lr = [lr] * self.num_levels

            for level_idx, level_lr in enumerate(lr):
                for param_group in self.optimizers[level_idx].param_groups:
                    param_group['lr'] = level_lr

    def get_update_history(self) -> Dict[int, List[int]]:
        """
        Get history of when each level was updated.

        Returns:
            Dictionary mapping level -> list of steps when it was updated
        """
        return self.update_history.copy()

    def get_update_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics about updates for each level.

        Returns:
            Dictionary with update statistics per level
        """
        stats = {}
        for level in range(self.num_levels):
            history = self.update_history[level]
            stats[level] = {
                'num_updates': len(history),
                'frequency': self.frequency_scheduler.get_frequency(level),
                'last_update': history[-1] if history else None,
                'update_rate': len(history) / max(1, self.current_step)
            }
        return stats

    def state_dict(self) -> Dict:
        """
        Get state dictionary for all optimizers.

        Returns:
            Dictionary with state for all levels
        """
        return {
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'frequency_scheduler': {
                'num_levels': self.frequency_scheduler.num_levels,
                'frequencies': self.frequency_scheduler.get_all_frequencies(),
                'scaling': self.frequency_scheduler.scaling
            },
            'update_history': self.update_history,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load state dictionary for all optimizers.

        Args:
            state_dict: State dictionary to load
        """
        # Load optimizer states
        for level, opt_state in enumerate(state_dict['optimizers']):
            self.optimizers[level].load_state_dict(opt_state)

        # Load update history
        if 'update_history' in state_dict:
            self.update_history = state_dict['update_history']

        if 'current_step' in state_dict:
            self.current_step = state_dict['current_step']

    def __repr__(self) -> str:
        """String representation."""
        opt_types = [type(opt).__name__ for opt in self.optimizers]
        freqs = self.frequency_scheduler.get_all_frequencies()
        return (f"NestedOptimizer(\n"
                f"  num_levels={self.num_levels},\n"
                f"  optimizers={opt_types},\n"
                f"  frequencies={freqs},\n"
                f"  current_step={self.current_step}\n"
                f")")


class NestedOptimizerBuilder:
    """
    Helper class for building NestedOptimizer with different parameter groups.

    Args:
        model: PyTorch model
        num_levels: Number of optimization levels

    Example:
        >>> builder = NestedOptimizerBuilder(model, num_levels=3)
        >>>
        >>> # Assign parameters to levels
        >>> builder.add_params(model.attention.parameters(), level=0)  # Fast
        >>> builder.add_params(model.feedforward.parameters(), level=1)  # Medium
        >>> builder.add_params(model.embeddings.parameters(), level=2)  # Slow
        >>>
        >>> # Build with different optimizers per level
        >>> nested_opt = builder.build(
        ...     optimizer_types=['adam', 'sgd', 'sgd'],
        ...     learning_rates=[0.001, 0.0001, 0.00001],
        ...     frequencies=[1, 10, 100]
        ... )
    """

    def __init__(self, model: torch.nn.Module, num_levels: int):
        self.model = model
        self.num_levels = num_levels
        self.param_groups: List[List] = [[] for _ in range(num_levels)]

    def add_params(self, params, level: int) -> None:
        """
        Add parameters to a specific level.

        Args:
            params: Parameters to add
            level: Level to assign parameters to
        """
        assert 0 <= level < self.num_levels, f"Level {level} out of range"
        self.param_groups[level].extend(list(params))

    def auto_assign_params(self, strategy: str = 'uniform') -> None:
        """
        Automatically assign parameters to levels.

        Args:
            strategy: Assignment strategy:
                - 'uniform': Distribute parameters uniformly
                - 'by_name': Assign based on parameter name patterns
                - 'by_size': Fast updates for small params, slow for large

        Example:
            >>> builder.auto_assign_params(strategy='by_name')
            >>> # Attention/norm params → fast (level 0)
            >>> # FFN params → medium (level 1)
            >>> # Embeddings → slow (level 2)
        """
        all_params = list(self.model.parameters())

        if strategy == 'uniform':
            # Distribute uniformly across levels
            params_per_level = len(all_params) // self.num_levels
            for level in range(self.num_levels):
                start = level * params_per_level
                end = start + params_per_level if level < self.num_levels - 1 else len(all_params)
                self.param_groups[level] = all_params[start:end]

        elif strategy == 'by_name':
            # Assign based on parameter names
            for name, param in self.model.named_parameters():
                if any(keyword in name.lower() for keyword in ['attention', 'norm', 'ln']):
                    level = 0  # Fast
                elif any(keyword in name.lower() for keyword in ['embed', 'position']):
                    level = self.num_levels - 1  # Slow
                else:
                    level = 1  # Medium (if 3+ levels, otherwise slow)
                    if self.num_levels == 2:
                        level = 1

                self.param_groups[level].append(param)

        elif strategy == 'by_size':
            # Fast updates for small params, slow for large
            param_sizes = [(param, param.numel()) for param in all_params]
            param_sizes.sort(key=lambda x: x[1])

            params_per_level = len(param_sizes) // self.num_levels
            for level in range(self.num_levels):
                start = level * params_per_level
                end = start + params_per_level if level < self.num_levels - 1 else len(param_sizes)
                self.param_groups[level] = [p for p, _ in param_sizes[start:end]]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def build(
        self,
        optimizer_types: Union[str, List[str]] = 'adam',
        learning_rates: Union[float, List[float]] = 0.001,
        frequencies: Optional[List[int]] = None,
        **optimizer_kwargs
    ) -> NestedOptimizer:
        """
        Build NestedOptimizer.

        Args:
            optimizer_types: Optimizer type(s) for each level
            learning_rates: Learning rate(s) for each level
            frequencies: Update frequencies for each level
            **optimizer_kwargs: Additional optimizer arguments

        Returns:
            Configured NestedOptimizer
        """
        # Handle single value inputs
        if isinstance(optimizer_types, str):
            optimizer_types = [optimizer_types] * self.num_levels
        if isinstance(learning_rates, (int, float)):
            learning_rates = [learning_rates] * self.num_levels

        assert len(optimizer_types) == self.num_levels
        assert len(learning_rates) == self.num_levels

        # Create optimizers for each level
        optimizers = []
        for level in range(self.num_levels):
            params = self.param_groups[level]
            if not params:
                raise ValueError(f"No parameters assigned to level {level}")

            opt_type = optimizer_types[level].lower()
            lr = learning_rates[level]

            if opt_type == 'adam':
                opt = torch.optim.Adam(params, lr=lr, **optimizer_kwargs)
            elif opt_type == 'sgd':
                opt = torch.optim.SGD(params, lr=lr, **optimizer_kwargs)
            elif opt_type == 'adamw':
                opt = torch.optim.AdamW(params, lr=lr, **optimizer_kwargs)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")

            optimizers.append(opt)

        # Create frequency scheduler
        if frequencies is None:
            # Default exponential scaling
            scheduler = FrequencyScheduler(
                num_levels=self.num_levels,
                scaling='exponential',
                ratio=10
            )
        else:
            scheduler = FrequencyScheduler(
                num_levels=self.num_levels,
                scaling='manual',
                manual_frequencies=frequencies
            )

        return NestedOptimizer(optimizers, scheduler)
