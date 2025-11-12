"""
Base optimizer interface for Nested Learning optimizers.
"""

import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional, Callable, Any


class BaseNestedOptimizer(Optimizer):
    """
    Base class for Nested Learning optimizers.

    Extends PyTorch's Optimizer with support for:
    - Multi-frequency parameter updates
    - Meta-learning capabilities
    - Integration with FrequencyScheduler

    Args:
        params: Iterable of parameters to optimize
        defaults: Default optimization options
    """

    def __init__(self, params, defaults: Dict[str, Any]):
        super().__init__(params, defaults)
        self.state_initialized = False

    def _init_state(self, param_group: Dict) -> None:
        """
        Initialize optimizer state for a parameter group.

        Args:
            param_group: Parameter group dictionary
        """
        raise NotImplementedError("Subclasses must implement _init_state")

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided
        """
        raise NotImplementedError("Subclasses must implement step")

    def get_state_dict(self) -> Dict:
        """Get optimizer state dictionary."""
        return self.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer state dictionary."""
        super().load_state_dict(state_dict)


class MetaOptimizer:
    """
    Meta-optimizer for learning optimizer parameters.

    Used for learning the parameters of learned optimizers like DMGD.
    This implements the outer loop in meta-learning.

    Args:
        optimizer_params: Parameters of the optimizer to meta-learn
        meta_lr: Meta-learning rate
        meta_optimizer: Type of meta-optimizer ('sgd' or 'adam')

    Example:
        >>> # Create DMGD optimizer
        >>> dmgd = DeepMomentumGD(model.parameters())
        >>> # Create meta-optimizer for DMGD's MLP parameters
        >>> meta_opt = MetaOptimizer(
        ...     dmgd.momentum_mlp.parameters(),
        ...     meta_lr=0.001
        ... )
    """

    def __init__(
        self,
        optimizer_params,
        meta_lr: float = 0.001,
        meta_optimizer: str = 'adam'
    ):
        self.meta_lr = meta_lr

        # Create optimizer for the optimizer's parameters
        if meta_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(optimizer_params, lr=meta_lr)
        elif meta_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(optimizer_params, lr=meta_lr)
        else:
            raise ValueError(f"Unknown meta_optimizer: {meta_optimizer}")

    def step(self) -> None:
        """Perform meta-optimization step."""
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero meta-optimizer gradients."""
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict:
        """Get meta-optimizer state."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load meta-optimizer state."""
        self.optimizer.load_state_dict(state_dict)


def get_gradient_stats(parameters) -> Dict[str, float]:
    """
    Compute statistics of gradients for monitoring.

    Args:
        parameters: Iterable of parameters

    Returns:
        Dictionary with gradient statistics
    """
    grad_norms = []
    total_norm = 0.0
    num_params = 0

    for param in parameters:
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms.append(grad_norm)
            total_norm += grad_norm ** 2
            num_params += 1

    total_norm = total_norm ** 0.5

    return {
        'total_norm': total_norm,
        'mean_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
        'max_norm': max(grad_norms) if grad_norms else 0.0,
        'min_norm': min(grad_norms) if grad_norms else 0.0,
        'num_params': num_params
    }


def clip_grad_norm(parameters, max_norm: float) -> float:
    """
    Clip gradient norm for stability.

    Args:
        parameters: Iterable of parameters
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
