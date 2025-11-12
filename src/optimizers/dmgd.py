"""
Deep Momentum Gradient Descent (DMGD) Optimizer

Replaces standard linear momentum with a learnable MLP that computes
momentum based on current gradients, previous momentum, and optionally
parameter values.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional, Callable, Dict, Any
from .base_optimizer import BaseNestedOptimizer


class MomentumMLP(nn.Module):
    """
    MLP for computing momentum in DMGD.

    Takes as input:
    - Current gradient
    - Previous momentum
    - (Optional) Current parameter value

    Outputs:
    - New momentum

    Args:
        input_dim: Dimension of concatenated input (gradient + momentum [+ params])
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'tanh', 'gelu')
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout probability (0 = no dropout)

    Example:
        >>> mlp = MomentumMLP(input_dim=128, hidden_dims=[64, 32])
        >>> # gradient: (64,), momentum: (64,)
        >>> input = torch.cat([gradient, momentum], dim=0)  # (128,)
        >>> new_momentum = mlp(input)  # (64,)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        activation: str = 'relu',
        use_layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (no activation, returns momentum directly)
        output_dim = input_dim // 2  # Momentum has same dim as gradient
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute momentum from input.

        Args:
            x: Concatenated [gradient, previous_momentum, (optional) params]

        Returns:
            New momentum vector
        """
        return self.mlp(x)


class DeepMomentumGD(BaseNestedOptimizer):
    """
    Deep Momentum Gradient Descent optimizer.

    Uses a learnable MLP to compute momentum instead of standard linear momentum.

    Standard momentum:
        m_{t+1} = β * m_t + (1-β) * g_t
        θ_{t+1} = θ_t - α * m_{t+1}

    DMGD:
        m_{t+1} = MLP([g_t, m_t])
        θ_{t+1} = θ_t - α * m_{t+1}

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum_hidden_dims: Hidden dimensions for momentum MLP
        momentum_activation: Activation function for MLP
        use_params_in_mlp: Whether to include parameter values in MLP input
        weight_decay: L2 penalty (default: 0)
        grad_clip: Gradient clipping value (None = no clipping)
        mlp_lr: Learning rate for momentum MLP meta-learning (None = no meta-learning)

    Example:
        >>> model = nn.Linear(10, 1)
        >>> optimizer = DeepMomentumGD(model.parameters(), lr=0.01)
        >>>
        >>> for epoch in range(100):
        >>>     loss = model(x).sum()
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum_hidden_dims: List[int] = [64, 32],
        momentum_activation: str = 'relu',
        use_params_in_mlp: bool = False,
        weight_decay: float = 0,
        grad_clip: Optional[float] = None,
        mlp_lr: Optional[float] = None
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip
        )
        super().__init__(params, defaults)

        self.use_params_in_mlp = use_params_in_mlp
        self.momentum_hidden_dims = momentum_hidden_dims
        self.momentum_activation = momentum_activation
        self.mlp_lr = mlp_lr

        # Momentum MLPs for each parameter group
        self.momentum_mlps = nn.ModuleDict()

        # Initialize MLPs for each parameter group
        self._init_momentum_mlps()

        # Meta-optimizer for learning MLP parameters (if mlp_lr is provided)
        if mlp_lr is not None:
            self.meta_optimizer = torch.optim.Adam(
                self.momentum_mlps.parameters(),
                lr=mlp_lr
            )
        else:
            self.meta_optimizer = None

    def _init_momentum_mlps(self):
        """Initialize momentum MLPs for each parameter group."""
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                if param.requires_grad:
                    param_size = param.numel()

                    # Input: [gradient, momentum] or [gradient, momentum, params]
                    if self.use_params_in_mlp:
                        input_dim = param_size * 3
                    else:
                        input_dim = param_size * 2

                    # Create MLP for this parameter
                    mlp_key = f"group{group_idx}_param{param_idx}"
                    self.momentum_mlps[mlp_key] = MomentumMLP(
                        input_dim=input_dim,
                        hidden_dims=self.momentum_hidden_dims,
                        activation=self.momentum_activation
                    )

    def _get_mlp_key(self, group_idx: int, param_idx: int) -> str:
        """Get MLP key for a specific parameter."""
        return f"group{group_idx}_param{param_idx}"

    @torch.no_grad()
    def _init_state(self, param: torch.Tensor) -> None:
        """Initialize optimizer state for a parameter."""
        state = self.state[param]
        if 'momentum' not in state:
            state['momentum'] = torch.zeros_like(param)
            state['step'] = 0

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates model and returns loss

        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            weight_decay = group['weight_decay']
            grad_clip = group['grad_clip']

            for param_idx, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                # Get gradient
                grad = param.grad.data

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                # Clip gradient if specified
                if grad_clip is not None:
                    grad = torch.clamp(grad, -grad_clip, grad_clip)

                # Initialize state
                self._init_state(param)
                state = self.state[param]

                # Get previous momentum
                prev_momentum = state['momentum']

                # Flatten for MLP
                grad_flat = grad.flatten()
                momentum_flat = prev_momentum.flatten()

                # Prepare MLP input
                if self.use_params_in_mlp:
                    param_flat = param.data.flatten()
                    mlp_input = torch.cat([grad_flat, momentum_flat, param_flat])
                else:
                    mlp_input = torch.cat([grad_flat, momentum_flat])

                # Compute new momentum using MLP
                mlp_key = self._get_mlp_key(group_idx, param_idx)
                mlp = self.momentum_mlps[mlp_key]

                new_momentum_flat = mlp(mlp_input)

                # Reshape to original parameter shape
                new_momentum = new_momentum_flat.reshape(param.shape)

                # Update momentum in state
                state['momentum'] = new_momentum

                # Update parameter
                param.data.add_(new_momentum, alpha=-lr)

                # Increment step counter
                state['step'] += 1

        return loss

    def meta_step(self, meta_loss: torch.Tensor) -> None:
        """
        Perform meta-optimization step to update MLP parameters.

        This implements the outer loop of meta-learning for the optimizer.

        Args:
            meta_loss: Meta-objective loss (e.g., validation loss)

        Example:
            >>> # Inner loop: train with DMGD
            >>> for step in range(k_steps):
            >>>     train_loss = model(train_x).sum()
            >>>     train_loss.backward()
            >>>     optimizer.step()
            >>>     optimizer.zero_grad()
            >>>
            >>> # Outer loop: update DMGD's MLP based on validation loss
            >>> val_loss = model(val_x).sum()
            >>> optimizer.meta_step(val_loss)
        """
        if self.meta_optimizer is None:
            raise RuntimeError("meta_optimizer is None. Set mlp_lr when creating optimizer.")

        # Zero meta-gradients
        self.meta_optimizer.zero_grad()

        # Compute gradients w.r.t. MLP parameters
        meta_loss.backward()

        # Update MLP parameters
        self.meta_optimizer.step()

    def get_momentum_mlp_params(self) -> List[torch.Tensor]:
        """Get all MLP parameters for external meta-optimization."""
        return list(self.momentum_mlps.parameters())

    def state_dict(self) -> Dict:
        """Get optimizer state including MLP parameters."""
        state_dict = super().state_dict()
        state_dict['momentum_mlps'] = self.momentum_mlps.state_dict()
        if self.meta_optimizer is not None:
            state_dict['meta_optimizer'] = self.meta_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer state including MLP parameters."""
        # Load MLP state
        if 'momentum_mlps' in state_dict:
            self.momentum_mlps.load_state_dict(state_dict['momentum_mlps'])
            del state_dict['momentum_mlps']

        # Load meta-optimizer state
        if 'meta_optimizer' in state_dict and self.meta_optimizer is not None:
            self.meta_optimizer.load_state_dict(state_dict['meta_optimizer'])
            del state_dict['meta_optimizer']

        # Load base optimizer state
        super().load_state_dict(state_dict)

    def __repr__(self) -> str:
        """String representation."""
        return (f"DeepMomentumGD(lr={self.defaults['lr']}, "
                f"mlp_hidden={self.momentum_hidden_dims}, "
                f"use_params={self.use_params_in_mlp}, "
                f"meta_lr={self.mlp_lr})")
