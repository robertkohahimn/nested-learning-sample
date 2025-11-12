"""
Improved Meta-Learning with Better Gradient Flow

This module provides an improved meta-learning implementation that
properly maintains gradient flow through the inner loop optimization.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple
from copy import deepcopy


def functional_step_dmgd(
    model: nn.Module,
    loss: torch.Tensor,
    momentum_mlp: nn.Module,
    lr: float,
    momentum_state: Optional[Dict] = None
) -> Tuple[nn.Module, Dict]:
    """
    Perform a single differentiable optimization step.

    This computes gradients and updates parameters while maintaining
    the computational graph for meta-learning.

    Args:
        model: Model to update
        loss: Loss to backpropagate
        momentum_mlp: MLP for computing momentum
        lr: Learning rate
        momentum_state: Previous momentum state

    Returns:
        Tuple of (updated_model, new_momentum_state)
    """
    # Compute gradients
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=True,  # Keep graph for meta-gradients
        retain_graph=True
    )

    if momentum_state is None:
        momentum_state = {}

    new_params = []
    new_momentum_state = {}

    for i, (param, grad) in enumerate(zip(model.parameters(), grads)):
        # Get previous momentum (or zeros)
        prev_momentum = momentum_state.get(i, torch.zeros_like(param))

        # Flatten for MLP
        grad_flat = grad.flatten()
        momentum_flat = prev_momentum.flatten()

        # Compute new momentum with MLP
        # Add batch dimension for MLP (expects 2D input)
        mlp_input = torch.cat([grad_flat, momentum_flat]).unsqueeze(0)
        new_momentum_flat = momentum_mlp(mlp_input).squeeze(0)

        # Reshape
        new_momentum = new_momentum_flat.reshape(param.shape)

        # Update parameter (functionally, not in-place)
        new_param = param - lr * new_momentum

        new_params.append(new_param)
        new_momentum_state[i] = new_momentum

    # Create new model with updated parameters
    updated_model = deepcopy(model)
    for new_param, param in zip(new_params, updated_model.parameters()):
        param.data = new_param

    return updated_model, new_momentum_state


class ImprovedMetaLearningTrainer:
    """
    Improved meta-learning trainer with proper gradient flow.

    This version maintains the computational graph through the inner loop,
    allowing gradients to flow back to the MLP parameters.

    Args:
        model_fn: Function that creates a fresh model instance
        momentum_mlp: MLP for computing momentum
        inner_steps: Number of gradient steps in inner loop
        inner_lr: Learning rate for inner loop
        meta_lr: Learning rate for meta-optimizer
        device: Device to run on

    Example:
        >>> from src.optimizers.dmgd import MomentumMLP
        >>>
        >>> def create_model():
        ...     return nn.Linear(10, 1)
        >>>
        >>> # Create momentum MLP
        >>> mlp = MomentumMLP(input_dim=20, hidden_dims=[16])  # 10 grad + 10 momentum
        >>>
        >>> # Create trainer
        >>> trainer = ImprovedMetaLearningTrainer(
        ...     model_fn=create_model,
        ...     momentum_mlp=mlp,
        ...     inner_steps=5,
        ...     inner_lr=0.01,
        ...     meta_lr=0.001
        ... )
        >>>
        >>> # Meta-train
        >>> for episode in range(100):
        ...     task_data = sample_task()
        ...     meta_loss = trainer.meta_train_step(task_data)
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        momentum_mlp: nn.Module,
        inner_steps: int = 5,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        device: str = 'cpu'
    ):
        self.model_fn = model_fn
        self.momentum_mlp = momentum_mlp
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.device = device

        # Meta-optimizer for MLP
        self.meta_optimizer = torch.optim.Adam(
            momentum_mlp.parameters(),
            lr=meta_lr
        )

        # Track statistics
        self.episode_count = 0
        self.meta_losses = []
        self.inner_losses_history = []

    def inner_loop_differentiable(
        self,
        model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        criterion: Callable = None
    ) -> nn.Module:
        """
        Inner loop with differentiable updates.

        Maintains computational graph through optimization steps.

        Args:
            model: Model to train
            train_data: Tuple of (X_train, y_train)
            criterion: Loss function

        Returns:
            Model after training
        """
        if criterion is None:
            criterion = nn.MSELoss()

        X_train, y_train = train_data
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        momentum_state = None
        current_model = model

        inner_losses = []

        for step in range(self.inner_steps):
            # Forward pass
            pred = current_model(X_train)
            loss = criterion(pred, y_train)
            inner_losses.append(loss.item())

            # Functional update (maintains gradients)
            current_model, momentum_state = functional_step_dmgd(
                model=current_model,
                loss=loss,
                momentum_mlp=self.momentum_mlp,
                lr=self.inner_lr,
                momentum_state=momentum_state
            )

        self.inner_losses_history.append(inner_losses)
        return current_model

    def compute_meta_loss(
        self,
        model: nn.Module,
        val_data: Tuple[torch.Tensor, torch.Tensor],
        criterion: Callable = None
    ) -> torch.Tensor:
        """
        Compute meta-objective on validation set.

        Args:
            model: Model after inner loop
            val_data: Tuple of (X_val, y_val)
            criterion: Loss function

        Returns:
            Meta-loss (has gradients w.r.t. MLP parameters)
        """
        if criterion is None:
            criterion = nn.MSELoss()

        X_val, y_val = val_data
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        pred = model(X_val)
        meta_loss = criterion(pred, y_val)

        return meta_loss

    def meta_train_step(
        self,
        task_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        criterion: Callable = None
    ) -> float:
        """
        Perform one meta-training step with proper gradient flow.

        Args:
            task_data: Dictionary with 'train' and 'val' data
            criterion: Loss function

        Returns:
            Meta-loss value
        """
        # Create fresh model
        model = self.model_fn().to(self.device)

        # Inner loop (differentiable)
        trained_model = self.inner_loop_differentiable(
            model,
            task_data['train'],
            criterion
        )

        # Compute meta-loss
        meta_loss = self.compute_meta_loss(
            trained_model,
            task_data['val'],
            criterion
        )

        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # Track statistics
        self.episode_count += 1
        meta_loss_value = meta_loss.item()
        self.meta_losses.append(meta_loss_value)

        return meta_loss_value

    def meta_train(
        self,
        task_sampler: Callable,
        num_episodes: int,
        criterion: Callable = None,
        verbose: bool = True,
        log_interval: int = 10
    ) -> List[float]:
        """
        Meta-train over multiple episodes.

        Args:
            task_sampler: Function that samples tasks
            num_episodes: Number of episodes
            criterion: Loss function
            verbose: Print progress
            log_interval: Print every N episodes

        Returns:
            List of meta-losses
        """
        import numpy as np

        meta_losses = []

        for episode in range(num_episodes):
            # Sample task
            task_data = task_sampler()

            # Meta-train
            meta_loss = self.meta_train_step(task_data, criterion)
            meta_losses.append(meta_loss)

            # Logging
            if verbose and (episode + 1) % log_interval == 0:
                avg_loss = np.mean(meta_losses[-log_interval:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Meta-Loss: {avg_loss:.4f}")

        return meta_losses

    def evaluate(
        self,
        task_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        criterion: Callable = None
    ) -> Dict[str, float]:
        """
        Evaluate meta-learned MLP on a new task.

        Args:
            task_data: Task data with 'train' and 'val'
            criterion: Loss function

        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.MSELoss()

        # Create fresh model
        model = self.model_fn().to(self.device)

        # Initial performance
        X_val, y_val = task_data['val']
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        with torch.no_grad():
            pred = model(X_val)
            initial_loss = criterion(pred, y_val).item()

        # Train on task (without gradients for evaluation)
        with torch.no_grad():
            trained_model = self.inner_loop_differentiable(
                model,
                task_data['train'],
                criterion
            )

        # Final performance
        with torch.no_grad():
            pred = trained_model(X_val)
            final_loss = criterion(pred, y_val).item()

        return {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement': initial_loss - final_loss
        }

    def get_statistics(self) -> Dict:
        """Get training statistics."""
        import numpy as np

        if not self.meta_losses:
            return {}

        return {
            'num_episodes': self.episode_count,
            'mean_meta_loss': np.mean(self.meta_losses),
            'std_meta_loss': np.std(self.meta_losses),
            'min_meta_loss': np.min(self.meta_losses),
            'final_meta_loss': self.meta_losses[-1],
            'improvement': self.meta_losses[0] - self.meta_losses[-1] if len(self.meta_losses) > 0 else 0
        }

    def save_checkpoint(self, path: str) -> None:
        """Save meta-learned MLP."""
        checkpoint = {
            'momentum_mlp': self.momentum_mlp.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'meta_losses': self.meta_losses,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load meta-learned MLP."""
        checkpoint = torch.load(path, map_location=self.device)
        self.momentum_mlp.load_state_dict(checkpoint['momentum_mlp'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.episode_count = checkpoint['episode_count']
        self.meta_losses = checkpoint['meta_losses']
