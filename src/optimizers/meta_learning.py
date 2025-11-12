"""
Meta-Learning Trainer for DMGD

Implements Model-Agnostic Meta-Learning (MAML) style training for
Deep Momentum GD optimizer, allowing the momentum MLP to learn
effective optimization strategies across multiple tasks.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple, Any
from copy import deepcopy
import numpy as np

from .dmgd import DeepMomentumGD


class MetaLearningTrainer:
    """
    Meta-learning trainer for DMGD optimizer.

    Implements episodic meta-learning where:
    - Inner loop: Train model on task using DMGD
    - Outer loop: Update DMGD's MLP parameters to minimize validation loss

    This allows the momentum MLP to learn how to compute effective momentum
    across different tasks, making DMGD a true "learned optimizer".

    Args:
        model_fn: Function that creates a fresh model instance
        dmgd_optimizer: DMGD optimizer instance (will be meta-trained)
        inner_steps: Number of gradient steps in inner loop
        inner_lr: Learning rate for inner loop task learning
        meta_lr: Learning rate for outer loop meta-update
        device: Device to run on ('cpu' or 'cuda')

    Example:
        >>> # Define model creation function
        >>> def create_model():
        ...     return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
        >>>
        >>> # Create DMGD optimizer
        >>> model = create_model()
        >>> dmgd = DeepMomentumGD(model.parameters(), lr=0.01, mlp_lr=0.001)
        >>>
        >>> # Create meta-trainer
        >>> trainer = MetaLearningTrainer(
        ...     model_fn=create_model,
        ...     dmgd_optimizer=dmgd,
        ...     inner_steps=5,
        ...     inner_lr=0.01,
        ...     meta_lr=0.001
        ... )
        >>>
        >>> # Meta-train on multiple tasks
        >>> for episode in range(100):
        ...     task_data = sample_task()
        ...     meta_loss = trainer.meta_train_step(task_data)
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        dmgd_optimizer: DeepMomentumGD,
        inner_steps: int = 5,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        device: str = 'cpu'
    ):
        self.model_fn = model_fn
        self.dmgd_optimizer = dmgd_optimizer
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.device = device

        # Meta-optimizer for DMGD's MLP parameters
        if dmgd_optimizer.meta_optimizer is None:
            # Create meta-optimizer if not provided
            self.meta_optimizer = torch.optim.Adam(
                dmgd_optimizer.get_momentum_mlp_params(),
                lr=meta_lr
            )
        else:
            self.meta_optimizer = dmgd_optimizer.meta_optimizer

        # Track statistics
        self.episode_count = 0
        self.meta_losses = []

    def inner_loop(
        self,
        model: nn.Module,
        optimizer: DeepMomentumGD,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        criterion: Callable = None
    ) -> nn.Module:
        """
        Inner loop: Train model on task using DMGD.

        Args:
            model: Model to train
            optimizer: DMGD optimizer
            train_data: Tuple of (X_train, y_train)
            criterion: Loss function (default: MSE)

        Returns:
            Trained model
        """
        if criterion is None:
            criterion = nn.MSELoss()

        X_train, y_train = train_data
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        model.train()

        for step in range(self.inner_steps):
            # Forward pass
            pred = model(X_train)
            loss = criterion(pred, y_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def compute_meta_loss(
        self,
        model: nn.Module,
        val_data: Tuple[torch.Tensor, torch.Tensor],
        criterion: Callable = None
    ) -> torch.Tensor:
        """
        Compute meta-objective (validation loss) after inner loop.

        Args:
            model: Model after inner loop training
            val_data: Tuple of (X_val, y_val)
            criterion: Loss function

        Returns:
            Meta-loss (validation loss)
        """
        if criterion is None:
            criterion = nn.MSELoss()

        X_val, y_val = val_data
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        model.eval()
        with torch.enable_grad():  # Need gradients for meta-update
            pred = model(X_val)
            meta_loss = criterion(pred, y_val)

        return meta_loss

    def meta_train_step(
        self,
        task_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        criterion: Callable = None
    ) -> float:
        """
        Perform one meta-training step (episode).

        Args:
            task_data: Dictionary with 'train' and 'val' data
                      Format: {'train': (X_train, y_train), 'val': (X_val, y_val)}
            criterion: Loss function

        Returns:
            Meta-loss value
        """
        # Create fresh model
        model = self.model_fn().to(self.device)

        # Create DMGD optimizer with current MLP parameters
        # We need to use the same MLP, so we share the momentum_mlps
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=self.inner_lr,
            momentum_hidden_dims=self.dmgd_optimizer.momentum_hidden_dims,
            momentum_activation=self.dmgd_optimizer.momentum_activation,
            use_params_in_mlp=self.dmgd_optimizer.use_params_in_mlp
        )

        # Share the MLP parameters (this is the key!)
        optimizer.momentum_mlps = self.dmgd_optimizer.momentum_mlps

        # Inner loop: Train on task
        model = self.inner_loop(model, optimizer, task_data['train'], criterion)

        # Compute meta-loss on validation set
        meta_loss = self.compute_meta_loss(model, task_data['val'], criterion)

        # Outer loop: Update MLP parameters
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
        task_sampler: Callable[[], Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        num_episodes: int,
        criterion: Callable = None,
        verbose: bool = True,
        log_interval: int = 10
    ) -> List[float]:
        """
        Meta-train DMGD optimizer over multiple episodes.

        Args:
            task_sampler: Function that samples a new task
                         Returns: {'train': (X, y), 'val': (X, y)}
            num_episodes: Number of meta-training episodes
            criterion: Loss function
            verbose: Print progress
            log_interval: Print every N episodes

        Returns:
            List of meta-losses
        """
        meta_losses = []

        for episode in range(num_episodes):
            # Sample new task
            task_data = task_sampler()

            # Meta-train on task
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
        Evaluate meta-learned DMGD on a new task.

        Args:
            task_data: Task data with 'train', 'val', and optionally 'test'
            criterion: Loss function

        Returns:
            Dictionary with losses at different stages
        """
        if criterion is None:
            criterion = nn.MSELoss()

        # Create fresh model
        model = self.model_fn().to(self.device)

        # Initial performance (before training)
        X_val, y_val = task_data['val']
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            initial_loss = criterion(pred, y_val).item()

        # Create DMGD optimizer with meta-learned MLPs
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=self.inner_lr,
            momentum_hidden_dims=self.dmgd_optimizer.momentum_hidden_dims,
            momentum_activation=self.dmgd_optimizer.momentum_activation
        )
        optimizer.momentum_mlps = self.dmgd_optimizer.momentum_mlps

        # Train on task
        model = self.inner_loop(model, optimizer, task_data['train'], criterion)

        # Final performance (after training)
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            final_loss = criterion(pred, y_val).item()

        results = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement': initial_loss - final_loss
        }

        # Test set evaluation if available
        if 'test' in task_data:
            X_test, y_test = task_data['test']
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            with torch.no_grad():
                pred = model(X_test)
                test_loss = criterion(pred, y_test).item()
            results['test_loss'] = test_loss

        return results

    def save_checkpoint(self, path: str) -> None:
        """Save meta-learned DMGD checkpoint."""
        checkpoint = {
            'momentum_mlps': self.dmgd_optimizer.momentum_mlps.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'meta_losses': self.meta_losses,
            'config': {
                'inner_steps': self.inner_steps,
                'inner_lr': self.inner_lr,
                'meta_lr': self.meta_lr,
                'momentum_hidden_dims': self.dmgd_optimizer.momentum_hidden_dims,
                'momentum_activation': self.dmgd_optimizer.momentum_activation,
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load meta-learned DMGD checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.dmgd_optimizer.momentum_mlps.load_state_dict(checkpoint['momentum_mlps'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.episode_count = checkpoint['episode_count']
        self.meta_losses = checkpoint['meta_losses']

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.meta_losses:
            return {}

        return {
            'num_episodes': self.episode_count,
            'mean_meta_loss': np.mean(self.meta_losses),
            'std_meta_loss': np.std(self.meta_losses),
            'min_meta_loss': np.min(self.meta_losses),
            'final_meta_loss': self.meta_losses[-1] if self.meta_losses else None,
            'improvement': (self.meta_losses[0] - self.meta_losses[-1])
                          if len(self.meta_losses) > 0 else 0
        }


def create_task_sampler(
    task_type: str = 'linear_regression',
    input_dim: int = 10,
    output_dim: int = 1,
    n_train: int = 50,
    n_val: int = 20
) -> Callable:
    """
    Create a task sampler for meta-learning.

    Args:
        task_type: Type of task ('linear_regression', 'sine', 'polynomial')
        input_dim: Input dimension
        output_dim: Output dimension
        n_train: Number of training samples
        n_val: Number of validation samples

    Returns:
        Task sampler function
    """

    def sample_linear_regression():
        """Sample a random linear regression task."""
        # Random linear function: y = Wx + b + noise
        W = torch.randn(output_dim, input_dim)
        b = torch.randn(output_dim)

        X_train = torch.randn(n_train, input_dim)
        y_train = X_train @ W.T + b + 0.1 * torch.randn(n_train, output_dim)

        X_val = torch.randn(n_val, input_dim)
        y_val = X_val @ W.T + b + 0.1 * torch.randn(n_val, output_dim)

        return {'train': (X_train, y_train), 'val': (X_val, y_val)}

    def sample_sine():
        """Sample a random sine wave task."""
        # Random sine: y = A * sin(ω*x + φ) + noise
        A = torch.rand(1) * 2 + 0.5  # Amplitude: [0.5, 2.5]
        omega = torch.rand(1) * 4 + 1  # Frequency: [1, 5]
        phi = torch.rand(1) * 2 * np.pi  # Phase: [0, 2π]

        X_train = torch.randn(n_train, 1) * 2
        y_train = A * torch.sin(omega * X_train + phi) + 0.1 * torch.randn(n_train, 1)

        X_val = torch.randn(n_val, 1) * 2
        y_val = A * torch.sin(omega * X_val + phi) + 0.1 * torch.randn(n_val, 1)

        return {'train': (X_train, y_train), 'val': (X_val, y_val)}

    def sample_polynomial():
        """Sample a random polynomial task."""
        # Random polynomial: y = c2*x^2 + c1*x + c0 + noise
        coeffs = torch.randn(3)

        X_train = torch.randn(n_train, 1) * 2
        y_train = (coeffs[0] * X_train**2 + coeffs[1] * X_train + coeffs[2] +
                   0.1 * torch.randn(n_train, 1))

        X_val = torch.randn(n_val, 1) * 2
        y_val = (coeffs[0] * X_val**2 + coeffs[1] * X_val + coeffs[2] +
                 0.1 * torch.randn(n_val, 1))

        return {'train': (X_train, y_train), 'val': (X_val, y_val)}

    samplers = {
        'linear_regression': sample_linear_regression,
        'sine': sample_sine,
        'polynomial': sample_polynomial
    }

    if task_type not in samplers:
        raise ValueError(f"Unknown task_type: {task_type}")

    return samplers[task_type]
