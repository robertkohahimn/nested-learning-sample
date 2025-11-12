"""
NestedTrainer: Coordinated training framework for Nested Learning.

Provides multi-frequency training coordination, checkpoint management,
and comprehensive metrics tracking.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import json
import time
from collections import defaultdict


class NestedTrainer:
    """
    Training framework for Nested Learning models.

    Coordinates multi-frequency parameter updates, manages checkpoints,
    tracks metrics, and provides training utilities.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer (can be NestedOptimizer)
        criterion: Loss function
        device: Device to train on ('cpu', 'cuda', etc.)
        use_nested_optimizer: If True, expects NestedOptimizer with step()
        callbacks: List of training callbacks

    Example:
        >>> from src.models import NestedMLP
        >>> from src.optimizers import NestedOptimizerBuilder
        >>> 
        >>> model = NestedMLP(784, [256, 128], 10)
        >>> builder = NestedOptimizerBuilder(model, num_levels=3)
        >>> builder.auto_assign_params('uniform')
        >>> optimizer = builder.build(
        ...     optimizer_types=['adam', 'sgd', 'sgd'],
        ...     learning_rates=[0.001, 0.01, 0.1],
        ...     frequencies=[1, 10, 100]
        ... )
        >>> 
        >>> trainer = NestedTrainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     criterion=nn.CrossEntropyLoss(),
        ...     use_nested_optimizer=True
        ... )
        >>> 
        >>> history = trainer.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     epochs=10
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        use_nested_optimizer: bool = False,
        callbacks: Optional[List] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_nested_optimizer = use_nested_optimizer
        self.callbacks = callbacks or []

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)

        # Metrics
        self.train_loss = 0.0
        self.train_steps = 0

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            verbose: If True, print training progress
            log_interval: Steps between logging

        Returns:
            Dictionary with training history
        """
        # Initialize callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Epoch start callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Training
            train_metrics = self._train_epoch(
                train_loader,
                verbose=verbose,
                log_interval=log_interval
            )

            # Validation
            if val_loader is not None:
                val_metrics = self._validate(val_loader, verbose=verbose)
            else:
                val_metrics = {}

            # Update history
            for key, value in {**train_metrics, **val_metrics}.items():
                self.history[key].append(value)

            # Epoch end callbacks
            stop_training = False
            for callback in self.callbacks:
                if callback.on_epoch_end(self, epoch, {**train_metrics, **val_metrics}):
                    stop_training = True
                    break

            if verbose:
                metrics_str = self._format_metrics(train_metrics, val_metrics)
                print(f"Epoch {epoch+1}/{epochs} - {metrics_str}")

            if stop_training:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Training end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        return dict(self.history)

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optimizer step (with global step for NestedOptimizer)
            if self.use_nested_optimizer:
                self.optimizer.step(step=self.global_step)
            else:
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            if output.shape[-1] > 1:  # Classification
                _, predicted = output.max(1)
                total_correct += predicted.eq(target).sum().item()
            total_samples += target.size(0)

            self.global_step += 1
            self.train_steps += 1

            # Logging
            if verbose and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Step {batch_idx+1}/{len(train_loader)} - "
                      f"loss: {avg_loss:.4f}")

            # Batch end callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, loss.item())

        # Compute epoch metrics
        epoch_time = time.time() - epoch_start_time
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'epoch_time': epoch_time
        }

        if total_correct > 0:
            metrics['train_acc'] = total_correct / total_samples

        return metrics

    def _validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                if output.shape[-1] > 1:  # Classification
                    _, predicted = output.max(1)
                    total_correct += predicted.eq(target).sum().item()
                total_samples += target.size(0)

        metrics = {
            'val_loss': total_loss / len(val_loader)
        }

        if total_correct > 0:
            metrics['val_acc'] = total_correct / total_samples

        # Update best val loss
        if metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']

        return metrics

    def _format_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> str:
        """Format metrics for printing."""
        parts = []

        for key, value in train_metrics.items():
            if key != 'epoch_time':
                parts.append(f"{key}: {value:.4f}")

        for key, value in val_metrics.items():
            parts.append(f"{key}: {value:.4f}")

        if 'epoch_time' in train_metrics:
            parts.append(f"time: {train_metrics['epoch_time']:.2f}s")

        return " - ".join(parts)

    def save_checkpoint(
        self,
        filepath: str,
        save_optimizer: bool = True,
        **extra_state
    ) -> None:
        """
        Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            save_optimizer: If True, save optimizer state
            **extra_state: Additional state to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history),
            **extra_state
        }

        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        # Save CMS memory if model has it
        if hasattr(self.model, 'get_memory_stats'):
            try:
                memory_stats = self.model.get_memory_stats()
                checkpoint['memory_stats'] = memory_stats
            except:
                pass

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)

    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file
            load_optimizer: If True, load optimizer state
            strict: If True, strictly enforce state dict keys match

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'history' in checkpoint:
            self.history = defaultdict(list, checkpoint['history'])

        return checkpoint

    def get_lr(self) -> List[float]:
        """Get current learning rate(s)."""
        if hasattr(self.optimizer, 'get_lr'):
            return self.optimizer.get_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

    def set_lr(self, lr: float) -> None:
        """Set learning rate for all parameter groups."""
        if hasattr(self.optimizer, 'set_lr'):
            self.optimizer.set_lr(lr)
        else:
            for group in self.optimizer.param_groups:
                group['lr'] = lr

    def get_update_stats(self) -> Optional[Dict]:
        """Get optimizer update statistics (for NestedOptimizer)."""
        if hasattr(self.optimizer, 'get_update_stats'):
            return self.optimizer.get_update_stats()
        return None

    def reset_history(self) -> None:
        """Reset training history."""
        self.history = defaultdict(list)
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
