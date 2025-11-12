"""
Training callbacks for NestedTrainer.

Provides reusable components for training control and monitoring.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


class Callback:
    """Base callback class."""

    def on_train_begin(self, trainer) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Called at the end of each epoch.

        Returns:
            bool: True to stop training, False to continue
        """
        return False

    def on_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.

    Stops training when monitored metric stops improving.

    Args:
        monitor: Metric to monitor ('val_loss', 'val_acc', etc.)
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
        verbose: If True, print messages

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor='val_loss',
        ...     patience=5,
        ...     min_delta=0.001,
        ...     mode='min'
        ... )
        >>> trainer = NestedTrainer(model, optimizer, criterion,
        ...                         callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.wait = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check if training should stop."""
        if self.monitor not in metrics:
            return False

        current = metrics[self.monitor]

        if self.mode == 'min':
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"\nEarly stopping triggered: {self.monitor} "
                      f"did not improve for {self.patience} epochs")
            return True

        return False


class CheckpointCallback(Callback):
    """
    Checkpoint saving callback.

    Automatically saves checkpoints during training.

    Args:
        filepath: Path template for checkpoints (can include {epoch})
        monitor: Metric to monitor for best checkpoint
        mode: 'min' for loss, 'max' for accuracy
        save_best_only: If True, only save when metric improves
        save_freq: Save every N epochs (if save_best_only=False)
        verbose: If True, print save messages

    Example:
        >>> checkpoint = CheckpointCallback(
        ...     filepath='checkpoints/epoch_{epoch}.pt',
        ...     monitor='val_loss',
        ...     save_best_only=True
        ... )
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
        verbose: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Save checkpoint if conditions are met."""
        should_save = False

        if self.save_best_only:
            if self.monitor in metrics:
                current = metrics[self.monitor]

                if self.mode == 'min':
                    improved = current < self.best_value
                else:
                    improved = current > self.best_value

                if improved:
                    self.best_value = current
                    should_save = True
        else:
            if (epoch + 1) % self.save_freq == 0:
                should_save = True

        if should_save:
            filepath = self.filepath.format(epoch=epoch+1)
            trainer.save_checkpoint(filepath)

            if self.verbose:
                print(f"Checkpoint saved: {filepath}")

        return False


class LRSchedulerCallback(Callback):
    """
    Learning rate scheduler callback.

    Args:
        scheduler: PyTorch LR scheduler
        monitor: Metric to pass to scheduler (for ReduceLROnPlateau)
        verbose: If True, print LR changes

    Example:
        >>> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ...     optimizer, mode='min', patience=3
        ... )
        >>> lr_callback = LRSchedulerCallback(
        ...     scheduler=scheduler,
        ...     monitor='val_loss'
        ... )
    """

    def __init__(
        self,
        scheduler,
        monitor: Optional[str] = None,
        verbose: bool = True
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
        self.last_lr = None

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Update learning rate."""
        # Get current LR before step
        current_lr = trainer.get_lr()

        # Step scheduler
        if self.monitor is not None and self.monitor in metrics:
            # ReduceLROnPlateau needs metric value
            self.scheduler.step(metrics[self.monitor])
        else:
            # Other schedulers
            self.scheduler.step()

        # Check if LR changed
        new_lr = trainer.get_lr()
        if self.verbose and current_lr != new_lr:
            print(f"Learning rate updated: {current_lr} -> {new_lr}")

        return False


class TensorBoardCallback(Callback):
    """
    TensorBoard logging callback.

    Args:
        log_dir: Directory for TensorBoard logs
        log_every_n_steps: Log training loss every N steps

    Example:
        >>> tb_callback = TensorBoardCallback(log_dir='runs/experiment1')
    """

    def __init__(
        self,
        log_dir: str,
        log_every_n_steps: int = 100
    ):
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.writer = None

    def on_train_begin(self, trainer) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("Warning: tensorboard not installed, logging disabled")
            self.writer = None

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return False

        for key, value in metrics.items():
            if key != 'epoch_time':
                self.writer.add_scalar(key, value, epoch)

        # Log learning rates
        lrs = trainer.get_lr()
        for i, lr in enumerate(lrs):
            self.writer.add_scalar(f'lr/level_{i}', lr, epoch)

        # Log update statistics if available
        update_stats = trainer.get_update_stats()
        if update_stats:
            for level, stats in update_stats.items():
                if 'update_count' in stats:
                    self.writer.add_scalar(
                        f'updates/level_{level}',
                        stats['update_count'],
                        epoch
                    )

        return False

    def on_train_end(self, trainer) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class MetricsLogger(Callback):
    """
    Simple metrics logger callback.

    Logs metrics to a JSON file.

    Args:
        filepath: Path to save metrics JSON

    Example:
        >>> logger = MetricsLogger('metrics.json')
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metrics_history = []

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Log metrics."""
        import json

        entry = {'epoch': epoch + 1, **metrics}
        self.metrics_history.append(entry)

        # Save to file
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        return False


class ProgressCallback(Callback):
    """
    Progress bar callback using tqdm.

    Args:
        total_epochs: Total number of epochs

    Example:
        >>> progress = ProgressCallback(total_epochs=10)
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self, trainer) -> None:
        """Initialize progress bar."""
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_epochs, desc="Training")
        except ImportError:
            self.pbar = None

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Update progress bar."""
        if self.pbar is not None:
            # Format metrics for display
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()
                                      if k != 'epoch_time'])
            self.pbar.set_postfix_str(metrics_str)
            self.pbar.update(1)

        return False

    def on_train_end(self, trainer) -> None:
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()
