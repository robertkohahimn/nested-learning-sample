"""
Training framework for Nested Learning.

Provides coordinated training with multi-frequency updates, checkpoint management,
and comprehensive metrics tracking.
"""

from .nested_trainer import NestedTrainer
from .callbacks import EarlyStopping, LRSchedulerCallback, CheckpointCallback
from .metrics import MetricsTracker

__all__ = [
    'NestedTrainer',
    'EarlyStopping',
    'LRSchedulerCallback',
    'CheckpointCallback',
    'MetricsTracker',
]
