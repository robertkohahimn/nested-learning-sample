"""
Optimizers module for Nested Learning.

Implements:
- Deep Momentum Gradient Descent (DMGD)
- Nested Optimizer wrapper for multi-frequency updates
"""

__all__ = [
    "DeepMomentumGD",
    "MomentumMLP",
    "NestedOptimizer",
    "NestedOptimizerBuilder",
    "BaseNestedOptimizer",
    "MetaOptimizer",
    "MetaLearningTrainer",
    "create_task_sampler",
]

# Imports
from .dmgd import DeepMomentumGD, MomentumMLP
from .nested_optimizer import NestedOptimizer, NestedOptimizerBuilder
from .base_optimizer import BaseNestedOptimizer, MetaOptimizer
from .meta_learning import MetaLearningTrainer, create_task_sampler
