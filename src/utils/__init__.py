"""
Utilities module for Nested Learning.

Implements:
- FrequencyScheduler for multi-level updates
- Visualization tools for training dynamics
"""

__all__ = [
    "FrequencyScheduler",
    "AdaptiveFrequencyScheduler",
    "visualize_training",
    "plot_memory_consolidation",
    "plot_training_curves",
    "plot_gradient_flow",
    "plot_update_frequency_heatmap",
    "plot_continual_learning_performance",
    "visualize_attention_weights",
]

# Imports
from .frequency_scheduler import FrequencyScheduler, AdaptiveFrequencyScheduler
from .visualization import (
    visualize_training,
    plot_memory_consolidation,
    plot_training_curves,
    plot_gradient_flow,
    plot_update_frequency_heatmap,
    plot_continual_learning_performance,
    visualize_attention_weights,
)
