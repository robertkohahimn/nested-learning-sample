"""
Visualization tools for Nested Learning

Provides utilities for visualizing:
- Update frequency patterns
- Memory consolidation
- Training dynamics
- Gradient flow
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch


def plot_update_frequency_heatmap(
    update_history: Dict[int, List[int]],
    max_steps: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap showing which levels updated at each step.

    Args:
        update_history: Dict mapping level -> list of steps when it updated
        max_steps: Maximum steps to display (None = all)
        figsize: Figure size
        save_path: Path to save figure (None = display only)

    Returns:
        Matplotlib figure

    Example:
        >>> update_history = {
        ...     0: [0, 1, 2, 3, 4, 5],  # Updates every step
        ...     1: [0, 10, 20],          # Updates every 10 steps
        ...     2: [0, 100]              # Updates every 100 steps
        ... }
        >>> fig = plot_update_frequency_heatmap(update_history)
    """
    num_levels = len(update_history)

    # Determine max steps
    if max_steps is None:
        max_steps = max(max(steps) for steps in update_history.values() if steps)

    # Create binary matrix: (num_levels, max_steps)
    update_matrix = np.zeros((num_levels, max_steps + 1))
    for level, steps in update_history.items():
        for step in steps:
            if step <= max_steps:
                update_matrix[level, step] = 1

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        update_matrix,
        cmap=['white', 'darkblue'],
        cbar=False,
        linewidths=0,
        ax=ax
    )

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Optimization Level', fontsize=12)
    ax.set_title('Multi-Frequency Update Pattern', fontsize=14, fontweight='bold')

    # Set y-axis labels
    ax.set_yticks(np.arange(num_levels) + 0.5)
    ax.set_yticklabels([f'Level {i}' for i in range(num_levels)])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_memory_consolidation(
    memory_stats_history: List[Dict[int, Dict[str, float]]],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot memory usage and consolidation over time.

    Args:
        memory_stats_history: List of memory stats at each checkpoint
                             Each entry is dict mapping level -> stats
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> # Collect stats during training
        >>> stats_history = []
        >>> for step in range(100):
        ...     stats = cms.get_memory_stats()
        ...     stats_history.append(stats)
        >>> fig = plot_memory_consolidation(stats_history)
    """
    if not memory_stats_history:
        raise ValueError("memory_stats_history is empty")

    num_levels = len(memory_stats_history[0])
    num_checkpoints = len(memory_stats_history)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Extract data for each level
    for level in range(num_levels):
        usage = [stats[level]['usage'] for stats in memory_stats_history]
        count = [stats[level]['count'] for stats in memory_stats_history]
        capacity = memory_stats_history[0][level]['capacity']

        steps = np.arange(num_checkpoints)

        # Plot 1: Memory usage fraction
        axes[0].plot(steps, usage, label=f'Level {level}', linewidth=2, marker='o', markersize=3)

        # Plot 2: Absolute memory count
        axes[1].plot(steps, count, label=f'Level {level}', linewidth=2, marker='o', markersize=3)

        # Plot 3: Capacity utilization
        utilization = [c / capacity * 100 for c in count]
        axes[2].plot(steps, utilization, label=f'Level {level}', linewidth=2, marker='o', markersize=3)

    # Configure subplots
    axes[0].set_xlabel('Checkpoint')
    axes[0].set_ylabel('Memory Usage Fraction')
    axes[0].set_title('Memory Usage Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Checkpoint')
    axes[1].set_ylabel('Number of Entries')
    axes[1].set_title('Memory Count Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Checkpoint')
    axes[2].set_ylabel('Capacity Utilization (%)')
    axes[2].set_title('Capacity Utilization Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_curves(
    losses: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves for different loss components.

    Args:
        losses: Dict mapping loss name -> list of values
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> losses = {
        ...     'total': [1.5, 1.2, 0.9, 0.7],
        ...     'task': [1.0, 0.8, 0.6, 0.5],
        ...     'memory': [0.5, 0.4, 0.3, 0.2]
        ... }
        >>> fig = plot_training_curves(losses)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: All losses together
    for name, values in losses.items():
        axes[0].plot(values, label=name, linewidth=2)

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Losses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Log scale for better visibility
    for name, values in losses.items():
        axes[1].semilogy(values, label=name, linewidth=2)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Training Losses (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_gradient_flow(
    gradient_magnitudes: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize gradient flow across different parameter groups/levels.

    Args:
        gradient_magnitudes: Dict mapping parameter name -> gradient magnitudes over time
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> grads = {
        ...     'level_0': [1.0, 0.9, 0.8, 0.7],
        ...     'level_1': [0.5, 0.4, 0.3, 0.3],
        ...     'level_2': [0.2, 0.15, 0.1, 0.1]
        ... }
        >>> fig = plot_gradient_flow(grads)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Gradient magnitude over time
    for name, magnitudes in gradient_magnitudes.items():
        axes[0].plot(magnitudes, label=name, linewidth=2, marker='o', markersize=3)

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Gradient Magnitude')
    axes[0].set_title('Gradient Flow Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Heatmap of gradient magnitudes
    param_names = list(gradient_magnitudes.keys())
    num_params = len(param_names)
    num_steps = len(next(iter(gradient_magnitudes.values())))

    grad_matrix = np.zeros((num_params, num_steps))
    for i, name in enumerate(param_names):
        grad_matrix[i] = gradient_magnitudes[name]

    im = axes[1].imshow(grad_matrix, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Parameter Group')
    axes[1].set_title('Gradient Magnitude Heatmap')
    axes[1].set_yticks(np.arange(num_params))
    axes[1].set_yticklabels(param_names)

    plt.colorbar(im, ax=axes[1], label='Gradient Magnitude')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_continual_learning_performance(
    accuracies: Dict[str, List[float]],
    task_boundaries: List[int],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance on multiple tasks during continual learning.

    Args:
        accuracies: Dict mapping task name -> accuracy over time
        task_boundaries: List of steps where new tasks begin
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> accuracies = {
        ...     'Task 1': [0.5, 0.8, 0.9, 0.92, 0.90, 0.88],
        ...     'Task 2': [0.0, 0.0, 0.0, 0.6, 0.85, 0.87],
        ...     'Task 3': [0.0, 0.0, 0.0, 0.0, 0.0, 0.75]
        ... }
        >>> boundaries = [0, 3, 5]
        >>> fig = plot_continual_learning_performance(accuracies, boundaries)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot accuracies
    for task_name, acc_values in accuracies.items():
        ax.plot(acc_values, label=task_name, linewidth=2, marker='o', markersize=4)

    # Add vertical lines for task boundaries
    for boundary in task_boundaries[1:]:  # Skip first boundary (0)
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5, linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Continual Learning: Task Performance Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add text annotations for task regions
    for i in range(len(task_boundaries)):
        start = task_boundaries[i]
        end = task_boundaries[i + 1] if i + 1 < len(task_boundaries) else len(next(iter(accuracies.values())))
        mid = (start + end) / 2
        ax.text(mid, 1.02, f'Task {i + 1}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_attention_weights(
    attention_weights: torch.Tensor,
    query_labels: Optional[List[str]] = None,
    key_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights from memory retrieval.

    Args:
        attention_weights: Tensor of shape (num_queries, num_keys)
        query_labels: Labels for queries
        key_labels: Labels for keys
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    ax.set_xlabel('Memory Keys')
    ax.set_ylabel('Queries')
    ax.set_title('Attention Weights for Memory Retrieval')

    if query_labels:
        ax.set_yticks(np.arange(len(query_labels)))
        ax.set_yticklabels(query_labels)

    if key_labels:
        ax.set_xticks(np.arange(len(key_labels)))
        ax.set_xticklabels(key_labels, rotation=45, ha='right')

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# Convenience function
def visualize_training(
    losses: Dict[str, List[float]],
    memory_stats: List[Dict[int, Dict[str, float]]],
    gradient_magnitudes: Optional[Dict[str, List[float]]] = None,
    update_history: Optional[Dict[int, List[int]]] = None,
    save_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive visualization of training dynamics.

    Args:
        losses: Training losses
        memory_stats: Memory statistics history
        gradient_magnitudes: Gradient flow data (optional)
        update_history: Update frequency history (optional)
        save_dir: Directory to save figures (None = don't save)

    Returns:
        Dictionary mapping plot name -> figure

    Example:
        >>> figures = visualize_training(
        ...     losses={'total': [...]},
        ...     memory_stats=[...],
        ...     gradient_magnitudes={'level_0': [...]},
        ...     update_history={0: [...], 1: [...]}
        ... )
    """
    figures = {}

    # Training curves
    fig = plot_training_curves(losses)
    figures['training_curves'] = fig
    if save_dir:
        fig.savefig(f'{save_dir}/training_curves.png')

    # Memory consolidation
    fig = plot_memory_consolidation(memory_stats)
    figures['memory_consolidation'] = fig
    if save_dir:
        fig.savefig(f'{save_dir}/memory_consolidation.png')

    # Gradient flow (if provided)
    if gradient_magnitudes:
        fig = plot_gradient_flow(gradient_magnitudes)
        figures['gradient_flow'] = fig
        if save_dir:
            fig.savefig(f'{save_dir}/gradient_flow.png')

    # Update frequency (if provided)
    if update_history:
        fig = plot_update_frequency_heatmap(update_history)
        figures['update_frequency'] = fig
        if save_dir:
            fig.savefig(f'{save_dir}/update_frequency.png')

    return figures
