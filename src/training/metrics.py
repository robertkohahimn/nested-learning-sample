"""
Metrics tracking for Nested Learning training.

Provides utilities for tracking and analyzing training metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
from pathlib import Path


class MetricsTracker:
    """
    Comprehensive metrics tracking for training.

    Tracks losses, accuracies, learning rates, update frequencies,
    and memory statistics across training.

    Example:
        >>> tracker = MetricsTracker()
        >>> 
        >>> # During training
        >>> tracker.update('train_loss', 0.5, step=100)
        >>> tracker.update('val_acc', 0.85, step=100)
        >>> 
        >>> # Get statistics
        >>> stats = tracker.get_stats('train_loss')
        >>> print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.steps = defaultdict(list)

    def update(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Update a metric.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        self.metrics[name].append(value)
        if step is not None:
            self.steps[name].append(step)

    def get(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])

    def get_last(self, name: str, default: float = 0.0) -> float:
        """Get last value for a metric."""
        values = self.metrics.get(name, [])
        return values[-1] if values else default

    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Returns:
            Dict with mean, std, min, max, last
        """
        values = self.metrics.get(name, [])
        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'last': 0.0,
                'count': 0
            }

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'last': values[-1],
            'count': len(values)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}

    def moving_average(self, name: str, window: int = 10) -> List[float]:
        """
        Calculate moving average of a metric.

        Args:
            name: Metric name
            window: Window size for moving average

        Returns:
            List of moving average values
        """
        values = self.metrics.get(name, [])
        if len(values) < window:
            return values

        ma = []
        for i in range(len(values) - window + 1):
            ma.append(np.mean(values[i:i+window]))

        return ma

    def get_best(self, name: str, mode: str = 'min') -> float:
        """
        Get best value for a metric.

        Args:
            name: Metric name
            mode: 'min' or 'max'

        Returns:
            Best value
        """
        values = self.metrics.get(name, [])
        if not values:
            return float('inf') if mode == 'min' else float('-inf')

        return min(values) if mode == 'min' else max(values)

    def get_best_epoch(self, name: str, mode: str = 'min') -> int:
        """
        Get epoch with best value for a metric.

        Args:
            name: Metric name
            mode: 'min' or 'max'

        Returns:
            Best epoch (0-indexed)
        """
        values = self.metrics.get(name, [])
        if not values:
            return -1

        if mode == 'min':
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset metrics.

        Args:
            name: Specific metric to reset, or None for all
        """
        if name is None:
            self.metrics = defaultdict(list)
            self.steps = defaultdict(list)
        else:
            self.metrics[name] = []
            self.steps[name] = []

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert metrics to dictionary."""
        return dict(self.metrics)

    def save(self, filepath: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            filepath: Path to save metrics
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metrics': dict(self.metrics),
            'steps': dict(self.steps)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load metrics from JSON file.

        Args:
            filepath: Path to load metrics from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metrics = defaultdict(list, data.get('metrics', {}))
        self.steps = defaultdict(list, data.get('steps', {}))

    def plot(self, names: List[str], filepath: Optional[str] = None) -> None:
        """
        Plot metrics.

        Args:
            names: List of metric names to plot
            filepath: Optional path to save plot

        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, cannot plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for name in names:
            values = self.metrics.get(name, [])
            if values:
                ax.plot(values, label=name)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Training Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


class UpdateFrequencyTracker:
    """
    Track parameter update frequencies for NestedOptimizer.

    Example:
        >>> tracker = UpdateFrequencyTracker(num_levels=3)
        >>> 
        >>> # During training
        >>> update_mask = nested_optimizer.step(step=100)
        >>> tracker.record(update_mask, step=100)
        >>> 
        >>> # Analyze
        >>> stats = tracker.get_statistics()
    """

    def __init__(self, num_levels: int):
        self.num_levels = num_levels
        self.update_counts = {i: 0 for i in range(num_levels)}
        self.update_history = {i: [] for i in range(num_levels)}

    def record(self, update_mask: Dict[int, bool], step: int) -> None:
        """
        Record which levels updated.

        Args:
            update_mask: Dict mapping level to bool (True if updated)
            step: Current training step
        """
        for level, updated in update_mask.items():
            if updated:
                self.update_counts[level] += 1
                self.update_history[level].append(step)

    def get_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get update statistics for all levels.

        Returns:
            Dict with statistics for each level
        """
        stats = {}

        for level in range(self.num_levels):
            count = self.update_counts[level]
            history = self.update_history[level]

            if len(history) > 1:
                intervals = np.diff(history)
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
            else:
                avg_interval = 0
                std_interval = 0

            stats[level] = {
                'update_count': count,
                'last_update': history[-1] if history else -1,
                'avg_interval': avg_interval,
                'std_interval': std_interval
            }

        return stats

    def reset(self) -> None:
        """Reset all tracking."""
        self.update_counts = {i: 0 for i in range(self.num_levels)}
        self.update_history = {i: [] for i in range(self.num_levels)}


class MemoryTracker:
    """
    Track memory usage during training.

    Example:
        >>> tracker = MemoryTracker()
        >>> 
        >>> # During training
        >>> if hasattr(model, 'get_memory_stats'):
        ...     stats = model.get_memory_stats()
        ...     tracker.record(stats, step=100)
    """

    def __init__(self):
        self.memory_history = []

    def record(self, memory_stats: Dict, step: int) -> None:
        """
        Record memory statistics.

        Args:
            memory_stats: Memory statistics from model
            step: Current training step
        """
        self.memory_history.append({
            'step': step,
            'stats': memory_stats
        })

    def get_utilization_over_time(self) -> Dict[str, List[float]]:
        """
        Get memory utilization over time for each level.

        Returns:
            Dict mapping level to utilization history
        """
        utilization = defaultdict(list)

        for entry in self.memory_history:
            stats = entry['stats']
            if isinstance(stats, list):
                # Hope model format
                for block in stats:
                    for level_key, level_stats in block.get('memory', {}).items():
                        utilization[level_key].append(
                            level_stats.get('utilization', 0)
                        )
            elif isinstance(stats, dict):
                # CMSBlock format
                for level_key, level_stats in stats.items():
                    utilization[level_key].append(
                        level_stats.get('utilization', 0)
                    )

        return dict(utilization)

    def get_average_utilization(self) -> Dict[str, float]:
        """
        Get average memory utilization for each level.

        Returns:
            Dict mapping level to average utilization
        """
        utilization = self.get_utilization_over_time()
        return {
            level: np.mean(values) if values else 0.0
            for level, values in utilization.items()
        }

    def reset(self) -> None:
        """Reset tracking."""
        self.memory_history = []
