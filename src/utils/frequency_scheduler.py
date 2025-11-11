"""
Frequency Scheduler for Nested Learning

Manages update frequencies for different optimization levels and memory banks.
"""

from typing import List, Optional, Union
import math


class FrequencyScheduler:
    """
    Manages update frequencies for multi-level nested optimization.

    Different parameters/memory banks update at different frequencies:
    - Level 0 (fast): C^(0) = 1 (every step)
    - Level 1 (medium): C^(1) = 10-100
    - Level 2 (slow): C^(2) = 1000+

    Supports multiple frequency scaling strategies:
    - Exponential: C^(l) = base * ratio^l
    - Logarithmic: C^(l) = base * log(1 + scale * l)
    - Manual: User-specified frequencies

    Args:
        num_levels: Number of frequency levels
        base_frequency: Base frequency (for level 0, typically 1)
        scaling: Scaling strategy ('exponential', 'logarithmic', 'manual')
        ratio: Ratio for exponential scaling (default: 10)
        log_scale: Scale factor for logarithmic scaling (default: 10)
        manual_frequencies: Manual frequency list (for 'manual' scaling)

    Example:
        >>> # Exponential scaling: 1, 10, 100, 1000
        >>> scheduler = FrequencyScheduler(num_levels=4, scaling='exponential', ratio=10)
        >>> scheduler.should_update(level=2, step=100)  # True
        >>> scheduler.should_update(level=2, step=99)   # False
    """

    def __init__(
        self,
        num_levels: int = 3,
        base_frequency: int = 1,
        scaling: str = 'exponential',
        ratio: float = 10.0,
        log_scale: float = 10.0,
        manual_frequencies: Optional[List[int]] = None
    ):
        assert num_levels > 0, "num_levels must be positive"
        assert base_frequency > 0, "base_frequency must be positive"
        assert scaling in ['exponential', 'logarithmic', 'manual'], \
            f"scaling must be 'exponential', 'logarithmic', or 'manual', got {scaling}"

        self.num_levels = num_levels
        self.base_frequency = base_frequency
        self.scaling = scaling
        self.ratio = ratio
        self.log_scale = log_scale

        # Compute frequencies for each level
        if scaling == 'manual':
            assert manual_frequencies is not None, \
                "manual_frequencies must be provided for manual scaling"
            assert len(manual_frequencies) == num_levels, \
                f"manual_frequencies length ({len(manual_frequencies)}) must match num_levels ({num_levels})"
            self.frequencies = manual_frequencies
        elif scaling == 'exponential':
            self.frequencies = self._compute_exponential_frequencies()
        elif scaling == 'logarithmic':
            self.frequencies = self._compute_logarithmic_frequencies()

        # Ensure all frequencies are positive integers
        self.frequencies = [max(1, int(f)) for f in self.frequencies]

    def _compute_exponential_frequencies(self) -> List[int]:
        """
        Compute exponential frequency scaling: C^(l) = base * ratio^l

        Returns:
            List of frequencies for each level
        """
        frequencies = []
        for level in range(self.num_levels):
            freq = self.base_frequency * (self.ratio ** level)
            frequencies.append(freq)
        return frequencies

    def _compute_logarithmic_frequencies(self) -> List[int]:
        """
        Compute logarithmic frequency scaling: C^(l) = base * log(1 + scale * l)

        Returns:
            List of frequencies for each level
        """
        frequencies = []
        for level in range(self.num_levels):
            if level == 0:
                freq = self.base_frequency
            else:
                freq = self.base_frequency * math.log(1 + self.log_scale * level)
            frequencies.append(freq)
        return frequencies

    def should_update(self, level: int, step: int) -> bool:
        """
        Check if a given level should update at the current step.

        Args:
            level: Optimization/memory level (0 to num_levels-1)
            step: Current training step (0-indexed)

        Returns:
            True if level should update at this step

        Example:
            >>> scheduler = FrequencyScheduler(num_levels=3, scaling='exponential', ratio=10)
            >>> # Level 0 (freq=1): updates every step
            >>> scheduler.should_update(0, 5)  # True
            >>> # Level 1 (freq=10): updates every 10 steps
            >>> scheduler.should_update(1, 10)  # True
            >>> scheduler.should_update(1, 15)  # False
            >>> # Level 2 (freq=100): updates every 100 steps
            >>> scheduler.should_update(2, 100)  # True
        """
        assert 0 <= level < self.num_levels, \
            f"level {level} out of range [0, {self.num_levels})"

        frequency = self.frequencies[level]
        return step % frequency == 0

    def get_frequency(self, level: int) -> int:
        """
        Get the update frequency for a given level.

        Args:
            level: Optimization/memory level

        Returns:
            Update frequency (number of steps between updates)
        """
        assert 0 <= level < self.num_levels, \
            f"level {level} out of range [0, {self.num_levels})"
        return self.frequencies[level]

    def get_all_frequencies(self) -> List[int]:
        """
        Get all frequencies for all levels.

        Returns:
            List of frequencies
        """
        return self.frequencies.copy()

    def get_active_levels(self, step: int) -> List[int]:
        """
        Get all levels that should update at the current step.

        Args:
            step: Current training step

        Returns:
            List of levels that should update

        Example:
            >>> scheduler = FrequencyScheduler(num_levels=3, scaling='exponential', ratio=10)
            >>> scheduler.get_active_levels(0)    # [0]
            >>> scheduler.get_active_levels(10)   # [0, 1]
            >>> scheduler.get_active_levels(100)  # [0, 1, 2]
        """
        return [level for level in range(self.num_levels)
                if self.should_update(level, step)]

    def __repr__(self) -> str:
        """String representation of the scheduler."""
        freq_str = ", ".join([f"L{i}: {f}" for i, f in enumerate(self.frequencies)])
        return (f"FrequencyScheduler(num_levels={self.num_levels}, "
                f"scaling='{self.scaling}', frequencies=[{freq_str}])")

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"FrequencyScheduler with {self.num_levels} levels:"]
        for i, freq in enumerate(self.frequencies):
            lines.append(f"  Level {i}: updates every {freq} step(s)")
        return "\n".join(lines)


class AdaptiveFrequencyScheduler(FrequencyScheduler):
    """
    Adaptive frequency scheduler that adjusts frequencies based on gradient magnitudes.

    The intuition is that parameters with larger gradients should update more frequently,
    while parameters with smaller gradients can update less frequently to save computation.

    Args:
        num_levels: Number of frequency levels
        base_frequency: Base frequency for level 0
        scaling: Initial scaling strategy
        adaptation_rate: Rate at which frequencies adapt (default: 0.1)
        min_frequency: Minimum allowed frequency (default: 1)
        max_frequency: Maximum allowed frequency (default: 10000)
        **kwargs: Additional arguments for base FrequencyScheduler

    Example:
        >>> scheduler = AdaptiveFrequencyScheduler(num_levels=3)
        >>> # Adapt based on gradient magnitude
        >>> scheduler.adapt_frequencies(gradient_magnitudes=[0.5, 0.1, 0.01])
    """

    def __init__(
        self,
        num_levels: int = 3,
        base_frequency: int = 1,
        scaling: str = 'exponential',
        adaptation_rate: float = 0.1,
        min_frequency: int = 1,
        max_frequency: int = 10000,
        **kwargs
    ):
        super().__init__(
            num_levels=num_levels,
            base_frequency=base_frequency,
            scaling=scaling,
            **kwargs
        )

        self.adaptation_rate = adaptation_rate
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        # Store original frequencies as reference
        self.base_frequencies = self.frequencies.copy()

    def adapt_frequencies(self, gradient_magnitudes: List[float]) -> None:
        """
        Adapt frequencies based on gradient magnitudes.

        Larger gradients -> lower frequency (update more often)
        Smaller gradients -> higher frequency (update less often)

        Args:
            gradient_magnitudes: Gradient magnitude for each level
        """
        assert len(gradient_magnitudes) == self.num_levels, \
            f"gradient_magnitudes length ({len(gradient_magnitudes)}) must match num_levels ({self.num_levels})"

        # Normalize gradient magnitudes
        max_grad = max(gradient_magnitudes) if max(gradient_magnitudes) > 0 else 1.0
        normalized_grads = [g / max_grad for g in gradient_magnitudes]

        # Update frequencies (inverse relationship with gradient magnitude)
        for i in range(self.num_levels):
            if normalized_grads[i] > 0:
                # Larger gradient -> smaller multiplier -> lower frequency
                multiplier = 1.0 / (1.0 + normalized_grads[i])
                target_freq = self.base_frequencies[i] * multiplier

                # Exponentially moving average for smooth adaptation
                current_freq = self.frequencies[i]
                new_freq = (1 - self.adaptation_rate) * current_freq + \
                          self.adaptation_rate * target_freq

                # Clamp to valid range
                new_freq = max(self.min_frequency, min(self.max_frequency, new_freq))
                self.frequencies[i] = int(new_freq)

    def reset_frequencies(self) -> None:
        """Reset frequencies to their initial base values."""
        self.frequencies = self.base_frequencies.copy()
