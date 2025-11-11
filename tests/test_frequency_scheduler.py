"""
Unit tests for FrequencyScheduler and AdaptiveFrequencyScheduler
"""

import pytest
from src.utils.frequency_scheduler import FrequencyScheduler, AdaptiveFrequencyScheduler


class TestFrequencyScheduler:
    """Tests for FrequencyScheduler"""

    def test_exponential_scaling(self):
        """Test exponential frequency scaling"""
        scheduler = FrequencyScheduler(
            num_levels=4,
            base_frequency=1,
            scaling='exponential',
            ratio=10
        )

        expected_frequencies = [1, 10, 100, 1000]
        assert scheduler.get_all_frequencies() == expected_frequencies

    def test_logarithmic_scaling(self):
        """Test logarithmic frequency scaling"""
        scheduler = FrequencyScheduler(
            num_levels=3,
            base_frequency=1,
            scaling='logarithmic',
            log_scale=10
        )

        frequencies = scheduler.get_all_frequencies()
        # Level 0 should be base frequency
        assert frequencies[0] == 1
        # Subsequent levels should increase logarithmically
        assert frequencies[1] > frequencies[0]
        assert frequencies[2] > frequencies[1]

    def test_manual_frequencies(self):
        """Test manual frequency specification"""
        manual_freqs = [1, 5, 25, 100]
        scheduler = FrequencyScheduler(
            num_levels=4,
            scaling='manual',
            manual_frequencies=manual_freqs
        )

        assert scheduler.get_all_frequencies() == manual_freqs

    def test_should_update(self):
        """Test should_update logic"""
        scheduler = FrequencyScheduler(
            num_levels=3,
            scaling='manual',
            manual_frequencies=[1, 10, 100]
        )

        # Level 0 (freq=1) should update every step
        assert scheduler.should_update(0, 0)
        assert scheduler.should_update(0, 1)
        assert scheduler.should_update(0, 99)

        # Level 1 (freq=10) should update every 10 steps
        assert scheduler.should_update(1, 0)
        assert scheduler.should_update(1, 10)
        assert scheduler.should_update(1, 20)
        assert not scheduler.should_update(1, 5)
        assert not scheduler.should_update(1, 15)

        # Level 2 (freq=100) should update every 100 steps
        assert scheduler.should_update(2, 0)
        assert scheduler.should_update(2, 100)
        assert scheduler.should_update(2, 200)
        assert not scheduler.should_update(2, 50)
        assert not scheduler.should_update(2, 150)

    def test_get_active_levels(self):
        """Test get_active_levels returns correct levels"""
        scheduler = FrequencyScheduler(
            num_levels=3,
            scaling='manual',
            manual_frequencies=[1, 10, 100]
        )

        # At step 0, all levels should update
        assert scheduler.get_active_levels(0) == [0, 1, 2]

        # At step 10, levels 0 and 1 should update
        assert scheduler.get_active_levels(10) == [0, 1]

        # At step 5, only level 0 should update
        assert scheduler.get_active_levels(5) == [0]

        # At step 100, all levels should update
        assert scheduler.get_active_levels(100) == [0, 1, 2]

    def test_get_frequency(self):
        """Test get_frequency for specific level"""
        scheduler = FrequencyScheduler(
            num_levels=3,
            scaling='exponential',
            ratio=10
        )

        assert scheduler.get_frequency(0) == 1
        assert scheduler.get_frequency(1) == 10
        assert scheduler.get_frequency(2) == 100

    def test_invalid_level(self):
        """Test error handling for invalid level"""
        scheduler = FrequencyScheduler(num_levels=3)

        with pytest.raises(AssertionError):
            scheduler.should_update(3, 0)  # Level 3 doesn't exist

        with pytest.raises(AssertionError):
            scheduler.should_update(-1, 0)  # Negative level

    def test_invalid_scaling(self):
        """Test error handling for invalid scaling type"""
        with pytest.raises(AssertionError):
            FrequencyScheduler(num_levels=3, scaling='invalid')

    def test_manual_without_frequencies(self):
        """Test that manual scaling requires frequencies"""
        with pytest.raises(AssertionError):
            FrequencyScheduler(num_levels=3, scaling='manual')

    def test_manual_wrong_length(self):
        """Test that manual frequencies must match num_levels"""
        with pytest.raises(AssertionError):
            FrequencyScheduler(
                num_levels=3,
                scaling='manual',
                manual_frequencies=[1, 10]  # Only 2, need 3
            )

    def test_repr(self):
        """Test string representation"""
        scheduler = FrequencyScheduler(num_levels=2, scaling='exponential', ratio=10)
        repr_str = repr(scheduler)
        assert "FrequencyScheduler" in repr_str
        assert "num_levels=2" in repr_str

    def test_str(self):
        """Test human-readable string"""
        scheduler = FrequencyScheduler(num_levels=2, scaling='exponential', ratio=10)
        str_repr = str(scheduler)
        assert "Level 0" in str_repr
        assert "Level 1" in str_repr


class TestAdaptiveFrequencyScheduler:
    """Tests for AdaptiveFrequencyScheduler"""

    def test_initialization(self):
        """Test adaptive scheduler initialization"""
        scheduler = AdaptiveFrequencyScheduler(
            num_levels=3,
            scaling='exponential',
            ratio=10
        )

        # Should have base frequencies stored
        assert len(scheduler.base_frequencies) == 3
        assert scheduler.base_frequencies == [1, 10, 100]

    def test_adapt_frequencies(self):
        """Test frequency adaptation based on gradients"""
        scheduler = AdaptiveFrequencyScheduler(
            num_levels=3,
            scaling='exponential',
            ratio=10,
            adaptation_rate=1.0  # Fast adaptation for testing
        )

        initial_frequencies = scheduler.get_all_frequencies().copy()

        # Large gradient for level 0 should decrease its frequency (update more often)
        # Small gradient for level 2 should increase its frequency (update less often)
        gradient_magnitudes = [1.0, 0.5, 0.1]
        scheduler.adapt_frequencies(gradient_magnitudes)

        adapted_frequencies = scheduler.get_all_frequencies()

        # Frequencies should have changed
        assert adapted_frequencies != initial_frequencies

    def test_reset_frequencies(self):
        """Test resetting frequencies to base values"""
        scheduler = AdaptiveFrequencyScheduler(
            num_levels=3,
            scaling='exponential',
            ratio=10
        )

        original_frequencies = scheduler.get_all_frequencies().copy()

        # Adapt frequencies
        scheduler.adapt_frequencies([1.0, 0.5, 0.1])
        adapted_frequencies = scheduler.get_all_frequencies()
        assert adapted_frequencies != original_frequencies

        # Reset should restore original
        scheduler.reset_frequencies()
        reset_frequencies = scheduler.get_all_frequencies()
        assert reset_frequencies == original_frequencies

    def test_adaptation_clamping(self):
        """Test that frequencies are clamped to valid range"""
        scheduler = AdaptiveFrequencyScheduler(
            num_levels=3,
            scaling='exponential',
            ratio=10,
            min_frequency=5,
            max_frequency=50,
            adaptation_rate=1.0
        )

        # Try to push frequencies outside bounds
        scheduler.adapt_frequencies([10.0, 1.0, 0.001])

        frequencies = scheduler.get_all_frequencies()

        # All frequencies should be within bounds
        for freq in frequencies:
            assert freq >= 5
            assert freq <= 50

    def test_adapt_wrong_length(self):
        """Test error when gradient_magnitudes length doesn't match"""
        scheduler = AdaptiveFrequencyScheduler(num_levels=3)

        with pytest.raises(AssertionError):
            scheduler.adapt_frequencies([1.0, 0.5])  # Only 2, need 3

    def test_adapt_with_zero_gradients(self):
        """Test adaptation with zero gradients"""
        scheduler = AdaptiveFrequencyScheduler(num_levels=3)

        # Should not crash with zero gradients
        scheduler.adapt_frequencies([0.0, 0.0, 0.0])

        # Frequencies should still be valid
        frequencies = scheduler.get_all_frequencies()
        assert all(f > 0 for f in frequencies)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
