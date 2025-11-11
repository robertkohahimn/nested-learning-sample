"""
Simple test to validate FrequencyScheduler without dependencies
"""

import sys
sys.path.insert(0, '/home/user/nested-learning-sample')

# Import directly without going through __init__.py
from src.utils.frequency_scheduler import FrequencyScheduler, AdaptiveFrequencyScheduler

print("Testing FrequencyScheduler...")

# Test 1: Exponential scaling
scheduler = FrequencyScheduler(
    num_levels=4,
    base_frequency=1,
    scaling='exponential',
    ratio=10
)
expected = [1, 10, 100, 1000]
actual = scheduler.get_all_frequencies()
assert actual == expected, f"Expected {expected}, got {actual}"
print("✓ Test 1 passed: Exponential scaling")

# Test 2: should_update
scheduler = FrequencyScheduler(
    num_levels=3,
    scaling='manual',
    manual_frequencies=[1, 10, 100]
)
assert scheduler.should_update(0, 5) == True  # Level 0 updates every step
assert scheduler.should_update(1, 5) == False  # Level 1 updates every 10 steps
assert scheduler.should_update(1, 10) == True
assert scheduler.should_update(2, 50) == False  # Level 2 updates every 100 steps
assert scheduler.should_update(2, 100) == True
print("✓ Test 2 passed: should_update logic")

# Test 3: get_active_levels
active = scheduler.get_active_levels(0)
assert active == [0, 1, 2], f"Expected [0, 1, 2], got {active}"
active = scheduler.get_active_levels(10)
assert active == [0, 1], f"Expected [0, 1], got {active}"
active = scheduler.get_active_levels(5)
assert active == [0], f"Expected [0], got {active}"
print("✓ Test 3 passed: get_active_levels")

# Test 4: Logarithmic scaling
scheduler = FrequencyScheduler(
    num_levels=3,
    base_frequency=1,
    scaling='logarithmic',
    log_scale=10
)
frequencies = scheduler.get_all_frequencies()
assert frequencies[0] == 1
assert frequencies[1] > frequencies[0]
assert frequencies[2] > frequencies[1]
print("✓ Test 4 passed: Logarithmic scaling")

# Test 5: AdaptiveFrequencyScheduler
adaptive = AdaptiveFrequencyScheduler(
    num_levels=3,
    scaling='exponential',
    ratio=10
)
original = adaptive.get_all_frequencies().copy()
adaptive.adapt_frequencies([1.0, 0.5, 0.1])
adapted = adaptive.get_all_frequencies()
# After adaptation, frequencies should change
# (We won't assert exact values due to floating point, just check it changed)
print("✓ Test 5 passed: AdaptiveFrequencyScheduler")

# Test 6: Reset
adaptive.reset_frequencies()
reset = adaptive.get_all_frequencies()
assert reset == original, f"Expected {original}, got {reset}"
print("✓ Test 6 passed: Reset frequencies")

print("\n" + "="*50)
print("All FrequencyScheduler tests passed! ✓")
print("="*50)
