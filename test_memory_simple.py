"""
Simple test to validate Memory components
"""

import sys
sys.path.insert(0, '/home/user/nested-learning-sample')

import torch

# Import directly
from src.memory.associative_memory import AssociativeMemory
from src.memory.cms import ContinuumMemorySystem
from src.utils.frequency_scheduler import FrequencyScheduler

print("Testing AssociativeMemory...")

# Test 1: Initialization
memory = AssociativeMemory(memory_size=100, key_dim=64, value_dim=128)
assert memory.memory_size == 100
assert memory.key_dim == 64
assert memory.value_dim == 128
assert memory.memory_count == 0
assert not memory.is_full()
print("✓ Test 1 passed: Initialization")

# Test 2: Write and read
keys = torch.randn(5, 64)
values = torch.randn(5, 128)
memory.write(keys, values, mode='append')
assert memory.memory_count == 5
print("✓ Test 2 passed: Write operation")

# Test 3: Read
query = torch.randn(2, 64)
retrieved = memory.read(query)
assert retrieved.shape == (2, 128)
print("✓ Test 3 passed: Read operation")

# Test 4: Memory usage
usage = memory.get_memory_usage()
assert usage == 0.05  # 5/100
print("✓ Test 4 passed: Memory usage calculation")

# Test 5: Clear
memory.clear()
assert memory.memory_count == 0
print("✓ Test 5 passed: Clear operation")

print("\nTesting ContinuumMemorySystem...")

# Test 6: CMS Initialization
cms = ContinuumMemorySystem(
    memory_sizes=50,
    key_dim=32,
    value_dim=64,
    num_levels=3,
    frequencies=[1, 10, 100]
)
assert cms.num_levels == 3
assert len(cms.memory_banks) == 3
print("✓ Test 6 passed: CMS initialization")

# Test 7: Store operation
keys = torch.randn(3, 32)
values = torch.randn(3, 64)
updated = cms.store(keys, values, step=0)
# At step 0, all levels should update
assert all(updated.values())
print("✓ Test 7 passed: Store operation")

# Test 8: Retrieve operation
query = torch.randn(2, 32)
retrieved = cms.retrieve(query)
assert retrieved.shape == (2, 64)
print("✓ Test 8 passed: Retrieve operation")

# Test 9: Frequency-based updates
updated = cms.store(keys, values, step=5)
# At step 5, only level 0 (freq=1) should update
assert updated[0] == True
assert updated[1] == False
assert updated[2] == False
print("✓ Test 9 passed: Frequency-based updates")

# Test 10: Memory stats
stats = cms.get_memory_stats()
assert len(stats) == 3
assert 'usage' in stats[0]
assert 'count' in stats[0]
assert 'capacity' in stats[0]
assert 'frequency' in stats[0]
print("✓ Test 10 passed: Memory statistics")

# Test 11: Consolidation
cms.clear()
keys = torch.randn(5, 32)
values = torch.randn(5, 64)
cms.store(keys, values, force_levels=[0])
assert cms.memory_banks[0].memory_count == 5
assert cms.memory_banks[1].memory_count == 0

# Consolidate from level 0 to level 1
cms.consolidate(from_level=0, to_level=1, num_samples=3)
assert cms.memory_banks[1].memory_count > 0
print("✓ Test 11 passed: Memory consolidation")

# Test 12: Integration with FrequencyScheduler
scheduler = FrequencyScheduler(num_levels=3, scaling='exponential', ratio=10)
cms2 = ContinuumMemorySystem(
    memory_sizes=50,
    key_dim=32,
    value_dim=64,
    num_levels=3,
    frequency_scheduler=scheduler
)
assert cms2.frequency_scheduler == scheduler
print("✓ Test 12 passed: Integration with FrequencyScheduler")

print("\n" + "="*50)
print("All Memory component tests passed! ✓")
print("="*50)
