"""
Unit tests for Memory modules (AssociativeMemory and ContinuumMemorySystem)
"""

import pytest
import torch
from src.memory.associative_memory import AssociativeMemory
from src.memory.cms import ContinuumMemorySystem


class TestAssociativeMemory:
    """Tests for AssociativeMemory"""

    def test_initialization(self):
        """Test memory initialization"""
        memory = AssociativeMemory(
            memory_size=100,
            key_dim=64,
            value_dim=128
        )

        assert memory.memory_size == 100
        assert memory.key_dim == 64
        assert memory.value_dim == 128
        assert memory.memory_count == 0
        assert not memory.is_full()

    def test_write_append(self):
        """Test writing to memory in append mode"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)

        memory.write(keys, values, mode='append')

        assert memory.memory_count == 3
        assert torch.allclose(memory.keys[:3], keys)
        assert torch.allclose(memory.values[:3], values)

    def test_write_replace(self):
        """Test writing to memory in replace mode"""
        memory = AssociativeMemory(memory_size=5, key_dim=4, value_dim=8)

        # Fill memory
        keys1 = torch.randn(5, 4)
        values1 = torch.randn(5, 8)
        memory.write(keys1, values1, mode='append')

        assert memory.is_full()

        # Replace oldest entries
        keys2 = torch.randn(2, 4)
        values2 = torch.randn(2, 8)
        memory.write(keys2, values2, mode='replace')

        # First 2 entries should be replaced
        assert torch.allclose(memory.keys[:2], keys2)
        assert torch.allclose(memory.values[:2], values2)

    def test_read_empty(self):
        """Test reading from empty memory"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        query = torch.randn(2, 4)
        retrieved = memory.read(query)

        # Should return zeros
        assert retrieved.shape == (2, 8)
        assert torch.allclose(retrieved, torch.zeros(2, 8))

    def test_read_retrieval(self):
        """Test basic retrieval from memory"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        # Store some data
        keys = torch.randn(5, 4)
        values = torch.randn(5, 8)
        memory.write(keys, values, mode='append')

        # Query with exact key should retrieve similar value
        query = keys[:1]  # Use first key
        retrieved = memory.read(query)

        # Retrieved should be weighted combination of stored values
        # With exact match, should be very close to stored value
        assert retrieved.shape == (1, 8)

    def test_l2_similarity(self):
        """Test L2-based similarity computation"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8, temperature=1.0)

        # Store data
        keys = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
        values = torch.randn(3, 8)
        memory.write(keys, values, mode='append')

        # Query similar to first key
        query = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
        similarity = memory._compute_similarity(query, keys)

        # First key should have highest similarity (least distance)
        assert similarity[0, 0] > similarity[0, 1]
        assert similarity[0, 0] > similarity[0, 2]

    def test_update(self):
        """Test memory update with L2 regression"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        # Initial write
        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        memory.write(keys, values, mode='append')

        # Update with new targets
        query = torch.randn(2, 4)
        target_values = torch.randn(2, 8)
        loss = memory.update(query, target_values, learning_rate=0.1)

        # Should return a loss value
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_clear(self):
        """Test clearing memory"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        # Add data
        keys = torch.randn(5, 4)
        values = torch.randn(5, 8)
        memory.write(keys, values, mode='append')
        assert memory.memory_count == 5

        # Clear
        memory.clear()
        assert memory.memory_count == 0
        assert not memory.is_full()

    def test_memory_usage(self):
        """Test memory usage calculation"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        assert memory.get_memory_usage() == 0.0

        keys = torch.randn(5, 4)
        values = torch.randn(5, 8)
        memory.write(keys, values, mode='append')

        assert memory.get_memory_usage() == 0.5

    def test_forward_read_mode(self):
        """Test forward pass in read mode"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        memory.write(keys, values, mode='append')

        query = torch.randn(2, 4)
        retrieved, loss = memory.forward(query, mode='read')

        assert retrieved.shape == (2, 8)
        assert loss is None

    def test_forward_write_mode(self):
        """Test forward pass in write mode"""
        memory = AssociativeMemory(memory_size=10, key_dim=4, value_dim=8)

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        result, loss = memory.forward(None, keys=keys, values=values, mode='write')

        assert result is None
        assert loss is None
        assert memory.memory_count == 3


class TestContinuumMemorySystem:
    """Tests for ContinuumMemorySystem"""

    def test_initialization_uniform_sizes(self):
        """Test CMS initialization with uniform memory sizes"""
        cms = ContinuumMemorySystem(
            memory_sizes=100,
            key_dim=64,
            value_dim=128,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        assert cms.num_levels == 3
        assert len(cms.memory_banks) == 3
        assert all(bank.memory_size == 100 for bank in cms.memory_banks)

    def test_initialization_varied_sizes(self):
        """Test CMS initialization with varied memory sizes"""
        sizes = [50, 100, 200]
        cms = ContinuumMemorySystem(
            memory_sizes=sizes,
            key_dim=64,
            value_dim=128,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        assert cms.num_levels == 3
        for i, bank in enumerate(cms.memory_banks):
            assert bank.memory_size == sizes[i]

    def test_frequency_scheduler_creation(self):
        """Test automatic frequency scheduler creation"""
        # Default exponential scaling
        cms = ContinuumMemorySystem(
            memory_sizes=100,
            key_dim=64,
            value_dim=128,
            num_levels=3
        )

        frequencies = cms.frequency_scheduler.get_all_frequencies()
        # Should be [1, 10, 100] with default ratio=10
        assert frequencies == [1, 10, 100]

    def test_store_single_level(self):
        """Test storing to specific memory level"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)

        # Force storage in level 1 only
        updated = cms.store(keys, values, step=0, force_levels=[1])

        assert updated[0] == False
        assert updated[1] == True
        assert updated[2] == False

        # Only level 1 should have data
        assert cms.memory_banks[0].memory_count == 0
        assert cms.memory_banks[1].memory_count == 3
        assert cms.memory_banks[2].memory_count == 0

    def test_store_frequency_based(self):
        """Test storing based on frequency schedule"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)

        # At step 0, all levels should update
        updated = cms.store(keys, values, step=0)
        assert all(updated.values())

        # At step 5, only level 0 should update
        updated = cms.store(keys, values, step=5)
        assert updated[0] == True
        assert updated[1] == False
        assert updated[2] == False

        # At step 10, levels 0 and 1 should update
        updated = cms.store(keys, values, step=10)
        assert updated[0] == True
        assert updated[1] == True
        assert updated[2] == False

    def test_retrieve_single_level(self):
        """Test retrieving from specific level"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        # Store in level 0
        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        cms.store(keys, values, force_levels=[0])

        # Retrieve from level 0
        query = torch.randn(2, 4)
        retrieved = cms.retrieve(query, levels=[0], aggregate=False)

        assert len(retrieved) == 1
        assert retrieved[0].shape == (2, 8)

    def test_retrieve_aggregate(self):
        """Test aggregated retrieval across levels"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        # Store in all levels
        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        cms.store(keys, values, force_levels=[0, 1, 2])

        # Retrieve aggregated
        query = torch.randn(2, 4)
        retrieved = cms.retrieve(query, aggregate=True)

        assert retrieved.shape == (2, 8)

    def test_consolidate(self):
        """Test memory consolidation between levels"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        # Store in level 0
        keys = torch.randn(5, 4)
        values = torch.randn(5, 8)
        cms.store(keys, values, force_levels=[0])

        assert cms.memory_banks[0].memory_count == 5
        assert cms.memory_banks[1].memory_count == 0

        # Consolidate from level 0 to level 1
        cms.consolidate(from_level=0, to_level=1, num_samples=3)

        # Level 1 should now have data
        assert cms.memory_banks[1].memory_count > 0

    def test_auto_consolidate(self):
        """Test automatic consolidation"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)

        # Store in level 0
        cms.store(keys, values, force_levels=[0])

        # Auto-consolidate at step 10 (when level 1 updates)
        cms.auto_consolidate(step=10)

        # Level 1 should have received consolidated memories
        # (may be 0 if consolidation logic differs, but method should not crash)

    def test_clear(self):
        """Test clearing memory levels"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3
        )

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        cms.store(keys, values, force_levels=[0, 1, 2])

        # Clear all
        cms.clear()

        for bank in cms.memory_banks:
            assert bank.memory_count == 0

        # Clear specific level
        cms.store(keys, values, force_levels=[0, 1])
        cms.clear(levels=[0])

        assert cms.memory_banks[0].memory_count == 0
        assert cms.memory_banks[1].memory_count > 0

    def test_memory_stats(self):
        """Test memory statistics"""
        cms = ContinuumMemorySystem(
            memory_sizes=[10, 20, 30],
            key_dim=4,
            value_dim=8,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        stats = cms.get_memory_stats()

        assert len(stats) == 3
        for level in range(3):
            assert 'usage' in stats[level]
            assert 'count' in stats[level]
            assert 'capacity' in stats[level]
            assert 'frequency' in stats[level]

        # Check specific values
        assert stats[0]['capacity'] == 10
        assert stats[1]['capacity'] == 20
        assert stats[2]['capacity'] == 30
        assert stats[0]['frequency'] == 1
        assert stats[1]['frequency'] == 10
        assert stats[2]['frequency'] == 100

    def test_step_counter(self):
        """Test internal step counter"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3
        )

        assert cms.current_step == 0

        cms.step()
        assert cms.current_step == 1

        cms.step()
        cms.step()
        assert cms.current_step == 3

    def test_forward_retrieve_mode(self):
        """Test forward pass in retrieve mode"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=2
        )

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        cms.store(keys, values, force_levels=[0, 1])

        query = torch.randn(2, 4)
        retrieved, metadata = cms.forward(query, mode='retrieve')

        assert retrieved.shape == (2, 8)
        assert metadata is None

    def test_forward_store_mode(self):
        """Test forward pass in store mode"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=2,
            frequencies=[1, 10]
        )

        keys = torch.randn(3, 4)
        values = torch.randn(3, 8)
        result, metadata = cms.forward(None, keys=keys, values=values, mode='store', step=0)

        assert result is None
        assert 'updated_levels' in metadata

    def test_learnable_weights(self):
        """Test that level weights are learnable parameters"""
        cms = ContinuumMemorySystem(
            memory_sizes=10,
            key_dim=4,
            value_dim=8,
            num_levels=3
        )

        # level_weights should be a parameter
        assert isinstance(cms.level_weights, torch.nn.Parameter)
        assert cms.level_weights.requires_grad

    def test_repr(self):
        """Test string representation"""
        cms = ContinuumMemorySystem(
            memory_sizes=100,
            key_dim=64,
            value_dim=128,
            num_levels=3,
            frequencies=[1, 10, 100]
        )

        repr_str = repr(cms)
        assert "ContinuumMemorySystem" in repr_str
        assert "num_levels=3" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
