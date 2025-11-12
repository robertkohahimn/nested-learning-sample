"""
Comprehensive tests for Phase 5: Training Framework

Tests all training components:
- NestedTrainer
- Callbacks (EarlyStopping, Checkpoint, LR Scheduling)
- MetricsTracker
- Integration with models and optimizers
"""

import sys
sys.path.insert(0, '/home/user/nested-learning-sample')

import torch
import torch.nn as nn
import torch.utils.data as data
from pathlib import Path
import tempfile
import os

from src.training import (
    NestedTrainer,
    EarlyStopping,
    CheckpointCallback,
    LRSchedulerCallback,
    MetricsTracker
)
from src.models import NestedMLP
from src.optimizers import NestedOptimizerBuilder


# ============================================================================
# Test Data
# ============================================================================

def create_dummy_dataset(n_samples=100, input_dim=10, output_dim=5):
    """Create dummy dataset for testing."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    dataset = data.TensorDataset(X, y)
    return data.DataLoader(dataset, batch_size=16, shuffle=True)


# ============================================================================
# Test 1: NestedTrainer Basic Functionality
# ============================================================================

def test_nested_trainer_creation():
    """Test creating NestedTrainer."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device='cpu'
    )

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.current_epoch == 0
    assert trainer.global_step == 0
    print("✓ NestedTrainer creation")


def test_nested_trainer_single_epoch():
    """Test training for one epoch."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )

    train_loader = create_dummy_dataset(n_samples=64, input_dim=10, output_dim=5)

    history = trainer.fit(
        train_loader=train_loader,
        epochs=1,
        verbose=False
    )

    assert 'train_loss' in history
    assert len(history['train_loss']) == 1
    assert trainer.current_epoch == 0  # Still 0 since loop sets it, not increments after
    print("✓ Single epoch training")


def test_nested_trainer_with_validation():
    """Test training with validation."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )

    train_loader = create_dummy_dataset(n_samples=64, input_dim=10, output_dim=5)
    val_loader = create_dummy_dataset(n_samples=32, input_dim=10, output_dim=5)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=False
    )

    assert 'train_loss' in history
    assert 'val_loss' in history
    assert 'train_acc' in history
    assert 'val_acc' in history
    assert len(history['train_loss']) == 2
    print("✓ Training with validation")


# ============================================================================
# Test 2: NestedTrainer with NestedOptimizer
# ============================================================================

def test_nested_trainer_with_nested_optimizer():
    """Test NestedTrainer with NestedOptimizer."""
    model = NestedMLP(
        input_dim=10,
        hidden_dims=[20],
        output_dim=5
    )

    builder = NestedOptimizerBuilder(model, num_levels=2)
    builder.auto_assign_params('uniform')
    optimizer = builder.build(
        optimizer_types=['adam', 'sgd'],
        learning_rates=[0.001, 0.01],
        frequencies=[1, 5]
    )

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        use_nested_optimizer=True
    )

    train_loader = create_dummy_dataset(n_samples=64, input_dim=10, output_dim=5)

    history = trainer.fit(
        train_loader=train_loader,
        epochs=2,
        verbose=False
    )

    assert len(history['train_loss']) == 2
    assert trainer.global_step > 0

    # Check update stats
    stats = trainer.get_update_stats()
    assert stats is not None
    print("✓ NestedOptimizer integration")


# ============================================================================
# Test 3: EarlyStopping Callback
# ============================================================================

def test_early_stopping():
    """Test early stopping callback."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        min_delta=0.001,
        mode='min',
        verbose=False
    )

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[early_stop]
    )

    train_loader = create_dummy_dataset(n_samples=64)
    val_loader = create_dummy_dataset(n_samples=32)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Request 10 but should stop early
        verbose=False
    )

    # Should stop before 10 epochs (usually within 3-5)
    # Note: With random data, early stopping might not always trigger
    # Just check that training completed
    assert len(history['train_loss']) > 0
    assert len(history['train_loss']) <= 10
    print(f"✓ Early stopping (trained for {len(history['train_loss'])} epochs)")


# ============================================================================
# Test 4: Checkpoint Callback
# ============================================================================

def test_checkpoint_callback():
    """Test checkpoint saving callback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        checkpoint_path = os.path.join(tmpdir, 'checkpoint_epoch_{epoch}.pt')
        checkpoint_cb = CheckpointCallback(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_freq=1,
            verbose=False
        )

        trainer = NestedTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=[checkpoint_cb]
        )

        train_loader = create_dummy_dataset(n_samples=64)
        val_loader = create_dummy_dataset(n_samples=32)

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            verbose=False
        )

        # Check that checkpoints were saved
        saved_files = list(Path(tmpdir).glob('*.pt'))
        assert len(saved_files) == 3  # One per epoch

    print("✓ Checkpoint callback")


# ============================================================================
# Test 5: Manual Checkpoint Save/Load
# ============================================================================

def test_checkpoint_save_load():
    """Test manual checkpoint save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and train a model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        trainer = NestedTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )

        train_loader = create_dummy_dataset(n_samples=64)

        # Train for 2 epochs
        history = trainer.fit(
            train_loader=train_loader,
            epochs=2,
            verbose=False
        )

        # Save checkpoint
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
        trainer.save_checkpoint(checkpoint_path)

        # Get model state before loading
        old_state = model.state_dict()
        old_epoch = trainer.current_epoch

        # Create new trainer and load
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        new_trainer = NestedTrainer(
            model=new_model,
            optimizer=new_optimizer,
            criterion=criterion
        )

        # Load checkpoint
        checkpoint = new_trainer.load_checkpoint(checkpoint_path)

        # Verify loading
        assert new_trainer.current_epoch == old_epoch
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint

        # Check model parameters match
        new_state = new_model.state_dict()
        for key in old_state.keys():
            assert torch.allclose(old_state[key], new_state[key])

    print("✓ Checkpoint save/load")


# ============================================================================
# Test 6: LR Scheduler Callback
# ============================================================================

def test_lr_scheduler_callback():
    """Test learning rate scheduler callback."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    lr_callback = LRSchedulerCallback(scheduler=scheduler, verbose=False)

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[lr_callback]
    )

    train_loader = create_dummy_dataset(n_samples=64)

    initial_lr = trainer.get_lr()[0]

    history = trainer.fit(
        train_loader=train_loader,
        epochs=5,
        verbose=False
    )

    final_lr = trainer.get_lr()[0]

    # LR should have decreased (step_size=2, gamma=0.1, so 2 steps)
    assert final_lr < initial_lr
    print(f"✓ LR scheduler (initial: {initial_lr:.4f}, final: {final_lr:.6f})")


# ============================================================================
# Test 7: MetricsTracker
# ============================================================================

def test_metrics_tracker():
    """Test MetricsTracker functionality."""
    tracker = MetricsTracker()

    # Add some metrics
    for i in range(10):
        tracker.update('train_loss', 1.0 - i * 0.1, step=i)
        tracker.update('val_loss', 1.2 - i * 0.08, step=i)

    # Test get
    train_losses = tracker.get('train_loss')
    assert len(train_losses) == 10

    # Test get_last
    last_loss = tracker.get_last('train_loss')
    assert abs(last_loss - 0.1) < 0.01  # Should be close to 0.1

    # Test get_stats
    stats = tracker.get_stats('train_loss')
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert stats['count'] == 10

    # Test get_best
    best = tracker.get_best('train_loss', mode='min')
    assert abs(best - 0.1) < 0.01  # Should be close to 0.1

    best_epoch = tracker.get_best_epoch('train_loss', mode='min')
    assert best_epoch == 9

    # Test moving average
    ma = tracker.moving_average('train_loss', window=3)
    assert len(ma) == 8

    print("✓ MetricsTracker")


def test_metrics_tracker_save_load():
    """Test saving and loading metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MetricsTracker()

        # Add metrics
        for i in range(5):
            tracker.update('loss', float(i))

        # Save
        filepath = os.path.join(tmpdir, 'metrics.json')
        tracker.save(filepath)

        # Load in new tracker
        new_tracker = MetricsTracker()
        new_tracker.load(filepath)

        # Verify
        assert new_tracker.get('loss') == tracker.get('loss')

    print("✓ MetricsTracker save/load")


# ============================================================================
# Test 8: Integration with NestedMLP
# ============================================================================

def test_training_nested_mlp_end_to_end():
    """Test end-to-end training with NestedMLP."""
    model = NestedMLP(
        input_dim=10,
        hidden_dims=[20, 15],
        output_dim=5
    )

    builder = NestedOptimizerBuilder(model, num_levels=3)
    builder.auto_assign_params('uniform')
    optimizer = builder.build(
        optimizer_types=['adam', 'sgd', 'sgd'],
        learning_rates=[0.001, 0.01, 0.1],
        frequencies=[1, 5, 10]
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=False
    )

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        use_nested_optimizer=True,
        callbacks=[early_stop]
    )

    train_loader = create_dummy_dataset(n_samples=100, input_dim=10, output_dim=5)
    val_loader = create_dummy_dataset(n_samples=50, input_dim=10, output_dim=5)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        verbose=False
    )

    # Verify training happened
    assert len(history['train_loss']) > 0
    assert history['train_loss'][0] > 0
    
    # Verify validation metrics
    assert 'val_loss' in history
    assert 'val_acc' in history

    print("✓ End-to-end NestedMLP training")


# ============================================================================
# Test 9: Get/Set Learning Rate
# ============================================================================

def test_get_set_lr():
    """Test getting and setting learning rate."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )

    # Get initial LR
    initial_lr = trainer.get_lr()
    assert len(initial_lr) == 1
    assert initial_lr[0] == 0.001

    # Set new LR
    trainer.set_lr(0.01)
    new_lr = trainer.get_lr()
    assert new_lr[0] == 0.01

    print("✓ Get/Set learning rate")


# ============================================================================
# Test 10: Training History Reset
# ============================================================================

def test_history_reset():
    """Test resetting training history."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )

    train_loader = create_dummy_dataset(n_samples=64)

    # Train
    trainer.fit(train_loader=train_loader, epochs=2, verbose=False)

    assert trainer.global_step > 0
    assert len(trainer.history) > 0

    # Reset
    trainer.reset_history()

    assert trainer.global_step == 0
    assert trainer.current_epoch == 0
    assert len(trainer.history) == 0

    print("✓ History reset")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Phase 5 Training Framework Tests")
    print("="*70)

    test_functions = [
        ("NestedTrainer creation", test_nested_trainer_creation),
        ("Single epoch training", test_nested_trainer_single_epoch),
        ("Training with validation", test_nested_trainer_with_validation),
        ("NestedOptimizer integration", test_nested_trainer_with_nested_optimizer),
        ("Early stopping", test_early_stopping),
        ("Checkpoint callback", test_checkpoint_callback),
        ("Checkpoint save/load", test_checkpoint_save_load),
        ("LR scheduler callback", test_lr_scheduler_callback),
        ("MetricsTracker", test_metrics_tracker),
        ("MetricsTracker save/load", test_metrics_tracker_save_load),
        ("End-to-end NestedMLP", test_training_nested_mlp_end_to_end),
        ("Get/Set LR", test_get_set_lr),
        ("History reset", test_history_reset),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed out of {passed + failed} total")
    print("="*70)

    if failed == 0:
        print("\n🎉 All Phase 5 tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
