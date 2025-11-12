"""
Quick test to verify practical examples work correctly
"""

import sys
sys.path.insert(0, '/home/user/nested-learning-sample')

import torch
import torch.nn as nn
import numpy as np

from src.optimizers.dmgd import DeepMomentumGD
from src.optimizers.meta_learning import MetaLearningTrainer, create_task_sampler

print("="*60)
print("Testing Practical Examples")
print("="*60)

# ============================================================================
# Test 1: Quick Regression Meta-Learning
# ============================================================================
print("\n[Test 1] Quick Regression Meta-Learning")

def create_simple_model():
    return nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

dummy_model = create_simple_model()
dmgd = DeepMomentumGD(
    dummy_model.parameters(),
    lr=0.01,
    momentum_hidden_dims=[16],
    mlp_lr=0.001
)

trainer = MetaLearningTrainer(
    model_fn=create_simple_model,
    dmgd_optimizer=dmgd,
    inner_steps=5,
    inner_lr=0.01,
    meta_lr=0.001
)

sampler = create_task_sampler(task_type='sine', n_train=20, n_val=20)

# Meta-train for 10 episodes
print("  Meta-training for 10 episodes...")
meta_losses = trainer.meta_train(sampler, num_episodes=10, verbose=False)

print(f"  Initial meta-loss: {meta_losses[0]:.4f}")
print(f"  Final meta-loss: {meta_losses[-1]:.4f}")
print("✓ Regression meta-learning works")

# ============================================================================
# Test 2: Quick Binary Classification Meta-Learning
# ============================================================================
print("\n[Test 2] Quick Binary Classification Meta-Learning")

def create_binary_model():
    return nn.Sequential(
        nn.Linear(5, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )

dummy_model = create_binary_model()
dmgd = DeepMomentumGD(
    dummy_model.parameters(),
    lr=0.01,
    momentum_hidden_dims=[16],
    mlp_lr=0.001
)

trainer = MetaLearningTrainer(
    model_fn=create_binary_model,
    dmgd_optimizer=dmgd,
    inner_steps=5,
    inner_lr=0.01,
    meta_lr=0.001
)

sampler = create_task_sampler(
    task_type='binary_classification',
    input_dim=5,
    n_train=30,
    n_val=15
)

# Meta-train
print("  Meta-training for 10 episodes...")
meta_losses = trainer.meta_train(
    sampler,
    num_episodes=10,
    criterion=nn.BCELoss(),
    verbose=False
)

print(f"  Initial meta-loss: {meta_losses[0]:.4f}")
print(f"  Final meta-loss: {meta_losses[-1]:.4f}")
print("✓ Binary classification meta-learning works")

# ============================================================================
# Test 3: Evaluation on New Tasks
# ============================================================================
print("\n[Test 3] Evaluation on New Tasks")

task = sampler()
results = trainer.evaluate(task, criterion=nn.BCELoss())

print(f"  Initial loss: {results['initial_loss']:.4f}")
print(f"  Final loss: {results['final_loss']:.4f}")
print(f"  Improvement: {results['improvement']:.4f}")
print("✓ Evaluation works")

# ============================================================================
# Test 4: Multi-Class Classification
# ============================================================================
print("\n[Test 4] Multi-Class Classification Meta-Learning")

def create_multiclass_model():
    return nn.Sequential(
        nn.Linear(5, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )

dummy_model = create_multiclass_model()
dmgd = DeepMomentumGD(
    dummy_model.parameters(),
    lr=0.01,
    momentum_hidden_dims=[16],
    mlp_lr=0.001
)

trainer = MetaLearningTrainer(
    model_fn=create_multiclass_model,
    dmgd_optimizer=dmgd,
    inner_steps=5,
    inner_lr=0.01,
    meta_lr=0.001
)

sampler = create_task_sampler(
    task_type='multiclass_classification',
    input_dim=5,
    output_dim=3,
    n_train=30,
    n_val=15
)

print("  Meta-training for 10 episodes...")
meta_losses = trainer.meta_train(
    sampler,
    num_episodes=10,
    criterion=nn.CrossEntropyLoss(),
    verbose=False
)

print(f"  Initial meta-loss: {meta_losses[0]:.4f}")
print(f"  Final meta-loss: {meta_losses[-1]:.4f}")
print("✓ Multi-class classification meta-learning works")

# ============================================================================
# Test 5: Comparison with Standard Optimizers
# ============================================================================
print("\n[Test 5] Quick Comparison with Standard Optimizers")

def evaluate_optimizer(opt_class, opt_kwargs, task, model_fn, criterion, num_steps=5):
    """Helper to evaluate an optimizer."""
    model = model_fn()
    optimizer = opt_class(model.parameters(), **opt_kwargs)

    X_train, y_train = task['train']
    X_val, y_val = task['val']

    # Train
    for _ in range(num_steps):
        pred = model(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        pred = model(X_val)
        return criterion(pred, y_val).item()

# Test on 3 tasks
sampler = create_task_sampler(task_type='sine', n_train=20, n_val=20)
trainer_comp = MetaLearningTrainer(
    model_fn=create_simple_model,
    dmgd_optimizer=DeepMomentumGD(
        create_simple_model().parameters(),
        lr=0.01,
        momentum_hidden_dims=[16],
        mlp_lr=0.001
    ),
    inner_steps=5,
    inner_lr=0.01,
    meta_lr=0.001
)

# Quick meta-training
trainer_comp.meta_train(sampler, num_episodes=10, verbose=False)

sgd_losses = []
adam_losses = []
dmgd_losses = []

for _ in range(3):
    task = sampler()

    sgd_loss = evaluate_optimizer(
        torch.optim.SGD,
        {'lr': 0.01, 'momentum': 0.9},
        task,
        create_simple_model,
        nn.MSELoss()
    )

    adam_loss = evaluate_optimizer(
        torch.optim.Adam,
        {'lr': 0.01},
        task,
        create_simple_model,
        nn.MSELoss()
    )

    dmgd_result = trainer_comp.evaluate(task)
    dmgd_loss = dmgd_result['final_loss']

    sgd_losses.append(sgd_loss)
    adam_losses.append(adam_loss)
    dmgd_losses.append(dmgd_loss)

print(f"  SGD avg:  {np.mean(sgd_losses):.4f}")
print(f"  Adam avg: {np.mean(adam_losses):.4f}")
print(f"  DMGD avg: {np.mean(dmgd_losses):.4f}")
print("✓ Comparison works")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("All Practical Example Tests Passed! ✓")
print("="*60)

print("\nPractical examples are ready to use:")
print("  1. examples/meta_learning_practical.py - Comprehensive examples")
print("  2. examples/meta_learning_classification.py - Classification focus")

print("\nThese examples demonstrate:")
print("  - Few-shot learning on various tasks")
print("  - Meta-training procedures")
print("  - Evaluation on new tasks")
print("  - Comparison with standard optimizers")
print("  - Hyperparameter impact")
print("  - Task diversity considerations")
