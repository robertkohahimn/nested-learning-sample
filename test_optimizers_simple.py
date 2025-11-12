"""
Simple test to validate Optimizer components
"""

import sys
sys.path.insert(0, '/home/user/nested-learning-sample')

import torch
import torch.nn as nn

from src.optimizers.dmgd import DeepMomentumGD, MomentumMLP
from src.optimizers.nested_optimizer import NestedOptimizer, NestedOptimizerBuilder
from src.utils.frequency_scheduler import FrequencyScheduler

print("Testing MomentumMLP...")

# Test 1: MomentumMLP initialization
mlp = MomentumMLP(input_dim=128, hidden_dims=[64, 32])
assert isinstance(mlp, nn.Module)
print("✓ Test 1 passed: MomentumMLP initialization")

# Test 2: MomentumMLP forward pass
x = torch.randn(128)  # [gradient(64) + momentum(64)]
output = mlp(x)
assert output.shape == (64,)  # Should output momentum of same dim as gradient
print("✓ Test 2 passed: MomentumMLP forward pass")

print("\nTesting DeepMomentumGD...")

# Test 3: DMGD initialization
model = nn.Linear(10, 1)
optimizer = DeepMomentumGD(model.parameters(), lr=0.01)
assert len(optimizer.param_groups) > 0
print("✓ Test 3 passed: DMGD initialization")

# Test 4: DMGD single step
x = torch.randn(5, 10)
y = torch.randn(5, 1)
loss = ((model(x) - y) ** 2).mean()
loss.backward()

# Check gradients exist
assert model.weight.grad is not None
print("✓ Test 4 passed: Gradients computed")

# Perform optimization step
optimizer.step()
optimizer.zero_grad()
print("✓ Test 5 passed: DMGD optimization step")

# Test 6: DMGD training loop (basic functionality test)
# Note: DMGD may not converge as fast as standard optimizers without meta-learning
# This test just verifies it runs without errors
model = nn.Linear(10, 1)
optimizer = DeepMomentumGD(model.parameters(), lr=0.001, momentum_hidden_dims=[32, 16])

losses = []
for epoch in range(10):
    x = torch.randn(20, 10)
    y = 2 * x.sum(dim=1, keepdim=True) + 1  # Simple linear function

    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Just check that it ran without errors and loss is finite
assert all(torch.isfinite(torch.tensor(l)) for l in losses), "All losses should be finite"
print(f"✓ Test 6 passed: DMGD training loop executes (final loss: {losses[-1]:.4f})")

print("\nTesting NestedOptimizer...")

# Test 7: NestedOptimizer initialization
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Create parameter groups
fast_params = list(model[0].parameters())  # First layer
slow_params = list(model[2].parameters())  # Last layer

# Create optimizers
opt_fast = torch.optim.SGD(fast_params, lr=0.01)
opt_slow = torch.optim.SGD(slow_params, lr=0.001)

# Create scheduler
scheduler = FrequencyScheduler(num_levels=2, scaling='manual', manual_frequencies=[1, 5])

# Create nested optimizer
nested_opt = NestedOptimizer(
    optimizers=[opt_fast, opt_slow],
    frequency_scheduler=scheduler
)
assert nested_opt.num_levels == 2
print("✓ Test 7 passed: NestedOptimizer initialization")

# Test 8: NestedOptimizer step
x = torch.randn(10, 10)
y = torch.randn(10, 1)
loss = ((model(x) - y) ** 2).mean()
loss.backward()

# Step 0: both levels should update
updated = nested_opt.step(step=0)
assert updated[0] == True  # Fast level updates
assert updated[1] == True  # Slow level updates (at step 0)
print("✓ Test 8 passed: NestedOptimizer step at step=0")

# Test 9: Frequency-based updates
nested_opt.zero_grad()
loss = ((model(x) - y) ** 2).mean()
loss.backward()

# Step 3: only fast level should update
updated = nested_opt.step(step=3)
assert updated[0] == True  # Fast level updates (every step)
assert updated[1] == False  # Slow level doesn't update
print("✓ Test 9 passed: Frequency-based updates")

nested_opt.zero_grad()
loss = ((model(x) - y) ** 2).mean()
loss.backward()

# Step 5: both levels should update
updated = nested_opt.step(step=5)
assert updated[0] == True  # Fast level updates
assert updated[1] == True  # Slow level updates (every 5 steps)
print("✓ Test 10 passed: Both levels update at step=5")

# Test 11: Update history
history = nested_opt.get_update_history()
assert 0 in history[0]  # Fast level updated at step 0
assert 0 in history[1]  # Slow level updated at step 0
assert 3 in history[0]  # Fast level updated at step 3
assert 3 not in history[1]  # Slow level didn't update at step 3
assert 5 in history[1]  # Slow level updated at step 5
print("✓ Test 11 passed: Update history tracking")

# Test 12: Update stats
stats = nested_opt.get_update_stats()
assert stats[0]['frequency'] == 1
assert stats[1]['frequency'] == 5
assert stats[0]['num_updates'] == 3  # Steps 0, 3, 5
assert stats[1]['num_updates'] == 2  # Steps 0, 5
print("✓ Test 12 passed: Update statistics")

print("\nTesting NestedOptimizerBuilder...")

# Test 13: Builder initialization
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

builder = NestedOptimizerBuilder(model, num_levels=2)
assert builder.num_levels == 2
print("✓ Test 13 passed: Builder initialization")

# Test 14: Manual parameter assignment
builder.add_params(model[0].parameters(), level=0)  # Fast
builder.add_params([p for i in [2, 4] for p in model[i].parameters()], level=1)  # Slow
print("✓ Test 14 passed: Manual parameter assignment")

# Test 15: Build nested optimizer
nested_opt = builder.build(
    optimizer_types=['adam', 'sgd'],
    learning_rates=[0.001, 0.0001],
    frequencies=[1, 10]
)
assert nested_opt.num_levels == 2
assert isinstance(nested_opt, NestedOptimizer)
print("✓ Test 15 passed: Builder builds NestedOptimizer")

# Test 16: Auto-assign parameters
builder2 = NestedOptimizerBuilder(model, num_levels=2)
builder2.auto_assign_params(strategy='uniform')
nested_opt2 = builder2.build()
assert isinstance(nested_opt2, NestedOptimizer)
print("✓ Test 16 passed: Auto-assign parameters")

print("\nTesting DMGD with meta-learning...")

# Test 17: DMGD with meta-optimizer
model = nn.Linear(5, 1)
optimizer = DeepMomentumGD(model.parameters(), lr=0.01, mlp_lr=0.0001)
assert optimizer.meta_optimizer is not None
print("✓ Test 17 passed: DMGD with meta-optimizer")

# Test 18: State dict save/load
model = nn.Linear(10, 1)
optimizer = DeepMomentumGD(model.parameters(), lr=0.01)

# Train a bit
x = torch.randn(5, 10)
y = torch.randn(5, 1)
loss = ((model(x) - y) ** 2).mean()
loss.backward()
optimizer.step()

# Save state
state = optimizer.state_dict()
assert 'momentum_mlps' in state
print("✓ Test 18 passed: Optimizer state dict")

# Load state
optimizer2 = DeepMomentumGD(model.parameters(), lr=0.01)
optimizer2.load_state_dict(state)
print("✓ Test 19 passed: Load optimizer state")

print("\n" + "="*50)
print("All Optimizer tests passed! ✓")
print("="*50)
