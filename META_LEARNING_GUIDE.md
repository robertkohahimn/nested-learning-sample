# Meta-Learning for DMGD - User Guide

## Overview

This guide explains how to use meta-learning with Deep Momentum Gradient Descent (DMGD) to create learned optimizers that can quickly adapt to new tasks.

---

## What is Meta-Learning for Optimizers?

**Standard Optimizers**: Use fixed rules (e.g., SGD momentum: `m = β*m + g`)

**Learned Optimizers (DMGD)**: Use a neural network to compute momentum: `m = MLP(g, m_prev)`

**Meta-Learning**: Train the MLP across multiple tasks so it learns effective momentum strategies that generalize

---

## Quick Start

###Step 1: Define Your Model

```python
def create_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
```

### Step 2: Create DMGD with Meta-Learning

```python
from src.optimizers import DeepMomentumGD, MetaLearningTrainer, create_task_sampler

# Create DMGD optimizer
dummy_model = create_model()
dmgd = DeepMomentumGD(
    dummy_model.parameters(),
    lr=0.01,
    momentum_hidden_dims=[32, 16],
    mlp_lr=0.001  # Meta-learning rate
)
```

### Step 3: Create Meta-Trainer

```python
trainer = MetaLearningTrainer(
    model_fn=create_model,
    dmgd_optimizer=dmgd,
    inner_steps=5,       # Steps per task
    inner_lr=0.01,       # Task learning rate
    meta_lr=0.001        # Meta-optimizer learning rate
)
```

### Step 4: Create Task Sampler

```python
# Sample random linear regression tasks
task_sampler = create_task_sampler(
    task_type='linear_regression',
    input_dim=10,
    output_dim=1,
    n_train=50,
    n_val=20
)
```

### Step 5: Meta-Train

```python
# Meta-train for 100 episodes
meta_losses = trainer.meta_train(
    task_sampler=task_sampler,
    num_episodes=100,
    verbose=True,
    log_interval=10
)
```

### Step 6: Evaluate on New Tasks

```python
# Test on a new task
new_task = task_sampler()
results = trainer.evaluate(new_task)

print(f"Initial loss: {results['initial_loss']:.4f}")
print(f"Final loss: {results['final_loss']:.4f}")
print(f"Improvement: {results['improvement']:.4f}")
```

---

## Task Types

The library provides three built-in task samplers:

### Linear Regression
```python
sampler = create_task_sampler(task_type='linear_regression')
# Generates: y = Wx + b + noise
```

### Sine Waves
```python
sampler = create_task_sampler(task_type='sine')
# Generates: y = A * sin(ω*x + φ) + noise
```

### Polynomials
```python
sampler = create_task_sampler(task_type='polynomial')
# Generates: y = c2*x² + c1*x + c0 + noise
```

---

## Custom Task Samplers

Create your own task distribution:

```python
def my_task_sampler():
    """Sample a custom task."""
    # Generate task-specific data
    X_train = torch.randn(50, 10)
    y_train = custom_function(X_train)

    X_val = torch.randn(20, 10)
    y_val = custom_function(X_val)

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val)
    }

# Use it with the trainer
meta_losses = trainer.meta_train(
    task_sampler=my_task_sampler,
    num_episodes=100
)
```

---

## How Meta-Learning Works

### Two-Loop Structure

**Inner Loop** (Task Learning):
1. Sample a new task
2. Create fresh model
3. Train model on task using DMGD
4. Measure validation loss

**Outer Loop** (Meta-Learning):
1. Compute gradients of validation loss w.r.t. MLP parameters
2. Update MLP to minimize validation loss
3. MLP learns better momentum computation

### Mathematical Formulation

**Inner Loop**:
```
For k steps:
    m_t = MLP_φ(g_t, m_{t-1})
    θ_t = θ_{t-1} - α * m_t
```

**Outer Loop**:
```
L_meta = Loss(θ_K, validation_data)
φ = φ - β * ∇_φ L_meta
```

Where:
- `θ`: Model parameters
- `φ`: MLP parameters (being meta-learned)
- `m`: Momentum
- `g`: Gradient
- `α`: Inner learning rate
- `β`: Meta learning rate

---

## Hyperparameter Tuning

### Critical Hyperparameters

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `inner_steps` | 5 | Steps per task | 3-10 for simple tasks, 10-50 for complex |
| `inner_lr` | 0.01 | Task learning rate | Match your typical LR |
| `meta_lr` | 0.001 | MLP learning rate | 10x smaller than inner_lr |
| `momentum_hidden_dims` | [64, 32] | MLP architecture | Smaller for simple tasks [16, 8] |
| `num_episodes` | 100 | Meta-training episodes | 100-1000 depending on task complexity |

### Tuning Strategy

1. **Start Simple**: Use small MLPs `[16]` and few episodes (50)
2. **Check Meta-Loss**: Should generally decrease
3. **Increase Capacity**: If underfitting, larger MLP
4. **More Episodes**: If still improving, train longer
5. **Adjust Rates**: If unstable, reduce meta_lr

---

## Evaluation and Diagnostics

### Check Meta-Training Progress

```python
stats = trainer.get_statistics()

print(f"Episodes: {stats['num_episodes']}")
print(f"Mean meta-loss: {stats['mean_meta_loss']:.4f}")
print(f"Improvement: {stats['improvement']:.4f}")
```

### Compare with Baselines

```python
def evaluate_optimizer(optimizer_fn, task_data, num_steps=10):
    model = create_model()
    opt = optimizer_fn(model.parameters())

    # Train
    for step in range(num_steps):
        loss = criterion(model(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    return criterion(model(X_val), y_val).item()

# Compare
sgd_loss = evaluate_optimizer(lambda p: torch.optim.SGD(p, lr=0.01), task)
adam_loss = evaluate_optimizer(lambda p: torch.optim.Adam(p, lr=0.01), task)
dmgd_loss = trainer.evaluate(task)['final_loss']

print(f"SGD:  {sgd_loss:.4f}")
print(f"Adam: {adam_loss:.4f}")
print(f"DMGD: {dmgd_loss:.4f}")
```

---

## Saving and Loading

### Save Meta-Learned Optimizer

```python
# After meta-training
trainer.save_checkpoint('dmgd_meta_learned.pt')
```

### Load and Use

```python
# Create new trainer
trainer = MetaLearningTrainer(
    model_fn=create_model,
    dmgd_optimizer=dmgd,
    inner_steps=5
)

# Load meta-learned weights
trainer.load_checkpoint('dmgd_meta_learned.pt')

# Use on new tasks
results = trainer.evaluate(new_task)
```

---

## Known Limitations

### 1. Gradient Flow Challenge

**Issue**: MLP parameters may not receive gradients during meta-training

**Why**: Optimizer step happens outside the computational graph

**Impact**: Meta-learning may not improve performance significantly

**Current Status**: Functional but requires further optimization

**Workaround**:
- Use smaller learning rates
- Longer meta-training
- Manual gradient management (advanced)

### 2. Memory Overhead

**Issue**: Separate MLP for each parameter group

**Impact**: High memory usage for large models

**Mitigation**:
- Use smaller MLPs: `momentum_hidden_dims=[16]`
- Apply selectively to important parameters
- Share MLPs across similar parameter groups

### 3. Computational Cost

**Issue**: Meta-training is expensive (double backward pass)

**Impact**: Slow training, especially for many episodes

**Mitigation**:
- Start with fewer episodes
- Use parallel task evaluation (future work)
- Save and reuse meta-learned optimizers

### 4. Hyperparameter Sensitivity

**Issue**: Performance highly dependent on hyperparameters

**Impact**: Requires careful tuning

**Mitigation**:
- Start with provided defaults
- Use grid search or Bayesian optimization
- Track meta-loss curves

---

## Best Practices

### ✅ DO:

1. **Start with simple tasks** to validate setup
2. **Monitor meta-loss** - should generally decrease
3. **Compare with baselines** (SGD, Adam) on test tasks
4. **Use smaller MLPs** for faster iteration
5. **Save checkpoints** regularly
6. **Test on diverse tasks** to check generalization

### ❌ DON'T:

1. **Don't expect immediate improvement** - meta-learning takes time
2. **Don't use huge MLPs** - start small
3. **Don't skip validation** - always test on new tasks
4. **Don't overtrain** - watch for overfitting to task distribution
5. **Don't use for single-task** - just use Adam instead

---

## Troubleshooting

### Meta-Loss Not Decreasing

**Possible Causes**:
- Learning rates too high/low
- Inner steps too few/many
- MLP too small/large
- Task distribution too diverse

**Solutions**:
- Reduce `meta_lr` by 10x
- Try different `inner_steps` (3, 5, 10)
- Adjust MLP size
- Simplify tasks initially

### DMGD Worse Than Adam

**Possible Causes**:
- Insufficient meta-training
- Poor hyperparameters
- Task mismatch
- Gradient flow issues

**Solutions**:
- Train for more episodes (500+)
- Tune hyperparameters systematically
- Ensure test tasks match training distribution
- Check if meta-loss is decreasing

### High Memory Usage

**Solutions**:
- Use `momentum_hidden_dims=[16]` or smaller
- Reduce batch size
- Use gradient checkpointing
- Apply DMGD selectively

---

## Advanced Usage

### Multi-Task Meta-Learning

Mix different task types:

```python
def mixed_task_sampler():
    task_type = np.random.choice(['linear_regression', 'sine', 'polynomial'])
    sampler = create_task_sampler(task_type=task_type)
    return sampler()

trainer.meta_train(mixed_task_sampler, num_episodes=200)
```

### Custom Loss Functions

```python
def custom_criterion(pred, target):
    mse = ((pred - target) ** 2).mean()
    reg = 0.01 * pred.abs().mean()
    return mse + reg

trainer.meta_train(task_sampler, criterion=custom_criterion)
```

### Continual Meta-Learning

Continue training from checkpoint:

```python
# Initial meta-training
trainer.meta_train(sampler1, num_episodes=100)
trainer.save_checkpoint('dmgd_v1.pt')

# Continue on new task distribution
trainer.load_checkpoint('dmgd_v1.pt')
trainer.meta_train(sampler2, num_episodes=100)
trainer.save_checkpoint('dmgd_v2.pt')
```

---

## Research Directions

Potential improvements and extensions:

1. **Higher-Order Gradients**: Use `torch.autograd.grad` for proper gradient flow
2. **Shared MLPs**: One MLP for all parameters (with parameter features)
3. **Task Embeddings**: Condition MLP on task representation
4. **Evolution Strategies**: Meta-learn without gradients
5. **Hyperparameter Optimization**: Auto-tune learning rates
6. **Multi-Step Lookahead**: Optimize for multiple future steps

---

## FAQ

**Q: Why is my meta-learned DMGD not better than Adam?**

A: Meta-learning requires careful tuning. Ensure:
- Sufficient meta-training episodes (100+)
- Appropriate task distribution
- Proper hyperparameters
- Meta-loss is decreasing

**Q: How many episodes do I need?**

A: Depends on task complexity:
- Simple (linear regression): 50-100
- Medium (sine waves): 100-500
- Complex (real datasets): 500-2000

**Q: Can I use this for classification?**

A: Yes! Just use classification tasks in your sampler:
```python
def classification_task_sampler():
    # Sample binary classification task
    X_train = torch.randn(100, 10)
    y_train = (X_train.sum(dim=1) > 0).long()
    # ... similar for validation
    return {'train': (X_train, y_train), 'val': (X_val, y_val)}
```

**Q: How do I know if meta-learning is working?**

A: Check these indicators:
1. Meta-loss decreases over episodes
2. Performance improves on held-out tasks
3. DMGD adapts faster than random initialization
4. Beats or matches Adam on new tasks

**Q: What if I get out-of-memory errors?**

A: Try:
1. Smaller MLP: `[16]` instead of `[64, 32]`
2. Fewer inner steps: 3 instead of 10
3. Smaller batch sizes in tasks
4. Gradient checkpointing (advanced)

---

## Citation

If you use this meta-learning implementation, please cite:

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

## Examples

See `examples/meta_learning_dmgd.py` for a complete working example.

---

## Support

For issues or questions:
1. Check this guide
2. Review example code
3. Open GitHub issue
4. Consult the paper

---

**Last Updated**: November 2025
**Status**: Functional (gradient flow optimization pending)
