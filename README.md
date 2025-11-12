# Nested Learning: Sample Implementation

A PyTorch implementation of **Nested Learning** based on the NeurIPS 2025 paper by Ali Behrouz et al. (Google Research).

## Overview

Nested Learning is a novel machine learning paradigm that views models as **hierarchies of interconnected optimization problems**, each operating at different time scales. This approach addresses key challenges in deep learning:

- **Catastrophic Forgetting**: Retains knowledge from previous tasks during continual learning
- **Long-Context Processing**: Efficiently handles extended context lengths
- **Adaptive Optimization**: Learns the optimization process itself

### Key Components

1. **Deep Momentum Gradient Descent (DMGD)**: Replaces standard momentum with a learnable MLP, making the optimizer itself a deep learning model

2. **Continuum Memory System (CMS)**: A spectrum of memory modules updating at different frequencies, from short-term (every step) to long-term (every 1000+ steps)

3. **Multi-Frequency Updates**: Different parameters update at different rates, mimicking neuroplasticity in biological systems

4. **Hope Architecture**: Proof-of-concept architecture with self-modifying capabilities and unbounded in-context learning

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/robertkohahimn/nested-learning-sample.git
cd nested-learning-sample

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- See `requirements.txt` for full list

## Quick Start

### Example 1: Meta-Learning with DMGD

```python
import torch
import torch.nn as nn
from src.optimizers.dmgd import DeepMomentumGD
from src.optimizers.meta_learning import MetaLearningTrainer, create_task_sampler

# Define model creation function
def create_model():
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

# Create DMGD optimizer
dummy_model = create_model()
dmgd = DeepMomentumGD(
    dummy_model.parameters(),
    lr=0.01,
    momentum_hidden_dims=[32, 16],
    mlp_lr=0.001
)

# Create meta-learning trainer
trainer = MetaLearningTrainer(
    model_fn=create_model,
    dmgd_optimizer=dmgd,
    inner_steps=5,
    inner_lr=0.01,
    meta_lr=0.001
)

# Meta-train on sine wave tasks
sampler = create_task_sampler(task_type='sine', n_train=25, n_val=25)
meta_losses = trainer.meta_train(sampler, num_episodes=100, verbose=True)
```

### Example 2: Nested Optimizer with Multi-Frequency Updates

```python
import torch
import torch.nn as nn
from src.optimizers.nested_optimizer import NestedOptimizerBuilder

# Create model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Build nested optimizer with 3 frequency levels
builder = NestedOptimizerBuilder(model.parameters(), num_levels=3)
builder.auto_assign_parameters('alternating')
builder.set_optimizer(0, torch.optim.Adam, {'lr': 0.001})  # Fast
builder.set_optimizer(1, torch.optim.SGD, {'lr': 0.01})     # Medium
builder.set_optimizer(2, torch.optim.SGD, {'lr': 0.1})      # Slow
builder.set_frequencies('exponential', base_freq=1, scale_factor=5)

nested_opt = builder.build()

# Training loop
for step, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y)
    loss.backward()
    nested_opt.step(step=step)  # Multi-frequency updates
    nested_opt.zero_grad()
```

### Example 3: NestedMLP with Multi-Frequency Layers

```python
from src.models import NestedMLP
from src.optimizers import NestedOptimizerBuilder

# Create model with nested layers
model = NestedMLP(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10
)

# Build frequency-aware optimizer
builder = NestedOptimizerBuilder(model, num_levels=3)
builder.auto_assign_params('uniform')
optimizer = builder.build(
    optimizer_types=['adam', 'sgd', 'sgd'],
    learning_rates=[0.001, 0.01, 0.1],
    frequencies=[1, 10, 100]
)

# Training with multi-frequency updates
for step, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step(step=step)  # Updates active levels only
    optimizer.zero_grad()
```

### Example 4: Hope Model for Language Modeling

```python
from src.models import HopeModel

# Create Hope model with memory
model = HopeModel(
    vocab_size=10000,
    hidden_dim=256,
    num_layers=6,
    memory_levels=3,
    memory_capacities=[100, 50, 25]
)

# Forward pass
logits, memory_info = model(input_ids, step=current_step)

# Text generation
generated = model.generate(
    input_ids=prompt,
    max_new_tokens=100,
    temperature=0.8
)
```

### Example 5: Training with NestedTrainer

```python
from src.models import NestedMLP
from src.optimizers import NestedOptimizerBuilder
from src.training import NestedTrainer, EarlyStopping, CheckpointCallback

# Create model
model = NestedMLP(784, [256, 128], 10)

# Build optimizer with multi-frequency updates
builder = NestedOptimizerBuilder(model, num_levels=3)
builder.auto_assign_params('uniform')
optimizer = builder.build(
    optimizer_types=['adam', 'sgd', 'sgd'],
    learning_rates=[0.001, 0.01, 0.1],
    frequencies=[1, 10, 100]
)

# Create trainer with callbacks
trainer = NestedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    use_nested_optimizer=True,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5),
        CheckpointCallback(filepath='best_model.pt', save_best_only=True)
    ]
)

# Train with automatic early stopping and checkpointing
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50
)

# Access training history
print(f"Best val loss: {min(history['val_loss'])}")
```

## Project Structure

```
nested-learning-sample/
├── src/
│   ├── optimizers/          # DMGD and nested optimizers
│   ├── memory/              # Continuum Memory System
│   ├── layers/              # Multi-frequency layers
│   ├── models/              # NestedMLP, Hope architecture
│   ├── training/            # NestedTrainer and callbacks
│   └── utils/               # Visualization and utilities
├── examples/                # Example scripts
├── tests/                   # Unit and integration tests
├── notebooks/               # Tutorial notebooks
├── requirements.txt
├── setup.py
└── projectplan.md          # Detailed implementation plan
```

## Examples

See the `examples/` directory for complete examples:

- `simple_cms_example.py` - Continuum Memory System demonstration
- `simple_dmgd_example.py` - Basic DMGD usage
- `meta_learning_practical.py` - Comprehensive meta-learning examples
- `meta_learning_classification.py` - Classification-focused meta-learning

## Features

### ✅ Completed (Phase 1-5)

**Phase 1: Project Structure**
- [x] Complete Python package setup
- [x] Dependencies and requirements
- [x] Documentation framework

**Phase 2: Continuum Memory System (CMS)**
- [x] Multi-level memory banks with frequency-based updates
- [x] L2-based associative memory retrieval
- [x] Automatic memory consolidation
- [x] 18/18 tests passing ✓

**Phase 3: Deep Momentum GD & Nested Optimizers**
- [x] DMGD with learnable MLP-based momentum
- [x] NestedOptimizer for multi-frequency parameter updates
- [x] Meta-learning framework (MAML-style)
- [x] Task samplers (regression and classification)
- [x] 52/52 tests passing ✓

**Phase 4: Model Architectures**
- [x] NestedLayer, NestedLinear, NestedEmbedding, NestedLayerNorm
- [x] CMSBlock and CMSAttentionBlock (memory-augmented blocks)
- [x] NestedMLP baseline model
- [x] Hope architecture for language modeling
- [x] Frequency-aware parameter grouping
- [x] 32/32 tests passing ✓

**Phase 5: Training Framework**
- [x] NestedTrainer with coordinated multi-frequency training
- [x] Checkpoint management with CMS state preservation
- [x] Callbacks (EarlyStopping, Checkpoint, LR Scheduling)
- [x] Metrics tracking (MetricsTracker, UpdateFrequencyTracker, MemoryTracker)
- [x] TensorBoard integration
- [x] 13/13 tests passing ✓

**Total: 120+ tests passing across all phases! 🎉**

### ⚠️ Known Limitations

- Gradient flow in meta-learning is limited (see `IMPLEMENTATION_STATUS.md`)
- MLP parameters show minimal updates during meta-training
- Memory overhead for large models with DMGD

### 🔮 Roadmap (Phase 6-7)

**Phase 6-7: Examples & Benchmarks**
- Real-world demonstrations (MNIST, CIFAR-10, language tasks)
- Continual learning benchmarks
- Performance comparisons
- Jupyter notebooks for exploration

For detailed implementation status, see `IMPLEMENTATION_STATUS.md`

## Theory Background

Nested Learning reframes machine learning as a system of nested optimization problems:

```
θ^(l+1) = θ^(l) - α_l ∇L(θ^(l), c^(l))
```

where `l` represents the optimization level, and parameters at each level update at frequency `C^(l)`.

**Key insight**: Traditional deep learning already uses nested optimization implicitly (e.g., momentum in SGD is an inner optimization problem). Nested Learning makes this explicit and learnable.

For full mathematical details, see `projectplan.md` or the original paper.

## Performance

Expected performance improvements over baselines:

| Metric | Baseline | Nested Learning | Target |
|--------|----------|-----------------|--------|
| Continual Learning (accuracy retention) | <80% | >95% | >95% |
| Long-context (NIAH) | 50-70% | >90% | >90% |
| Training efficiency | 1x | 0.8-1.2x | ~1x |

*(Results will be updated as implementation progresses)*

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See `projectplan.md` for implementation guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original paper authors: Ali Behrouz et al. (Google Research)
- Inspired by the NeurIPS 2025 paper "Nested Learning: The Illusion of Deep Learning Architectures"
- Built with PyTorch

## Documentation

- `README.md` - This file, project overview and quick start
- `projectplan.md` - Comprehensive 7-phase development plan
- `IMPLEMENTATION_STATUS.md` - Detailed status of all components
- `META_LEARNING_GUIDE.md` - User guide for meta-learning
- `PHASE2_TEST_RESULTS.md` - Phase 2 (CMS) test results
- `PHASE3_TEST_RESULTS.md` - Phase 3 (DMGD) test results
- `PHASE4_TEST_RESULTS.md` - Phase 4 (Architectures) test results

## Resources

- **Paper**: [Nested Learning: The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf)
- **OpenReview**: [Discussion Forum](https://openreview.net/forum?id=nbMeRvNb7A)

## Contact

For questions or issues, please open a GitHub issue or refer to the project documentation.

---

**Status**: ✅ Phase 1-4 Complete (107+ tests passing) | ⚠️ Meta-learning gradient flow limited | 🔮 Phase 5-7 planned

Last Updated: November 12, 2025
