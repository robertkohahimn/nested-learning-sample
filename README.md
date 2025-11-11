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

### Example 1: Simple Regression with DMGD

```python
import torch
from src.optimizers.dmgd import DeepMomentumGD
from src.models.nested_mlp import NestedMLP

# Create a simple model
model = NestedMLP(input_dim=1, hidden_dim=64, output_dim=1)

# Use Deep Momentum GD optimizer
optimizer = DeepMomentumGD(model.parameters(), lr=0.01)

# Training loop
for x, y in dataloader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Example 2: Continual Learning

```python
from src.models.nested_mlp import NestedMLP
from src.memory.cms import ContinuumMemorySystem

# Create model with CMS
model = NestedMLP(
    input_dim=784,
    hidden_dim=256,
    output_dim=10,
    use_cms=True,
    frequency_levels=[1, 10, 100]  # Short, medium, long-term
)

# Train on sequential tasks
for task in tasks:
    train_on_task(model, task)
    # CMS automatically consolidates important memories
```

## Project Structure

```
nested-learning-sample/
├── src/
│   ├── optimizers/          # DMGD and nested optimizers
│   ├── memory/              # Continuum Memory System
│   ├── layers/              # Multi-frequency layers
│   ├── models/              # NestedMLP, Hope architecture
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

- `simple_regression.py` - Basic DMGD demonstration
- `continual_learning.py` - Catastrophic forgetting mitigation
- `long_context_task.py` - Needle-in-a-haystack benchmark

## Features

### Current Implementation

- [x] Project structure and setup
- [ ] Deep Momentum Gradient Descent (DMGD)
- [ ] Continuum Memory System (CMS)
- [ ] Multi-frequency neural layers
- [ ] NestedMLP model
- [ ] Simplified Hope architecture
- [ ] Training framework
- [ ] Example scripts and benchmarks

### Roadmap

- Meta-learning loop for DMGD
- Advanced CMS features (adaptive frequencies, memory consolidation)
- Full Hope architecture with self-modification
- Distributed training support
- Pre-trained models
- Integration with Hugging Face

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

## Resources

- **Paper**: [Nested Learning: The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf)
- **OpenReview**: [Discussion Forum](https://openreview.net/forum?id=nbMeRvNb7A)
- **Google Research Blog**: [Introducing Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)

## Contact

For questions or issues, please open a GitHub issue or refer to the project documentation.

---

**Status**: 🚧 Under Active Development - Phase 1 Complete

Last Updated: November 2025
