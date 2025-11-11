# Nested Learning: Implementation Plan

**Based on the NeurIPS 2025 paper by Ali Behrouz (Google Research)**

---

## 📋 Overview of Nested Learning

**Key Concepts:**

1. **Nested Optimization**: Views ML models as hierarchies of interconnected optimization problems, each with its own update frequency
2. **Continuum Memory System (CMS)**: Memory spectrum with components updating at different frequencies (not just short/long-term)
3. **Deep Momentum Gradient Descent (DMGD)**: Replaces linear momentum with learnable MLP, making the optimizer itself a deep learning model
4. **Multi-Time-Scale Updates**: Different layers/components update at different rates to prevent catastrophic forgetting
5. **Hope Architecture**: Proof-of-concept that extends Titans architecture with self-modifying capabilities

### Core Insights

- **Context Compression**: All deep learning fundamentally "learns by compressing context" - whether data samples or gradient histories
- **Catastrophic Forgetting Mitigation**: Multi-frequency updates allow models to retain prior knowledge while learning new tasks
- **Unified Framework**: Connects optimizers, memory modules, and network layers under one theoretical framework

---

## 🏗️ Development Plan

### Phase 1: Foundation & Core Components (Week 1-2)

#### 1.1 Project Structure Setup

```
nested-learning-sample/
├── src/
│   ├── __init__.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── base_optimizer.py
│   │   ├── dmgd.py              # Deep Momentum GD
│   │   └── nested_optimizer.py   # Nested optimization wrapper
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── cms.py                # Continuum Memory System
│   │   └── associative_memory.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── nested_layer.py       # Multi-frequency layer
│   │   └── cms_block.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── nested_mlp.py
│   │   └── hope.py               # Hope architecture
│   └── utils/
│       ├── __init__.py
│       ├── frequency_scheduler.py
│       └── visualization.py
├── examples/
│   ├── simple_regression.py
│   ├── continual_learning.py
│   └── long_context_task.py
├── tests/
│   ├── __init__.py
│   ├── test_optimizers.py
│   ├── test_memory.py
│   └── test_models.py
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_dmgd_tutorial.ipynb
│   └── 03_continual_learning_demo.ipynb
├── requirements.txt
├── README.md
├── setup.py
└── projectplan.md (this file)
```

#### 1.2 Dependencies

**Core:**
- `torch>=2.0.0` - PyTorch for deep learning
- `numpy>=1.24.0` - Numerical operations
- `einops>=0.7.0` - Tensor operations

**Visualization & Monitoring:**
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `tensorboard>=2.13.0` - Training monitoring
- `wandb>=0.15.0` (optional) - Experiment tracking

**Development:**
- `pytest>=7.4.0` - Testing
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `flake8>=6.0.0` - Linting
- `mypy>=1.4.0` - Type checking

#### 1.3 Core Mathematical Formulation

**Nested Optimization Loop:**
```
θ^(l+1) = θ^(l) - α_l ∇L(θ^(l), c^(l))

where:
- l: optimization level
- θ^(l): parameters at level l
- c^(l): context flow at level l
- α_l: learning rate at level l
```

**Multi-Level Parameter Updates:**
```
Parameters at level l update every C^(l) steps:
- Level 0 (fast): C^(0) = 1 (every step)
- Level 1 (medium): C^(1) = 10-100
- Level 2 (slow): C^(2) = 1000+
```

**Deliverables:**
- [ ] Complete project structure created
- [ ] requirements.txt with all dependencies
- [ ] Basic setup.py for package installation
- [ ] README.md with project overview

---

### Phase 2: Continuum Memory System (CMS) (Week 2-3)

#### 2.1 CMS Module Implementation

```python
class ContinuumMemorySystem(nn.Module):
    """
    Memory spectrum with components updating at different frequencies.

    Architecture:
    - Short-term memory: updates every step (C=1)
    - Mid-term memory: updates every C steps (C=10-100)
    - Long-term memory: updates every C steps (C=1000+)

    Each memory bank uses associative memory with L2 regression.
    """
```

**Key Features:**
- Multiple memory banks with configurable update frequencies
- Associative memory using L2 regression loss (not just dot-product similarity)
- Context compression and retrieval mechanisms
- Memory consolidation across time scales
- Efficient memory bank management

**Mathematical Formulation:**
```
Memory Update (at level l, when step % C^(l) == 0):
M^(l)_{t+1} = M^(l)_t + η_l (K_t - M^(l)_t Q_t^T) Q_t

where:
- M^(l): memory bank at level l
- K_t: keys (context to store)
- Q_t: queries (access patterns)
- η_l: consolidation rate
```

#### 2.2 Update Frequency Scheduler

```python
class FrequencyScheduler:
    """
    Manages update frequencies for different memory/parameter levels.

    Supports:
    - Exponential frequency scaling: C^(l) = C_0 * r^l
    - Logarithmic frequency scaling: C^(l) = C_0 * log(1 + l)
    - Adaptive frequency based on gradient magnitude
    """
```

**Deliverables:**
- [ ] ContinuumMemorySystem class
- [ ] AssociativeMemory module with L2 regression
- [ ] FrequencyScheduler for multi-level updates
- [ ] Unit tests for CMS
- [ ] Visualization of memory updates over time

---

### Phase 3: Deep Momentum Gradient Descent (DMGD) (Week 3-4)

#### 3.1 DMGD Optimizer

```python
class DeepMomentumGD(torch.optim.Optimizer):
    """
    Learnable optimizer where momentum is computed by an MLP.

    Standard momentum: m_{t+1} = β * m_t + (1-β) * ∇L_t
    DMGD momentum: m_{t+1} = MLP(∇L_t, m_t, θ_t)

    The MLP learns to compute optimal momentum based on:
    - Current gradient ∇L_t
    - Previous momentum m_t
    - Current parameters θ_t (optional, for second-order info)
    """
```

**Key Components:**

1. **Momentum MLP Architecture:**
   ```python
   MLP(
       Input: [gradient, prev_momentum, optional(parameters)]
       Hidden: 2-3 layers with ReLU/GELU
       Output: new_momentum
   )
   ```

2. **Meta-Learning Loop:**
   - Inner loop: Update model parameters with DMGD
   - Outer loop: Update MLP parameters to minimize validation loss

3. **Gradient Transformation:**
   - Learn non-linear transformations of gradients
   - Adapt to local loss landscape geometry

**Mathematical Formulation:**
```
Standard SGD with momentum:
m_{t+1} = β * m_t + ∇L_t
θ_{t+1} = θ_t - α * m_{t+1}

DMGD:
m_{t+1} = f_φ(∇L_t, m_t; φ)  # f_φ is MLP with params φ
θ_{t+1} = θ_t - α * m_{t+1}
φ_{t+1} = φ_t - γ * ∇_φ L_val(θ_{t+1})  # meta-update
```

#### 3.2 Nested Optimizer Wrapper

```python
class NestedOptimizer:
    """
    Coordinates multiple optimizers operating at different frequencies.

    Manages:
    - Hierarchical optimizer structure
    - Gradient flow between optimization levels
    - Coordinated parameter updates
    """
```

**Deliverables:**
- [ ] DeepMomentumGD optimizer class
- [ ] Momentum MLP implementation
- [ ] Meta-learning training loop
- [ ] NestedOptimizer wrapper
- [ ] Comparison benchmarks (SGD, Adam, DMGD)
- [ ] Unit tests for optimizers

---

### Phase 4: Nested Learning Architecture (Week 4-5)

#### 4.1 Multi-Frequency Neural Layers

```python
class NestedLayer(nn.Module):
    """
    Neural layer with parameters updating at different frequencies.

    Parameter groups:
    - Fast parameters (C=1): e.g., attention weights, layer norms
    - Medium parameters (C=10-100): e.g., feedforward weights
    - Slow parameters (C=1000+): e.g., embedding matrices, projection layers

    This mimics neuroplasticity in the brain where different connections
    adapt at different rates.
    """
```

**Key Features:**
- Separate parameter groups with frequency assignments
- Gradient accumulation for slow parameters
- Efficient computation (only compute necessary updates)
- Compatible with standard PyTorch modules

#### 4.2 CMS Blocks

```python
class CMSBlock(nn.Module):
    """
    Transformer/MLP block integrated with Continuum Memory System.

    Components:
    - Standard computation path (attention/feedforward)
    - CMS memory banks for context storage
    - Multi-scale context processing
    - Memory-augmented forward pass
    """
```

**Architecture:**
```
Input → [CMS Retrieval] → Attention/FFN → [CMS Update] → Output
         ↓                                     ↑
         └─── Memory Banks (multi-frequency) ──┘
```

#### 4.3 Hope Architecture (Simplified)

```python
class HopeModel(nn.Module):
    """
    Simplified Hope architecture inspired by the paper.

    Based on concepts from Titans architecture with:
    - CMS blocks for memory
    - Self-modifying capabilities
    - Recursive optimization loops
    - Unbounded in-context learning

    Note: Full Titans architecture is complex; this is a
    conceptual demonstration of Nested Learning principles.
    """
```

**Key Components:**
- Stacked CMS blocks
- Multi-frequency parameter updates
- Recurrent processing for extended context
- Optional self-attention with linear complexity

**Deliverables:**
- [ ] NestedLayer implementation
- [ ] CMSBlock module
- [ ] NestedMLP model (simple baseline)
- [ ] Simplified Hope architecture
- [ ] Model configuration utilities
- [ ] Architecture visualization tools

---

### Phase 5: Training Framework (Week 5-6)

#### 5.1 Nested Training Loop

```python
class NestedTrainer:
    """
    Training loop supporting Nested Learning paradigm.

    Features:
    - Multi-frequency parameter updates
    - Nested optimization coordination
    - Catastrophic forgetting mitigation
    - Long-context handling
    - Memory-efficient gradient computation
    """
```

**Training Algorithm:**
```
for step in training_steps:
    # Forward pass
    output = model(input)
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Multi-frequency updates
    for level in optimization_levels:
        if step % frequency[level] == 0:
            optimizer.step(level)
            optimizer.zero_grad(level)

    # CMS consolidation
    if step % consolidation_freq == 0:
        model.consolidate_memory()

    # Logging
    log_metrics(step, loss, memory_usage)
```

#### 5.2 Key Features

**Memory Consolidation:**
- Periodic transfer from short-term to long-term memory
- Gradient-based importance weighting
- Forgetting mechanism for irrelevant information

**Gradient Checkpointing:**
- Trade computation for memory
- Essential for long-context tasks
- Selective checkpointing based on layer importance

**Mixed Precision Training:**
- FP16 for fast parameters
- FP32 for slow parameters (stability)
- Automatic loss scaling

**Monitoring:**
- Loss tracking at multiple optimization levels
- Memory bank utilization
- Update frequency statistics
- Catastrophic forgetting metrics

**Deliverables:**
- [ ] NestedTrainer class
- [ ] Training configuration system
- [ ] Checkpoint saving/loading
- [ ] TensorBoard integration
- [ ] Training utilities (early stopping, LR scheduling)

---

### Phase 6: Demonstrations & Validation (Week 6-7)

#### 6.1 Example 1: Simple Regression with Nested Learning

**Goal:** Demonstrate DMGD on toy problem

```python
# examples/simple_regression.py
# Problem: Learn y = sin(x) + noise
# Compare: SGD, Adam, DMGD
# Metrics: Convergence speed, final loss
```

**Expected Results:**
- DMGD converges faster than SGD
- Learned momentum adapts to problem structure
- Visualization of momentum evolution

#### 6.2 Example 2: Continual Learning

**Goal:** Show mitigation of catastrophic forgetting

```python
# examples/continual_learning.py
# Tasks: MNIST → Fashion-MNIST → CIFAR-10
# Baseline: Standard fine-tuning
# Nested: Multi-frequency updates + CMS
# Metrics: Accuracy on all tasks over time
```

**Expected Results:**
- Baseline: >20% accuracy drop on previous tasks
- Nested Learning: <5% accuracy drop
- Memory consolidation preserves important features

**Benchmark:**
| Method | Task 1 Final | Task 2 Final | Task 3 Final | Avg |
|--------|--------------|--------------|--------------|-----|
| Fine-tune | 65% | 72% | 78% | 72% |
| EWC | 82% | 80% | 78% | 80% |
| Nested Learning | 89% | 87% | 85% | 87% |

#### 6.3 Example 3: Long-Context Task

**Goal:** Handle extended context lengths

```python
# examples/long_context_task.py
# Task: Needle-in-a-haystack (NIAH)
# Context: 4K, 8K, 16K tokens
# Metrics: Retrieval accuracy, memory usage
```

**Expected Results:**
- CMS enables processing of longer contexts
- Memory usage scales sub-linearly
- Consistent performance across context lengths

#### 6.4 Visualization Tools

```python
# src/utils/visualization.py
```

**Visualizations:**
1. **Update Frequency Heatmap:** Which parameters updated when
2. **Memory Consolidation:** Flow from short to long-term memory
3. **Gradient Flow:** Magnitude across optimization levels
4. **Catastrophic Forgetting:** Accuracy retention over tasks
5. **Loss Landscape:** How DMGD navigates vs standard optimizers

**Deliverables:**
- [ ] Three complete example scripts
- [ ] Comparison with baseline methods
- [ ] Comprehensive visualization suite
- [ ] Results documentation
- [ ] Jupyter notebooks for interactive exploration

---

### Phase 7: Documentation & Refinement (Week 7-8)

#### 7.1 Documentation

**README.md:**
- Project overview and motivation
- Installation instructions
- Quick start guide
- Links to examples and tutorials

**API Documentation:**
```python
# Use Google/NumPy docstring style
# Auto-generate with Sphinx/pdoc
```

**Tutorial Notebooks:**
1. `01_introduction.ipynb`: Nested Learning concepts
2. `02_dmgd_tutorial.ipynb`: Deep Momentum GD walkthrough
3. `03_continual_learning_demo.ipynb`: Catastrophic forgetting mitigation
4. `04_cms_visualization.ipynb`: Memory system analysis

**Theory Document:**
- Mathematical foundations
- Relationship to original paper
- Implementation choices and simplifications

#### 7.2 Testing

**Unit Tests:**
- Individual component testing (optimizers, layers, memory)
- Edge cases and error handling
- Numerical stability checks

**Integration Tests:**
- End-to-end training pipeline
- Multi-GPU compatibility (if applicable)
- Checkpoint save/load consistency

**Benchmark Suite:**
- Performance comparisons (speed, memory, accuracy)
- Ablation studies (which components matter most)
- Scalability tests (model size, context length)

**Test Coverage Goal:** >80%

#### 7.3 Code Quality

**Type Hints:**
```python
from typing import Dict, List, Optional, Tuple
import torch

def forward(
    self,
    x: torch.Tensor,
    memory: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ...
```

**Code Formatting:**
- Black (line length 88)
- isort (import organization)
- flake8 (linting)

**Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
  - repo: https://github.com/pycqa/isort
  - repo: https://github.com/pycqa/flake8
```

**CI/CD Pipeline:**
- GitHub Actions for automated testing
- Coverage reporting
- Documentation builds

**Deliverables:**
- [ ] Complete documentation
- [ ] Tutorial notebooks
- [ ] Comprehensive test suite (>80% coverage)
- [ ] Pre-commit hooks configured
- [ ] CI/CD pipeline setup
- [ ] Code review and refactoring

---

## 🎯 Key Implementation Challenges

### 1. Computational Complexity

**Challenge:** Multi-frequency updates and learnable optimizers increase computation

**Solutions:**
- Gradient checkpointing to save memory
- Mixed precision training (FP16/FP32)
- Efficient memory bank implementations (sparse storage)
- Profile and optimize hot paths
- Consider approximations for large-scale models

### 2. Hyperparameter Tuning

**Challenge:** Many new hyperparameters (frequencies C^(l), MLP sizes, consolidation rates)

**Solutions:**
- Provide sensible defaults based on heuristics
- Start with simple configurations (2-3 frequency levels)
- Grid search on toy problems
- Document hyperparameter sensitivity
- Consider automatic hyperparameter optimization (Optuna)

### 3. Memory Management

**Challenge:** CMS maintains multiple memory banks, increasing memory usage

**Solutions:**
- Fixed-size memory banks with eviction policies
- Sparse memory representations
- Optional disk caching for long-term memory
- Memory profiling and optimization
- Configurable memory budget

### 4. Validation Against Paper

**Challenge:** Hard to validate implementation without full Titans architecture and large-scale experiments

**Solutions:**
- Focus on core principles rather than exact replication
- Validate on toy problems where ground truth is known
- Qualitative comparisons with expected behaviors
- Ablation studies to verify component contributions
- Community feedback and iteration

### 5. Numerical Stability

**Challenge:** Nested optimization and learned optimizers can be unstable

**Solutions:**
- Careful initialization of MLP optimizers
- Gradient clipping at multiple levels
- Stability checks during training
- Fallback to standard methods if instability detected
- Comprehensive logging for debugging

---

## 📊 Success Metrics

### Quantitative Metrics

1. **Continual Learning Performance:**
   - Target: <5% accuracy drop on previous tasks
   - Baseline: >20% accuracy drop with standard fine-tuning

2. **Training Efficiency:**
   - Convergence speed: Competitive or better than Adam
   - Wall-clock time: <2x standard training
   - Memory usage: <2x baseline model

3. **Long-Context Handling:**
   - Context length: Handle 2-4x longer contexts than baseline
   - NIAH accuracy: >90% retrieval accuracy
   - Memory scaling: Sub-linear with context length

4. **Model Performance:**
   - Accuracy: Competitive with standard architectures
   - Generalization: Better on out-of-distribution data

### Qualitative Metrics

1. **Code Quality:**
   - Test coverage >80%
   - Clean, documented, maintainable code
   - Type hints throughout

2. **Usability:**
   - Easy to install and run examples
   - Clear documentation and tutorials
   - Intuitive API design

3. **Educational Value:**
   - Demonstrates Nested Learning concepts clearly
   - Helpful for understanding the paper
   - Inspires further research

---

## 🚀 Minimal Viable Implementation (MVP)

**Goal:** Get core concepts working in Week 1

### MVP Components (Priority Order)

1. **Basic DMGD Optimizer** (Day 1-2)
   - Simple MLP for momentum computation (2-layer)
   - Fixed learning rate (no meta-learning yet)
   - Works with standard PyTorch models

2. **Simple CMS with 2-3 Frequency Levels** (Day 3-4)
   - Short-term (C=1), medium-term (C=10), long-term (C=100)
   - Basic associative memory (key-value storage)
   - Fixed memory bank sizes

3. **NestedMLP Model** (Day 4-5)
   - Standard MLP with multi-frequency parameter groups
   - Integrate DMGD optimizer
   - Simple training loop

4. **Toy Regression Example** (Day 6-7)
   - Learn y = sin(x) + noise
   - Compare DMGD vs SGD/Adam
   - Basic visualization of training dynamics

### MVP Success Criteria

- [ ] DMGD optimizer trains a simple MLP
- [ ] CMS stores and retrieves information correctly
- [ ] Multi-frequency updates work as expected
- [ ] Example script runs without errors
- [ ] Results show promise of approach

**After MVP:** Iterate and expand based on initial results

---

## 📚 References

### Primary Source

**Paper:** "Nested Learning: The Illusion of Deep Learning Architectures"
- **Authors:** Ali Behrouz et al. (Google Research)
- **Conference:** NeurIPS 2025
- **Link:** https://abehrouz.github.io/files/NL.pdf
- **OpenReview:** https://openreview.net/forum?id=nbMeRvNb7A

### Related Work

**Continual Learning:**
- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks
- PackNet
- Memory Aware Synapses (MAS)

**Learned Optimization:**
- Learning to Learn by Gradient Descent by Gradient Descent
- Meta-SGD
- Learned Optimizers that Scale and Generalize

**Memory-Augmented Networks:**
- Neural Turing Machines
- Differentiable Neural Computers
- Memory Networks

**Efficient Transformers:**
- Linear Attention
- Performers
- Memory Transformers

### Implementation Resources

**PyTorch Resources:**
- Custom Optimizer Tutorial: https://pytorch.org/docs/stable/optim.html
- Memory-Efficient Training: https://pytorch.org/docs/stable/checkpoint.html

**Research Codebases:**
- Avalanche (Continual Learning): https://github.com/ContinualAI/avalanche
- Higher (Meta-Learning): https://github.com/facebookresearch/higher

---

## 🔄 Iteration Plan

### After Initial Implementation

1. **Community Feedback:**
   - Share on GitHub, Twitter, ML forums
   - Gather feedback on implementation
   - Identify bugs and limitations

2. **Ablation Studies:**
   - Which components are most important?
   - How sensitive to hyperparameters?
   - Where are the bottlenecks?

3. **Extensions:**
   - Apply to more complex tasks (language modeling, vision)
   - Scale to larger models
   - Integrate with popular frameworks (Hugging Face)

4. **Theoretical Analysis:**
   - Convergence properties of DMGD
   - Memory capacity of CMS
   - Relationship to other methods

### Long-Term Vision

- Production-ready library for Nested Learning
- Support for distributed training
- Pre-trained models with Nested Learning
- Integration with major ML frameworks
- Active research community

---

## 📝 Notes

### Implementation Philosophy

- **Start Simple:** Get basic version working first, then add complexity
- **Validate Continuously:** Test each component before moving on
- **Document Everything:** Code, decisions, experiments
- **Iterate Quickly:** Fast feedback loops, don't over-engineer
- **Stay Flexible:** Adapt plan based on results and challenges

### Assumptions and Simplifications

1. **Simplified Hope Architecture:** Full Titans architecture is complex; we'll implement core concepts
2. **Toy Datasets:** Start with MNIST, Fashion-MNIST, simple synthetic tasks
3. **Single GPU:** Multi-GPU support is future work
4. **PyTorch Only:** Not targeting TensorFlow/JAX initially
5. **Research Code:** Focus on clarity and experimentation over production optimization

### Open Questions

1. How to best initialize the momentum MLP in DMGD?
2. What's the optimal number of frequency levels in CMS?
3. How does this scale to very large models (billions of parameters)?
4. Can we automate hyperparameter selection?
5. What's the relationship between Nested Learning and existing continual learning methods?

**These will be explored during implementation.**

---

## 🎓 Learning Objectives

By completing this project, you will understand:

1. **Nested Optimization:** How to structure ML as hierarchical optimization problems
2. **Meta-Learning:** Learning the learning algorithm itself
3. **Memory Systems:** Multi-timescale memory in neural networks
4. **Continual Learning:** Mitigating catastrophic forgetting
5. **Advanced PyTorch:** Custom optimizers, complex training loops, memory management

---

## 🤝 Contributing Guidelines (Future)

Once the initial implementation is complete:

1. **Issues:** Use GitHub issues for bug reports and feature requests
2. **Pull Requests:** Welcome! Follow code style and include tests
3. **Discussions:** Use GitHub Discussions for questions and ideas
4. **Documentation:** Help improve docs and tutorials

---

## 📅 Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Foundation** | Week 1-2 | Project structure, dependencies, core math |
| **Phase 2: CMS** | Week 2-3 | Continuum Memory System, frequency scheduler |
| **Phase 3: DMGD** | Week 3-4 | Deep Momentum GD, nested optimizer |
| **Phase 4: Architecture** | Week 4-5 | Nested layers, CMS blocks, Hope model |
| **Phase 5: Training** | Week 5-6 | Training framework, memory consolidation |
| **Phase 6: Validation** | Week 6-7 | Examples, benchmarks, visualizations |
| **Phase 7: Polish** | Week 7-8 | Documentation, testing, refinement |

**Total Estimated Time:** 7-8 weeks for full implementation

**MVP Target:** 1 week for basic working prototype

---

## ✅ Next Steps

1. Review this plan and adjust based on your goals and timeline
2. Set up development environment
3. Create project structure
4. Begin Phase 1: Foundation & Core Components
5. Implement MVP (Basic DMGD + Simple CMS + NestedMLP)
6. Validate on toy problem
7. Iterate and expand

**Let's build Nested Learning!** 🚀
