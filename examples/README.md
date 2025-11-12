# Nested Learning Examples

This directory contains practical examples demonstrating Nested Learning on real-world tasks.

## Quick Start Examples

### 1. Quick Comparison (`quick_comparison.py`)
**Fastest way to see Nested Learning in action!**

A lightweight comparison that runs in minutes on CPU using synthetic data.

```bash
python examples/quick_comparison.py
```

**What it demonstrates:**
- Multi-frequency parameter updates
- Training efficiency comparison
- Generalization performance
- Visual comparison plots

**Runtime:** ~2-3 minutes on CPU

---

### 2. Simple Examples

**Continuum Memory System** (`simple_cms_example.py`)
```bash
python examples/simple_cms_example.py
```
Demonstrates multi-level memory banks with frequency-based updates.

**Deep Momentum GD** (`simple_dmgd_example.py`)
```bash
python examples/simple_dmgd_example.py
```
Shows learnable MLP-based momentum optimization.

---

## Advanced Examples

### 3. MNIST Classification (`mnist_classification.py`)
**Complete benchmark on a standard dataset**

Comprehensive comparison of Nested Learning vs standard training on MNIST.

```bash
python examples/mnist_classification.py
```

**What it demonstrates:**
- NestedMLP with multi-frequency layers
- NestedOptimizer with 3 frequency levels (Fast/Medium/Slow)
- NestedTrainer with callbacks (EarlyStopping, Checkpointing)
- Learning rate scheduling
- Full training/validation/test pipeline
- Side-by-side comparison with standard MLP

**Features:**
- Automatic data download
- Progress tracking
- Checkpoint saving
- Comparison visualization
- Detailed metrics

**Expected Results:**
- Training time: ~5-10 minutes on CPU, ~2-3 minutes on GPU
- Test accuracy: 97-98% (both approaches competitive)
- Shows training dynamics differences

**Output:**
- `mnist_comparison.png` - Training curves and accuracy plots
- `best_nested_mnist.pt` - Best nested model checkpoint
- `best_standard_mnist.pt` - Best standard model checkpoint

---

### 4. Continual Learning (`continual_learning.py`)
**Preventing catastrophic forgetting**

Demonstrates how Nested Learning with CMS retains knowledge when learning sequential tasks.

```bash
python examples/continual_learning.py
```

**What it demonstrates:**
- Task 1: Learn digits 0-4
- Task 2: Learn digits 5-9
- Measure how well Task 1 is retained after learning Task 2

**Key Insights:**
- **Standard MLP:** Shows significant catastrophic forgetting (~20-30% accuracy drop on Task 1)
- **Nested Learning:** Better retention through multi-frequency updates (~10-15% drop)

**Metrics:**
- Forgetting: Drop in Task 1 accuracy after Task 2 training
- Retention: Percentage of Task 1 knowledge preserved

**Expected Results:**
- Nested Learning shows 40-60% better retention than standard
- Demonstrates the power of multi-timescale learning

**Output:**
- `continual_learning.png` - Forgetting and retention comparison
- Detailed forgetting analysis

---

### 5. Meta-Learning Examples

**Regression Tasks** (`meta_learning_practical.py`)
```bash
python examples/meta_learning_practical.py
```
Comprehensive meta-learning on regression tasks (sine waves, linear, polynomial).

**Classification Tasks** (`meta_learning_classification.py`)
```bash
python examples/meta_learning_classification.py
```
Meta-learning for few-shot classification (binary and multi-class).

**What they demonstrate:**
- MAML-style meta-learning with DMGD
- Few-shot learning scenarios
- Task distribution sampling
- Meta-training and meta-testing
- Comparison with standard optimizers

---

## Running the Examples

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# If you want to visualize results
pip install matplotlib

# Optional: For progress bars
pip install tqdm
```

### Basic Usage

```bash
# From the repository root
python examples/<example_name>.py

# Or from the examples directory
cd examples
python <example_name>.py
```

### GPU Usage

All examples automatically use GPU if available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Example Comparison

| Example | Dataset | Runtime (CPU) | Runtime (GPU) | Difficulty | Key Feature |
|---------|---------|---------------|---------------|------------|-------------|
| `quick_comparison.py` | Synthetic | 2-3 min | <1 min | ⭐ Easy | Quick demo |
| `simple_cms_example.py` | Synthetic | <1 min | <1 min | ⭐ Easy | Memory system |
| `simple_dmgd_example.py` | Synthetic | <1 min | <1 min | ⭐ Easy | Learnable momentum |
| `mnist_classification.py` | MNIST | 10-15 min | 2-3 min | ⭐⭐ Medium | Full pipeline |
| `continual_learning.py` | MNIST | 15-20 min | 3-5 min | ⭐⭐ Medium | Catastrophic forgetting |
| `meta_learning_practical.py` | Synthetic | 5-10 min | 2-3 min | ⭐⭐⭐ Advanced | Meta-learning |
| `meta_learning_classification.py` | Synthetic | 5-10 min | 2-3 min | ⭐⭐⭐ Advanced | Few-shot learning |

---

## Understanding the Results

### Multi-Frequency Training Benefits

**When Nested Learning Excels:**
1. **Continual Learning:** Prevents catastrophic forgetting
2. **Long Training Runs:** Better long-term stability
3. **Complex Tasks:** Separates fast/slow learning components
4. **Transfer Learning:** Stable low-level features, adaptable high-level features

**What to Look For:**
- Smoother training curves
- Better generalization on validation set
- Improved retention in continual learning
- Faster convergence on new tasks (meta-learning)

### Interpreting Plots

**Training Loss:**
- Nested Learning often shows smoother curves
- May converge slightly slower initially (slower params updated less frequently)
- Usually achieves lower final loss

**Test/Validation Accuracy:**
- Better generalization (less overfitting)
- More stable across epochs
- Particularly strong on continual learning tasks

**Catastrophic Forgetting:**
- Standard: Large accuracy drop on old tasks (~20-30%)
- Nested: Smaller accuracy drop (~10-15%)
- Retention rate: Nested typically 40-60% better

---

## Customization

### Adjusting Frequency Levels

```python
# Fast updates for quick adaptation
builder.build(
    optimizer_types=['adam', 'sgd', 'sgd'],
    learning_rates=[0.001, 0.01, 0.1],
    frequencies=[1, 5, 25]  # Adjust these!
)
```

**Common Patterns:**
- `[1, 10, 100]` - Default (good for most tasks)
- `[1, 5, 25]` - Faster adaptation
- `[1, 20, 200]` - More stable, slower adaptation
- `[1, 50, 500]` - Very stable, research settings

### Adjusting Architecture

```python
# More capacity
model = NestedMLP(
    input_dim=784,
    hidden_dims=[512, 256, 128],  # Deeper network
    output_dim=10,
    dropout=0.3  # More regularization
)
```

---

## Troubleshooting

### Low Accuracy
- Check data normalization
- Try different learning rates
- Increase model capacity
- Train for more epochs

### Slow Training
- Reduce batch size (if GPU memory limited)
- Use GPU instead of CPU
- Reduce model size for quick experiments

### Out of Memory
- Reduce batch size
- Reduce model size
- Use gradient accumulation

### No Improvement Over Standard
- Some tasks may not benefit from multi-frequency updates
- Try different frequency configurations
- Ensure data has multi-scale structure

---

## Next Steps

1. **Start with `quick_comparison.py`** to see the basics
2. **Run `mnist_classification.py`** for a complete benchmark
3. **Try `continual_learning.py`** to see catastrophic forgetting prevention
4. **Explore meta-learning** examples for few-shot scenarios
5. **Adapt examples** to your own datasets and tasks

---

## Contributing

Have a cool example to share? Please contribute!

1. Create your example script
2. Add clear documentation
3. Include expected results
4. Add to this README
5. Submit a pull request

---

## Citation

If you use these examples in your research, please cite:

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

## License

MIT License - See LICENSE file for details
