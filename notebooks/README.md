# Nested Learning Jupyter Notebooks

Interactive notebooks for hands-on exploration of Nested Learning concepts.

## 📚 Notebook Series

### 1. Introduction to Nested Learning
**File:** `01_introduction_to_nested_learning.ipynb`

**What you'll learn:**
- Core concepts of Nested Learning
- Multi-frequency parameter updates
- Basic comparison with standard training
- Parameter assignment strategies

**Prerequisites:** Basic PyTorch knowledge

**Runtime:** ~10-15 minutes

**Topics covered:**
- Setting up a simple Nested Learning model
- Comparing training dynamics
- Understanding update frequency patterns
- Visualizing multi-frequency updates

---

### 2. Multi-Frequency Training Deep Dive
**File:** `02_multi_frequency_training.ipynb`

**What you'll learn:**
- Impact of different frequency configurations
- Learning rate strategies per level
- Optimizer selection for each frequency
- Hyperparameter tuning best practices

**Prerequisites:** Notebook 01

**Runtime:** ~20-30 minutes

**Topics covered:**
- Comparing frequency configurations: `[1,5,25]` vs `[1,10,100]` vs `[1,20,200]`
- Learning rate strategies: Increasing, uniform, decreasing
- Optimizer combinations: All Adam, Adam/SGD mix, All SGD
- Performance analysis and recommendations

---

### 3. Continuum Memory System (CMS)
**File:** `03_continuum_memory_system.ipynb`

**What you'll learn:**
- How CMS provides unbounded context
- Multi-level memory banks
- Memory consolidation process
- CMSBlock integration in networks

**Prerequisites:** Notebook 01

**Runtime:** ~15-20 minutes

**Topics covered:**
- Storing and retrieving memories
- Visualizing memory dynamics
- Memory utilization over time
- Associative retrieval quality
- Preventing catastrophic forgetting

---

### 4. Meta-Learning with DMGD *(Coming Soon)*
**File:** `04_meta_learning_dmgd.ipynb`

**What you'll learn:**
- Deep Momentum GD (learnable momentum)
- Meta-learning fundamentals
- Few-shot learning applications
- MAML-style optimization

**Prerequisites:** Notebooks 01-02

**Runtime:** ~25-30 minutes

---

### 5. Hope Architecture *(Coming Soon)*
**File:** `05_hope_architecture.ipynb`

**What you'll learn:**
- Hope model for language modeling
- Self-modifying architectures
- Long-context processing with CMS
- Text generation strategies

**Prerequisites:** Notebooks 01, 03

**Runtime:** ~20-25 minutes

---

## 🚀 Getting Started

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install Jupyter
pip install jupyter notebook

# Optional: For better visualization
pip install matplotlib seaborn
```

### Running the Notebooks

**Option 1: Jupyter Notebook**
```bash
cd notebooks
jupyter notebook
```

**Option 2: JupyterLab (recommended)**
```bash
cd notebooks
jupyter lab
```

**Option 3: VS Code**
- Open `.ipynb` files directly in VS Code
- Requires Python and Jupyter extensions

---

## 📋 Recommended Learning Path

### For Beginners
1. Start with **Notebook 01** (Introduction)
2. Try modifying the code examples
3. Move to **Notebook 02** (Multi-Frequency Training)
4. Explore **Notebook 03** (CMS)

### For Practitioners
1. Quick review of **Notebook 01**
2. Deep dive into **Notebook 02** for hyperparameter tuning
3. Study **Notebook 03** for continual learning applications
4. **Notebook 04** for meta-learning (when available)

### For Researchers
1. All notebooks in order
2. Modify experiments with your own data
3. Compare with baselines from `examples/` directory
4. Explore edge cases and failure modes

---

## 🎯 Learning Objectives

By completing these notebooks, you will:

### Understand
- ✅ Multi-frequency parameter updates
- ✅ How nested optimization works
- ✅ CMS architecture and memory consolidation
- ✅ When to use Nested Learning
- ✅ How to configure hyperparameters

### Be able to
- ✅ Build Nested Learning models
- ✅ Configure multi-frequency optimizers
- ✅ Integrate CMS into architectures
- ✅ Tune hyperparameters for your tasks
- ✅ Diagnose and fix common issues

---

## 💡 Tips for Success

### 1. Run Cells in Order
- Notebooks are designed to be run sequentially
- Don't skip cells unless noted as optional
- Restart kernel if you encounter errors

### 2. Experiment!
- Modify hyperparameters
- Try different datasets
- Add your own visualizations
- Break things and learn why

### 3. Take Notes
- Add markdown cells with your observations
- Document what works and what doesn't
- Save interesting parameter configurations

### 4. Compare Results
- Run experiments multiple times with different random seeds
- Compare with standard baselines
- Look for patterns in the results

---

## 🔧 Troubleshooting

### Notebook won't start
```bash
# Reinstall Jupyter
pip install --upgrade jupyter notebook

# Check Python version (3.7+ required)
python --version
```

### Import errors
```bash
# Make sure you're in the right directory
cd nested-learning-sample/notebooks

# Install missing dependencies
pip install -r ../requirements.txt
```

### Out of memory
- Reduce batch sizes in examples
- Use CPU instead of GPU for exploration
- Close other applications
- Restart kernel and clear outputs

### Plots not showing
```python
# Add at the top of notebook
%matplotlib inline

# Or for interactive plots
%matplotlib widget
```

---

## 📊 Notebook Features

### Interactive Elements
- ✅ Editable code cells
- ✅ Live visualizations
- ✅ Real-time training progress
- ✅ Hyperparameter sliders (in some notebooks)

### Visualizations
- Training loss curves
- Update frequency patterns
- Memory utilization heatmaps
- Parameter distribution plots
- Comparison charts

### Code Organization
- Clear section headers
- Documented functions
- Inline comments
- Type hints where helpful

---

## 🎓 Additional Resources

### Related Examples
- `examples/quick_comparison.py` - Fast script version of Notebook 01
- `examples/mnist_classification.py` - Full MNIST benchmark
- `examples/continual_learning.py` - Catastrophic forgetting demo

### Documentation
- `../README.md` - Project overview
- `../IMPLEMENTATION_STATUS.md` - Detailed component status
- `../examples/README.md` - Example scripts guide
- `../projectplan.md` - Development roadmap

### Paper
- [Nested Learning Paper](https://abehrouz.github.io/files/NL.pdf) - Original research paper

---

## 🤝 Contributing

Have improvements or new notebook ideas?

1. Create a new notebook following the naming convention
2. Add clear markdown documentation
3. Include visualizations
4. Test thoroughly
5. Update this README
6. Submit a pull request

### Notebook Guidelines
- Clear learning objectives
- Progressive difficulty
- Plenty of visualizations
- Runnable in <30 minutes
- Well-commented code

---

## 📝 Notebook Checklist

Before submitting a new notebook:

- [ ] Clear title and description
- [ ] Learning objectives stated
- [ ] Prerequisites listed
- [ ] Code runs without errors
- [ ] Visualizations render correctly
- [ ] Markdown cells for explanations
- [ ] Key takeaways section
- [ ] Links to next notebook
- [ ] Runtime estimate provided
- [ ] Tested on fresh kernel

---

## 🆘 Getting Help

### Issues
- Check troubleshooting section above
- Review notebook output carefully
- Try restarting the kernel
- Check for dependency conflicts

### Questions
- Open a GitHub issue
- Include error messages
- Specify which notebook
- Share your environment details

### Community
- Share your experiments!
- Post interesting findings
- Help others learn

---

## 🎯 Quick Reference

### Keyboard Shortcuts (Jupyter)
- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell
- `M`: Change to markdown
- `Y`: Change to code

### Common Commands
```python
# Reset random seed
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clear GPU memory
torch.cuda.empty_cache()

# Save model
torch.save(model.state_dict(), 'model.pt')
```

---

## 📈 Progress Tracking

Track your learning progress:

- [ ] Completed Notebook 01 - Introduction
- [ ] Completed Notebook 02 - Multi-Frequency Training
- [ ] Completed Notebook 03 - Continuum Memory System
- [ ] Completed Notebook 04 - Meta-Learning with DMGD
- [ ] Completed Notebook 05 - Hope Architecture
- [ ] Ran examples from `examples/` directory
- [ ] Modified code with own data
- [ ] Built a custom Nested Learning model

---

## 🌟 Next Steps

After completing the notebooks:

1. **Apply to your own data**
   - Start with `quick_comparison.py` template
   - Adapt to your dataset
   - Experiment with configurations

2. **Read the paper**
   - Understand theoretical foundations
   - Compare with implementations
   - Explore advanced topics

3. **Contribute**
   - Share your experiments
   - Report issues or bugs
   - Suggest improvements
   - Create new examples

4. **Stay updated**
   - Watch the repository
   - Check for new notebooks
   - Follow development progress

---

**Happy Learning! 🚀**

For questions or feedback, please open an issue on GitHub.

---

**Last Updated:** November 12, 2025
**Status:** Notebooks 01-03 Complete | 04-05 Coming Soon
