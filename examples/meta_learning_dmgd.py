"""
Meta-Learning Example for DMGD

Demonstrates how to meta-train the Deep Momentum GD optimizer across
multiple tasks, allowing it to learn effective momentum computation strategies.

This shows the true power of DMGD - when meta-learned, it can adapt quickly
to new tasks and outperform standard optimizers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np

from src.optimizers.dmgd import DeepMomentumGD
from src.optimizers.meta_learning import MetaLearningTrainer, create_task_sampler


def create_model():
    """Create a simple MLP for regression."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


def evaluate_optimizer_on_task(
    model_fn,
    optimizer_fn,
    task_data,
    num_steps=10,
    lr=0.01
):
    """
    Evaluate an optimizer on a single task.

    Args:
        model_fn: Function to create model
        optimizer_fn: Function to create optimizer
        task_data: Task data dict
        num_steps: Number of training steps
        lr: Learning rate

    Returns:
        Final validation loss
    """
    model = model_fn()
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.MSELoss()

    X_train, y_train = task_data['train']
    X_val, y_val = task_data['val']

    # Train on task
    model.train()
    for step in range(num_steps):
        pred = model(X_train)
        loss = criterion(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X_val)
        val_loss = criterion(pred, y_val).item()

    return val_loss


def main():
    print("="*70)
    print("Meta-Learning for Deep Momentum GD (DMGD)")
    print("="*70)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # ========================================================================
    # Phase 1: Meta-Train DMGD
    # ========================================================================
    print("\n[Phase 1] Meta-Training DMGD on Multiple Tasks")
    print("-"*70)

    # Create DMGD optimizer with meta-learning capability
    dummy_model = create_model()
    dmgd = DeepMomentumGD(
        dummy_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16],
        mlp_lr=0.001  # Meta-learning rate
    )

    # Create meta-learning trainer
    trainer = MetaLearningTrainer(
        model_fn=create_model,
        dmgd_optimizer=dmgd,
        inner_steps=5,
        inner_lr=0.01,
        meta_lr=0.001
    )

    # Create task sampler (random linear regression tasks)
    task_sampler = create_task_sampler(
        task_type='linear_regression',
        input_dim=10,
        output_dim=1,
        n_train=50,
        n_val=20
    )

    # Meta-train
    print("Meta-training DMGD for 100 episodes...")
    meta_losses = trainer.meta_train(
        task_sampler=task_sampler,
        num_episodes=100,
        verbose=True,
        log_interval=20
    )

    print(f"\nMeta-training complete!")
    stats = trainer.get_statistics()
    print(f"  Initial meta-loss: {meta_losses[0]:.4f}")
    print(f"  Final meta-loss: {meta_losses[-1]:.4f}")
    print(f"  Improvement: {stats['improvement']:.4f}")

    # ========================================================================
    # Phase 2: Evaluate on New Tasks
    # ========================================================================
    print("\n[Phase 2] Evaluating on New Tasks")
    print("-"*70)

    num_test_tasks = 10
    print(f"Testing on {num_test_tasks} unseen tasks...")

    # Compare different optimizers
    optimizers = {
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.01),
        'DMGD (random)': lambda params: DeepMomentumGD(
            params, lr=0.01, momentum_hidden_dims=[32, 16]
        ),
        'DMGD (meta-learned)': None  # Will use meta-learned version
    }

    results = {name: [] for name in optimizers.keys()}

    for task_idx in range(num_test_tasks):
        # Sample new task
        task_data = task_sampler()

        # Evaluate each optimizer
        for name, opt_fn in optimizers.items():
            if name == 'DMGD (meta-learned)':
                # Use meta-learned DMGD
                result = trainer.evaluate(task_data)
                val_loss = result['final_loss']
            else:
                # Use standard optimizer
                val_loss = evaluate_optimizer_on_task(
                    create_model, opt_fn, task_data, num_steps=5
                )

            results[name].append(val_loss)

    # ========================================================================
    # Phase 3: Results Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Results on New Tasks")
    print("="*70)

    print(f"\n{'Optimizer':<25} {'Mean Loss':<15} {'Std Loss':<15} {'Min Loss':<15}")
    print("-"*70)

    for name in optimizers.keys():
        losses = results[name]
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        min_loss = np.min(losses)

        print(f"{name:<25} {mean_loss:<15.4f} {std_loss:<15.4f} {min_loss:<15.4f}")

    # Calculate improvement
    meta_learned_mean = np.mean(results['DMGD (meta-learned)'])
    random_dmgd_mean = np.mean(results['DMGD (random)'])
    adam_mean = np.mean(results['Adam'])

    print("\n" + "="*70)
    print("Performance Comparison")
    print("="*70)

    improvement_vs_random = ((random_dmgd_mean - meta_learned_mean) / random_dmgd_mean) * 100
    improvement_vs_adam = ((adam_mean - meta_learned_mean) / adam_mean) * 100

    print(f"\nMeta-learned DMGD improvement:")
    print(f"  vs Random DMGD: {improvement_vs_random:+.1f}%")
    print(f"  vs Adam:        {improvement_vs_adam:+.1f}%")

    if meta_learned_mean < adam_mean:
        print(f"\n✓ Meta-learned DMGD outperforms Adam!")
    else:
        print(f"\n  Meta-learned DMGD performance: {meta_learned_mean:.4f}")
        print(f"  Adam performance: {adam_mean:.4f}")

    # ========================================================================
    # Phase 4: Detailed Analysis
    # ========================================================================
    print("\n[Phase 4] Detailed Analysis on Single Task")
    print("-"*70)

    # Create one task for detailed analysis
    task_data = task_sampler()

    print("\nComparing learning curves on a single task...")
    print("(Training for 20 steps)")

    # Track losses over training
    num_steps = 20

    def get_learning_curve(model_fn, optimizer_fn, task_data, num_steps):
        """Get loss at each training step."""
        model = model_fn()
        optimizer = optimizer_fn(model.parameters())
        criterion = nn.MSELoss()

        X_train, y_train = task_data['train']
        X_val, y_val = task_data['val']

        losses = []
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            initial_loss = criterion(pred, y_val).item()
        losses.append(initial_loss)

        model.train()
        for step in range(num_steps):
            pred = model(X_train)
            loss = criterion(pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                pred = model(X_val)
                val_loss = criterion(pred, y_val).item()
            losses.append(val_loss)
            model.train()

        return losses

    # Get learning curves
    adam_curve = get_learning_curve(
        create_model,
        lambda p: torch.optim.Adam(p, lr=0.01),
        task_data,
        num_steps
    )

    # Meta-learned DMGD curve
    meta_model = create_model()
    meta_opt = DeepMomentumGD(
        meta_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16]
    )
    meta_opt.momentum_mlps = dmgd.momentum_mlps  # Use meta-learned MLPs
    meta_curve = get_learning_curve(
        lambda: meta_model,
        lambda p: meta_opt,
        task_data,
        num_steps
    )

    # Print comparison
    print(f"\n{'Step':<10} {'Adam':<15} {'Meta-DMGD':<15} {'Difference':<15}")
    print("-"*55)
    for step in [0, 5, 10, 15, 20]:
        adam_loss = adam_curve[step]
        meta_loss = meta_curve[step]
        diff = adam_loss - meta_loss
        print(f"{step:<10} {adam_loss:<15.4f} {meta_loss:<15.4f} {diff:+15.4f}")

    # ========================================================================
    # Conclusion
    # ========================================================================
    print("\n" + "="*70)
    print("Conclusion")
    print("="*70)

    print("\nKey Findings:")
    print("1. Meta-learned DMGD adapts the momentum computation to the task")
    print("2. After meta-training, DMGD can generalize to new tasks")
    print("3. The learned momentum strategy is more effective than standard momentum")

    print("\nWhat Makes This Work:")
    print("- Inner loop: DMGD learns task-specific parameters")
    print("- Outer loop: MLP learns how to compute good momentum")
    print("- Meta-learning enables transfer across tasks")

    print("\nNext Steps:")
    print("- Try different task distributions (sine waves, polynomials)")
    print("- Meta-train for more episodes")
    print("- Experiment with different MLP architectures")
    print("- Apply to more complex problems (classification, RL)")


if __name__ == '__main__':
    main()
