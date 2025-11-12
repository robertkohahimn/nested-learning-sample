"""
Practical Meta-Learning Examples with DMGD

This script demonstrates practical use cases for meta-learning with the
Deep Momentum Gradient Descent optimizer. It shows:

1. Few-shot learning on regression tasks
2. Meta-learning for quick adaptation
3. Comparison with standard optimizers
4. Best practices for hyperparameter selection

Note: Current implementation has limited gradient flow to the momentum MLP.
The examples demonstrate the framework and show qualitative behavior.
"""

import sys
sys.path.insert(0, '/home/user/nested-learning-sample')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.optimizers.dmgd import DeepMomentumGD
from src.optimizers.meta_learning import MetaLearningTrainer, create_task_sampler


def create_regression_model():
    """Simple regression model for meta-learning."""
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


def example1_sine_wave_adaptation():
    """
    Example 1: Meta-Learning for Quick Adaptation to Sine Waves

    Goal: Learn an optimizer that can quickly adapt to new sine wave tasks
    with only a few gradient steps (few-shot learning).
    """
    print("="*70)
    print("Example 1: Few-Shot Learning on Sine Wave Tasks")
    print("="*70)

    # Create DMGD optimizer (shared across episodes)
    dummy_model = create_regression_model()
    dmgd = DeepMomentumGD(
        dummy_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16],
        mlp_lr=0.001
    )

    # Create meta-learning trainer
    trainer = MetaLearningTrainer(
        model_fn=create_regression_model,
        dmgd_optimizer=dmgd,
        inner_steps=5,  # Only 5 gradient steps per task
        inner_lr=0.01,
        meta_lr=0.001
    )

    # Create sine wave task sampler
    sampler = create_task_sampler(
        task_type='sine',
        input_dim=1,
        output_dim=1,
        n_train=25,  # Few training samples
        n_val=25
    )

    print("\n1. Meta-Training Phase")
    print("-" * 70)
    print(f"Inner steps: {trainer.inner_steps}")
    print(f"Training samples per task: 25")
    print(f"Meta-training for 100 episodes...")

    # Meta-train
    meta_losses = trainer.meta_train(
        task_sampler=sampler,
        num_episodes=100,
        verbose=True,
        log_interval=20
    )

    print(f"\nMeta-training complete!")
    print(f"  Initial meta-loss: {meta_losses[0]:.4f}")
    print(f"  Final meta-loss: {meta_losses[-1]:.4f}")
    print(f"  Improvement: {meta_losses[0] - meta_losses[-1]:.4f}")

    # Evaluate on new tasks
    print("\n2. Evaluation on New Tasks")
    print("-" * 70)

    results_list = []
    for i in range(5):
        task = sampler()
        results = trainer.evaluate(task)
        results_list.append(results)
        print(f"Task {i+1}: Initial loss={results['initial_loss']:.4f}, "
              f"Final loss={results['final_loss']:.4f}, "
              f"Improvement={results['improvement']:.4f}")

    avg_improvement = np.mean([r['improvement'] for r in results_list])
    print(f"\nAverage improvement: {avg_improvement:.4f}")

    return meta_losses, results_list


def example2_comparison_with_standard_optimizers():
    """
    Example 2: Compare Meta-Learned DMGD with Standard Optimizers

    Shows how meta-learned optimizer performs compared to SGD and Adam
    on few-shot regression tasks.
    """
    print("\n" + "="*70)
    print("Example 2: Comparison with Standard Optimizers")
    print("="*70)

    # Meta-train DMGD
    print("\n1. Meta-Training DMGD...")
    dummy_model = create_regression_model()
    dmgd = DeepMomentumGD(
        dummy_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16],
        mlp_lr=0.001
    )

    trainer = MetaLearningTrainer(
        model_fn=create_regression_model,
        dmgd_optimizer=dmgd,
        inner_steps=10,
        inner_lr=0.01,
        meta_lr=0.001
    )

    sampler = create_task_sampler(task_type='sine', n_train=20, n_val=20)

    # Quick meta-training (50 episodes)
    trainer.meta_train(sampler, num_episodes=50, verbose=False)
    print("   Meta-training complete!")

    # Test on new tasks
    print("\n2. Testing on 10 New Tasks")
    print("-" * 70)

    def evaluate_optimizer(opt_class, opt_kwargs, task, num_steps=10):
        """Helper to evaluate an optimizer on a task."""
        model = create_regression_model()
        optimizer = opt_class(model.parameters(), **opt_kwargs)
        criterion = nn.MSELoss()

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

    sgd_losses = []
    adam_losses = []
    dmgd_losses = []

    for i in range(10):
        task = sampler()

        sgd_loss = evaluate_optimizer(
            torch.optim.SGD,
            {'lr': 0.01, 'momentum': 0.9},
            task
        )

        adam_loss = evaluate_optimizer(
            torch.optim.Adam,
            {'lr': 0.01},
            task
        )

        dmgd_result = trainer.evaluate(task)
        dmgd_loss = dmgd_result['final_loss']

        sgd_losses.append(sgd_loss)
        adam_losses.append(adam_loss)
        dmgd_losses.append(dmgd_loss)

    print(f"\nResults over 10 tasks:")
    print(f"  SGD (momentum=0.9):  {np.mean(sgd_losses):.4f} ± {np.std(sgd_losses):.4f}")
    print(f"  Adam:                {np.mean(adam_losses):.4f} ± {np.std(adam_losses):.4f}")
    print(f"  Meta-Learned DMGD:   {np.mean(dmgd_losses):.4f} ± {np.std(dmgd_losses):.4f}")

    # Determine winner
    best_optimizer = min(
        [('SGD', np.mean(sgd_losses)),
         ('Adam', np.mean(adam_losses)),
         ('DMGD', np.mean(dmgd_losses))],
        key=lambda x: x[1]
    )[0]

    print(f"\nBest performer: {best_optimizer}")

    return {
        'sgd': sgd_losses,
        'adam': adam_losses,
        'dmgd': dmgd_losses
    }


def example3_hyperparameter_tuning():
    """
    Example 3: Impact of Hyperparameters

    Demonstrates how different hyperparameters affect meta-learning performance.
    """
    print("\n" + "="*70)
    print("Example 3: Hyperparameter Impact")
    print("="*70)

    sampler = create_task_sampler(task_type='polynomial', n_train=30, n_val=20)

    configs = [
        {'name': 'Few inner steps', 'inner_steps': 3, 'mlp_hidden': [16]},
        {'name': 'Many inner steps', 'inner_steps': 10, 'mlp_hidden': [16]},
        {'name': 'Large MLP', 'inner_steps': 5, 'mlp_hidden': [64, 32]},
        {'name': 'Small MLP', 'inner_steps': 5, 'mlp_hidden': [8]},
    ]

    print("\nTesting different configurations:")
    print("-" * 70)

    results = {}

    for config in configs:
        print(f"\n{config['name']}: inner_steps={config['inner_steps']}, "
              f"MLP={config['mlp_hidden']}")

        dummy_model = create_regression_model()
        dmgd = DeepMomentumGD(
            dummy_model.parameters(),
            lr=0.01,
            momentum_hidden_dims=config['mlp_hidden'],
            mlp_lr=0.001
        )

        trainer = MetaLearningTrainer(
            model_fn=create_regression_model,
            dmgd_optimizer=dmgd,
            inner_steps=config['inner_steps'],
            inner_lr=0.01,
            meta_lr=0.001
        )

        # Short meta-training
        meta_losses = trainer.meta_train(sampler, num_episodes=30, verbose=False)

        # Evaluate
        eval_results = []
        for _ in range(3):
            task = sampler()
            eval_results.append(trainer.evaluate(task)['improvement'])

        avg_improvement = np.mean(eval_results)
        final_meta_loss = meta_losses[-1]

        print(f"  Final meta-loss: {final_meta_loss:.4f}")
        print(f"  Avg improvement on new tasks: {avg_improvement:.4f}")

        results[config['name']] = {
            'meta_losses': meta_losses,
            'avg_improvement': avg_improvement
        }

    # Find best config
    best_config = max(results.items(), key=lambda x: x[1]['avg_improvement'])
    print(f"\nBest configuration: {best_config[0]}")

    return results


def example4_task_diversity():
    """
    Example 4: Meta-Learning Across Different Task Types

    Shows behavior when meta-training on diverse tasks vs. similar tasks.
    """
    print("\n" + "="*70)
    print("Example 4: Task Diversity Impact")
    print("="*70)

    print("\n1. Meta-Training on Single Task Type (Sine)")
    print("-" * 70)

    # Train on only sine waves
    sampler_sine = create_task_sampler(task_type='sine')

    dummy_model = create_regression_model()
    dmgd_sine = DeepMomentumGD(
        dummy_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16],
        mlp_lr=0.001
    )

    trainer_sine = MetaLearningTrainer(
        model_fn=create_regression_model,
        dmgd_optimizer=dmgd_sine,
        inner_steps=5,
        inner_lr=0.01,
        meta_lr=0.001
    )

    trainer_sine.meta_train(sampler_sine, num_episodes=50, verbose=False)

    # Evaluate on sine and polynomial
    print("\nEvaluation on sine tasks:")
    sine_results = []
    for _ in range(5):
        task = sampler_sine()
        sine_results.append(trainer_sine.evaluate(task)['improvement'])

    print(f"  Average improvement: {np.mean(sine_results):.4f}")

    sampler_poly = create_task_sampler(task_type='polynomial')
    print("\nEvaluation on polynomial tasks (out-of-distribution):")
    poly_results = []
    for _ in range(5):
        task = sampler_poly()
        poly_results.append(trainer_sine.evaluate(task)['improvement'])

    print(f"  Average improvement: {np.mean(poly_results):.4f}")

    return {
        'sine_on_sine': np.mean(sine_results),
        'sine_on_poly': np.mean(poly_results)
    }


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PRACTICAL META-LEARNING EXAMPLES WITH DMGD")
    print("="*70)

    print("\nThese examples demonstrate the meta-learning framework.")
    print("Current limitation: Gradient flow to momentum MLP is limited.")
    print("The framework is functional and shows qualitative behavior.\n")

    # Run examples
    try:
        print("\n" + "#"*70)
        print("# RUNNING EXAMPLES")
        print("#"*70)

        # Example 1: Few-shot learning
        meta_losses1, results1 = example1_sine_wave_adaptation()

        # Example 2: Comparison
        comparison_results = example2_comparison_with_standard_optimizers()

        # Example 3: Hyperparameters
        hp_results = example3_hyperparameter_tuning()

        # Example 4: Task diversity
        diversity_results = example4_task_diversity()

        # Summary
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETE")
        print("="*70)

        print("\nKey Takeaways:")
        print("  1. Meta-learning framework is functional and flexible")
        print("  2. Different hyperparameters significantly affect performance")
        print("  3. Task diversity impacts generalization")
        print("  4. The system provides a foundation for learned optimization")

        print("\nNext Steps:")
        print("  - Fine-tune hyperparameters for your specific task distribution")
        print("  - Experiment with different MLP architectures")
        print("  - Try meta-learning on your domain-specific tasks")
        print("  - Consider gradient flow improvements for better MLP learning")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
