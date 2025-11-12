"""
Meta-Learning for Classification Tasks

This example demonstrates how to use meta-learning with DMGD for
classification problems, including:

1. Binary classification
2. Multi-class classification
3. Quick adaptation to new classification tasks
"""

import sys
sys.path.insert(0, '/home/user/nested-learning-sample')

import torch
import torch.nn as nn
import numpy as np

from src.optimizers.dmgd import DeepMomentumGD
from src.optimizers.meta_learning import MetaLearningTrainer, create_task_sampler


def create_binary_classifier(input_dim=10):
    """Create binary classification model."""
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )


def create_multiclass_classifier(input_dim=10, num_classes=3):
    """Create multi-class classification model."""
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes)
    )


def binary_classification_example():
    """
    Example: Meta-Learning for Binary Classification

    Scenario: Learning to quickly classify different binary classification tasks
    with only a few examples (few-shot classification).
    """
    print("="*70)
    print("Binary Classification Meta-Learning")
    print("="*70)

    input_dim = 10

    # Create DMGD with proper input dimensions
    # For binary classifier: parameters vary, MLP input = grad + momentum
    def create_model():
        return create_binary_classifier(input_dim=input_dim)

    dummy_model = create_model()
    dmgd = DeepMomentumGD(
        dummy_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16],
        mlp_lr=0.001,
        use_params_in_mlp=False  # Simpler: only use grad + momentum
    )

    # Create trainer
    trainer = MetaLearningTrainer(
        model_fn=create_model,
        dmgd_optimizer=dmgd,
        inner_steps=10,  # 10 gradient steps per task
        inner_lr=0.01,
        meta_lr=0.001
    )

    # Create binary classification task sampler
    sampler = create_task_sampler(
        task_type='binary_classification',
        input_dim=input_dim,
        output_dim=1,
        n_train=40,  # 40 training examples
        n_val=20     # 20 validation examples
    )

    print("\n1. Meta-Training Phase")
    print("-" * 70)
    print(f"Task type: Binary classification")
    print(f"Input dimension: {input_dim}")
    print(f"Training samples per task: 40")
    print(f"Inner loop steps: {trainer.inner_steps}")

    # Meta-train
    print("\nMeta-training for 80 episodes...")
    meta_losses = trainer.meta_train(
        task_sampler=sampler,
        num_episodes=80,
        criterion=nn.BCELoss(),
        verbose=True,
        log_interval=20
    )

    print(f"\nMeta-training statistics:")
    stats = trainer.get_statistics()
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Mean meta-loss: {stats['mean_meta_loss']:.4f}")
    print(f"  Final meta-loss: {stats['final_meta_loss']:.4f}")

    # Evaluate on new tasks
    print("\n2. Evaluation on New Binary Classification Tasks")
    print("-" * 70)

    accuracies = []
    improvements = []

    for i in range(5):
        task = sampler()

        # Evaluate with trained optimizer
        results = trainer.evaluate(task, criterion=nn.BCELoss())

        # Calculate accuracy
        model = create_model()
        X_val, y_val = task['val']

        # Initial accuracy
        with torch.no_grad():
            pred_initial = (model(X_val) > 0.5).float()
            acc_initial = (pred_initial == y_val).float().mean().item()

        # Train with meta-learned optimizer
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=0.01,
            momentum_hidden_dims=[32, 16]
        )
        optimizer.momentum_mlps = dmgd.momentum_mlps  # Use meta-learned MLP

        X_train, y_train = task['train']
        criterion = nn.BCELoss()

        for _ in range(trainer.inner_steps):
            pred = model(X_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final accuracy
        with torch.no_grad():
            pred_final = (model(X_val) > 0.5).float()
            acc_final = (pred_final == y_val).float().mean().item()

        print(f"Task {i+1}: Initial acc={acc_initial:.3f}, "
              f"Final acc={acc_final:.3f}, "
              f"Improvement={acc_final - acc_initial:.3f}")

        accuracies.append(acc_final)
        improvements.append(acc_final - acc_initial)

    print(f"\nAverage final accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"Average improvement: {np.mean(improvements):.3f} ± {np.std(improvements):.3f}")

    return meta_losses, accuracies


def multiclass_classification_example():
    """
    Example: Meta-Learning for Multi-Class Classification

    Scenario: Learning to quickly adapt to different 5-class classification tasks.
    """
    print("\n" + "="*70)
    print("Multi-Class Classification Meta-Learning")
    print("="*70)

    input_dim = 10
    num_classes = 5

    # Create model function
    def create_model():
        return create_multiclass_classifier(input_dim=input_dim, num_classes=num_classes)

    dummy_model = create_model()
    dmgd = DeepMomentumGD(
        dummy_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16],
        mlp_lr=0.001,
        use_params_in_mlp=False
    )

    # Create trainer
    trainer = MetaLearningTrainer(
        model_fn=create_model,
        dmgd_optimizer=dmgd,
        inner_steps=10,
        inner_lr=0.01,
        meta_lr=0.001
    )

    # Create multi-class task sampler
    sampler = create_task_sampler(
        task_type='multiclass_classification',
        input_dim=input_dim,
        output_dim=num_classes,
        n_train=50,
        n_val=30
    )

    print("\n1. Meta-Training Phase")
    print("-" * 70)
    print(f"Task type: {num_classes}-class classification")
    print(f"Input dimension: {input_dim}")
    print(f"Training samples per task: 50")

    # Meta-train
    print("\nMeta-training for 80 episodes...")
    meta_losses = trainer.meta_train(
        task_sampler=sampler,
        num_episodes=80,
        criterion=nn.CrossEntropyLoss(),
        verbose=True,
        log_interval=20
    )

    print(f"\nMeta-training complete!")

    # Evaluate on new tasks
    print("\n2. Evaluation on New Multi-Class Tasks")
    print("-" * 70)

    accuracies = []
    improvements = []

    for i in range(5):
        task = sampler()

        # Create model
        model = create_model()
        X_val, y_val = task['val']

        # Initial accuracy
        with torch.no_grad():
            pred_initial = model(X_val).argmax(dim=1)
            acc_initial = (pred_initial == y_val).float().mean().item()

        # Train with meta-learned optimizer
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=0.01,
            momentum_hidden_dims=[32, 16]
        )
        optimizer.momentum_mlps = dmgd.momentum_mlps

        X_train, y_train = task['train']
        criterion = nn.CrossEntropyLoss()

        for _ in range(trainer.inner_steps):
            pred = model(X_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final accuracy
        with torch.no_grad():
            pred_final = model(X_val).argmax(dim=1)
            acc_final = (pred_final == y_val).float().mean().item()

        print(f"Task {i+1}: Initial acc={acc_initial:.3f}, "
              f"Final acc={acc_final:.3f}, "
              f"Improvement={acc_final - acc_initial:.3f}")

        accuracies.append(acc_final)
        improvements.append(acc_final - acc_initial)

    print(f"\nAverage final accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"Average improvement: {np.mean(improvements):.3f} ± {np.std(improvements):.3f}")

    return meta_losses, accuracies


def comparison_example():
    """
    Compare meta-learned DMGD with standard optimizers on classification.
    """
    print("\n" + "="*70)
    print("Comparison: DMGD vs Standard Optimizers (Classification)")
    print("="*70)

    input_dim = 10

    # Meta-train DMGD
    print("\n1. Meta-Training DMGD...")
    def create_model():
        return create_binary_classifier(input_dim=input_dim)

    dummy_model = create_model()
    dmgd = DeepMomentumGD(
        dummy_model.parameters(),
        lr=0.01,
        momentum_hidden_dims=[32, 16],
        mlp_lr=0.001
    )

    trainer = MetaLearningTrainer(
        model_fn=create_model,
        dmgd_optimizer=dmgd,
        inner_steps=10,
        inner_lr=0.01,
        meta_lr=0.001
    )

    sampler = create_task_sampler(
        task_type='binary_classification',
        input_dim=input_dim,
        n_train=40,
        n_val=20
    )

    # Quick meta-training
    trainer.meta_train(sampler, num_episodes=60, criterion=nn.BCELoss(), verbose=False)
    print("   Meta-training complete!")

    # Compare on new tasks
    print("\n2. Testing on 8 New Tasks")
    print("-" * 70)

    def evaluate_optimizer(opt_class, opt_kwargs, task, num_steps=10):
        """Evaluate optimizer and return accuracy."""
        model = create_model()
        optimizer = opt_class(model.parameters(), **opt_kwargs)
        criterion = nn.BCELoss()

        X_train, y_train = task['train']
        X_val, y_val = task['val']

        # Train
        for _ in range(num_steps):
            pred = model(X_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate accuracy
        with torch.no_grad():
            pred = (model(X_val) > 0.5).float()
            accuracy = (pred == y_val).float().mean().item()

        return accuracy

    sgd_accs = []
    adam_accs = []
    dmgd_accs = []

    for i in range(8):
        task = sampler()

        sgd_acc = evaluate_optimizer(
            torch.optim.SGD,
            {'lr': 0.01, 'momentum': 0.9},
            task
        )

        adam_acc = evaluate_optimizer(
            torch.optim.Adam,
            {'lr': 0.01},
            task
        )

        # DMGD
        model = create_model()
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=0.01,
            momentum_hidden_dims=[32, 16]
        )
        optimizer.momentum_mlps = dmgd.momentum_mlps

        X_train, y_train = task['train']
        X_val, y_val = task['val']
        criterion = nn.BCELoss()

        for _ in range(trainer.inner_steps):
            pred = model(X_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = (model(X_val) > 0.5).float()
            dmgd_acc = (pred == y_val).float().mean().item()

        sgd_accs.append(sgd_acc)
        adam_accs.append(adam_acc)
        dmgd_accs.append(dmgd_acc)

        print(f"Task {i+1}: SGD={sgd_acc:.3f}, Adam={adam_acc:.3f}, DMGD={dmgd_acc:.3f}")

    print(f"\nResults over 8 tasks:")
    print(f"  SGD:   {np.mean(sgd_accs):.3f} ± {np.std(sgd_accs):.3f}")
    print(f"  Adam:  {np.mean(adam_accs):.3f} ± {np.std(adam_accs):.3f}")
    print(f"  DMGD:  {np.mean(dmgd_accs):.3f} ± {np.std(dmgd_accs):.3f}")

    return {'sgd': sgd_accs, 'adam': adam_accs, 'dmgd': dmgd_accs}


def main():
    """Run classification meta-learning examples."""
    print("\n" + "="*70)
    print("META-LEARNING FOR CLASSIFICATION TASKS")
    print("="*70)

    print("\nThis demonstrates meta-learning with DMGD on classification tasks.")
    print("The framework enables quick adaptation to new classification problems.\n")

    try:
        # Binary classification
        print("\n" + "#"*70)
        print("# Example 1: Binary Classification")
        print("#"*70)
        binary_meta_losses, binary_accs = binary_classification_example()

        # Multi-class classification
        print("\n" + "#"*70)
        print("# Example 2: Multi-Class Classification")
        print("#"*70)
        multi_meta_losses, multi_accs = multiclass_classification_example()

        # Comparison
        print("\n" + "#"*70)
        print("# Example 3: Comparison with Standard Optimizers")
        print("#"*70)
        comparison_results = comparison_example()

        # Summary
        print("\n" + "="*70)
        print("ALL CLASSIFICATION EXAMPLES COMPLETE")
        print("="*70)

        print("\nSummary:")
        print(f"  Binary classification avg accuracy: {np.mean(binary_accs):.3f}")
        print(f"  Multi-class classification avg accuracy: {np.mean(multi_accs):.3f}")

        print("\nKey Insights:")
        print("  1. Meta-learning enables quick adaptation on classification tasks")
        print("  2. Framework works for both binary and multi-class problems")
        print("  3. Can compare with standard optimizers for few-shot learning")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    main()
