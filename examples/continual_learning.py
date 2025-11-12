"""
Continual Learning: Preventing Catastrophic Forgetting

This example demonstrates how Nested Learning with CMS prevents catastrophic
forgetting in continual learning scenarios.

Experiment:
1. Train on MNIST digits 0-4 (Task 1)
2. Train on MNIST digits 5-9 (Task 2)
3. Evaluate on both tasks to measure forgetting

Comparison:
- Standard MLP: Shows catastrophic forgetting
- NestedMLP + CMS: Retains knowledge from Task 1 while learning Task 2
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NestedMLP, HopeModel
from src.optimizers import NestedOptimizerBuilder
from src.training import NestedTrainer, CheckpointCallback


def create_task_datasets(data_dir='./data'):
    """
    Create two tasks from MNIST:
    - Task 1: Digits 0-4
    - Task 2: Digits 5-9

    Args:
        data_dir: Directory to store dataset

    Returns:
        task1_train, task1_test, task2_train, task2_test
    """
    print("Creating continual learning tasks...")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Download full MNIST
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split into Task 1 (digits 0-4) and Task 2 (digits 5-9)
    def get_task_indices(dataset, start_digit, end_digit):
        indices = []
        for idx, (_, label) in enumerate(dataset):
            if start_digit <= label <= end_digit:
                indices.append(idx)
        return indices

    # Task 1: Digits 0-4
    task1_train_indices = get_task_indices(train_dataset, 0, 4)
    task1_test_indices = get_task_indices(test_dataset, 0, 4)

    task1_train = Subset(train_dataset, task1_train_indices)
    task1_test = Subset(test_dataset, task1_test_indices)

    # Task 2: Digits 5-9
    task2_train_indices = get_task_indices(train_dataset, 5, 9)
    task2_test_indices = get_task_indices(test_dataset, 5, 9)

    task2_train = Subset(train_dataset, task2_train_indices)
    task2_test = Subset(test_dataset, task2_test_indices)

    print(f"✓ Task 1 (digits 0-4): {len(task1_train)} train, {len(task1_test)} test")
    print(f"✓ Task 2 (digits 5-9): {len(task2_train)} train, {len(task2_test)} test")

    return task1_train, task1_test, task2_train, task2_test


def create_standard_model(device='cpu'):
    """Create standard MLP without memory."""
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer


def create_nested_model(device='cpu'):
    """Create NestedMLP with multi-frequency updates."""
    model = NestedMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        dropout=0.2,
        use_norm=True
    ).to(device)

    # Build multi-frequency optimizer
    builder = NestedOptimizerBuilder(model, num_levels=3)
    builder.auto_assign_params('uniform')
    optimizer = builder.build(
        optimizer_types=['adam', 'sgd', 'sgd'],
        learning_rates=[0.001, 0.01, 0.05],
        frequencies=[1, 10, 100]
    )

    return model, optimizer


def train_on_task(model, optimizer, task_loader, task_name, device='cpu',
                  epochs=10, use_nested=False):
    """
    Train model on a specific task.

    Args:
        model: Model to train
        optimizer: Optimizer
        task_loader: DataLoader for task
        task_name: Name of task (for logging)
        device: Device to train on
        epochs: Number of epochs
        use_nested: Whether using NestedOptimizer

    Returns:
        training history
    """
    print(f"\n{'='*60}")
    print(f"Training on {task_name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()

    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_nested_optimizer=use_nested
    )

    history = trainer.fit(
        train_loader=task_loader,
        val_loader=None,
        epochs=epochs
    )

    print(f"✓ {task_name} training complete")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final train accuracy: {history['train_accuracy'][-1]:.4f}")

    return history


def evaluate_on_task(model, task_loader, task_name, device='cpu'):
    """
    Evaluate model on a specific task.

    Args:
        model: Model to evaluate
        task_loader: DataLoader for task
        task_name: Name of task (for logging)
        device: Device to evaluate on

    Returns:
        accuracy, loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in task_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    accuracy = correct / total
    loss = total_loss / total

    print(f"{task_name}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")

    return accuracy, loss


def run_continual_learning_experiment(model_type='standard', device='cpu',
                                      epochs_per_task=10):
    """
    Run continual learning experiment.

    Args:
        model_type: 'standard' or 'nested'
        device: Device to run on
        epochs_per_task: Epochs to train on each task

    Returns:
        results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Continual Learning with {model_type.upper()} model")
    print(f"{'='*60}")

    # Create datasets
    task1_train, task1_test, task2_train, task2_test = create_task_datasets()

    # Create data loaders
    task1_train_loader = DataLoader(task1_train, batch_size=128, shuffle=True)
    task1_test_loader = DataLoader(task1_test, batch_size=128, shuffle=False)
    task2_train_loader = DataLoader(task2_train, batch_size=128, shuffle=True)
    task2_test_loader = DataLoader(task2_test, batch_size=128, shuffle=False)

    # Create model
    if model_type == 'standard':
        model, optimizer = create_standard_model(device)
        use_nested = False
    else:
        model, optimizer = create_nested_model(device)
        use_nested = True

    # Phase 1: Train on Task 1
    print("\n--- Phase 1: Learning Task 1 (digits 0-4) ---")
    task1_history = train_on_task(
        model, optimizer, task1_train_loader,
        "Task 1", device, epochs_per_task, use_nested
    )

    # Evaluate on Task 1 after training
    print("\nEvaluation after Task 1 training:")
    task1_acc_after_task1, task1_loss_after_task1 = evaluate_on_task(
        model, task1_test_loader, "Task 1", device
    )

    # Phase 2: Train on Task 2
    print("\n--- Phase 2: Learning Task 2 (digits 5-9) ---")
    task2_history = train_on_task(
        model, optimizer, task2_train_loader,
        "Task 2", device, epochs_per_task, use_nested
    )

    # Evaluate on both tasks after training on Task 2
    print("\nEvaluation after Task 2 training:")
    task1_acc_after_task2, task1_loss_after_task2 = evaluate_on_task(
        model, task1_test_loader, "Task 1", device
    )
    task2_acc_after_task2, task2_loss_after_task2 = evaluate_on_task(
        model, task2_test_loader, "Task 2", device
    )

    # Compute forgetting
    forgetting = task1_acc_after_task1 - task1_acc_after_task2
    retention = task1_acc_after_task2 / task1_acc_after_task1

    print(f"\n{'='*60}")
    print(f"Catastrophic Forgetting Analysis")
    print(f"{'='*60}")
    print(f"Task 1 accuracy after Task 1: {task1_acc_after_task1:.4f}")
    print(f"Task 1 accuracy after Task 2: {task1_acc_after_task2:.4f}")
    print(f"Forgetting (drop in accuracy): {forgetting:.4f} ({forgetting*100:.2f}%)")
    print(f"Retention rate: {retention:.4f} ({retention*100:.2f}%)")

    return {
        'model_type': model_type,
        'task1_acc_after_task1': task1_acc_after_task1,
        'task1_acc_after_task2': task1_acc_after_task2,
        'task2_acc_after_task2': task2_acc_after_task2,
        'forgetting': forgetting,
        'retention': retention,
        'task1_history': task1_history,
        'task2_history': task2_history
    }


def plot_continual_learning_results(standard_results, nested_results,
                                    save_path='continual_learning.png'):
    """
    Plot continual learning comparison.

    Args:
        standard_results: Results from standard model
        nested_results: Results from nested model
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy comparison
    ax = axes[0]
    x = np.arange(3)
    width = 0.35

    standard_accs = [
        standard_results['task1_acc_after_task1'],
        standard_results['task1_acc_after_task2'],
        standard_results['task2_acc_after_task2']
    ]

    nested_accs = [
        nested_results['task1_acc_after_task1'],
        nested_results['task1_acc_after_task2'],
        nested_results['task2_acc_after_task2']
    ]

    ax.bar(x - width/2, standard_accs, width, label='Standard', color='red', alpha=0.7)
    ax.bar(x + width/2, nested_accs, width, label='Nested Learning', color='blue', alpha=0.7)

    ax.set_xlabel('Evaluation Point')
    ax.set_ylabel('Accuracy')
    ax.set_title('Continual Learning Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Task 1\nafter Task 1', 'Task 1\nafter Task 2', 'Task 2\nafter Task 2'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])

    # Add value labels on bars
    for i, (s, n) in enumerate(zip(standard_accs, nested_accs)):
        ax.text(i - width/2, s + 0.02, f'{s:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, n + 0.02, f'{n:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Forgetting comparison
    ax = axes[1]
    forgetting_data = [
        ['Standard', standard_results['forgetting'], standard_results['retention']],
        ['Nested', nested_results['forgetting'], nested_results['retention']]
    ]

    x = np.arange(2)
    forgetting_vals = [standard_results['forgetting'], nested_results['forgetting']]
    retention_vals = [standard_results['retention'], nested_results['retention']]

    ax2 = ax.twinx()

    bars1 = ax.bar(x - 0.2, forgetting_vals, 0.4, label='Forgetting (lower is better)',
                   color=['red', 'blue'], alpha=0.7)
    bars2 = ax2.bar(x + 0.2, retention_vals, 0.4, label='Retention (higher is better)',
                    color=['orange', 'cyan'], alpha=0.7)

    ax.set_xlabel('Model')
    ax.set_ylabel('Forgetting (Accuracy Drop)', color='black')
    ax2.set_ylabel('Retention Rate', color='black')
    ax.set_title('Catastrophic Forgetting Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard', 'Nested Learning'])
    ax.grid(True, alpha=0.3, axis='y')

    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Add value labels
    for i, (f, r) in enumerate(zip(forgetting_vals, retention_vals)):
        ax.text(i - 0.2, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + 0.2, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results plot saved to: {save_path}")
    plt.close()


def main():
    """Main function to run continual learning experiment."""

    print("\n" + "="*60)
    print("Continual Learning: Preventing Catastrophic Forgetting")
    print("="*60)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Run experiments
    print("\n" + "="*60)
    print("Running Experiments")
    print("="*60)

    # Experiment 1: Standard model (shows catastrophic forgetting)
    standard_results = run_continual_learning_experiment(
        model_type='standard',
        device=device,
        epochs_per_task=10
    )

    # Experiment 2: Nested model (prevents forgetting)
    nested_results = run_continual_learning_experiment(
        model_type='nested',
        device=device,
        epochs_per_task=10
    )

    # Plot results
    plot_continual_learning_results(standard_results, nested_results)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\nStandard Model:")
    print(f"  Forgetting: {standard_results['forgetting']:.4f} ({standard_results['forgetting']*100:.2f}%)")
    print(f"  Retention:  {standard_results['retention']:.4f} ({standard_results['retention']*100:.2f}%)")

    print("\nNested Learning Model:")
    print(f"  Forgetting: {nested_results['forgetting']:.4f} ({nested_results['forgetting']*100:.2f}%)")
    print(f"  Retention:  {nested_results['retention']:.4f} ({nested_results['retention']*100:.2f}%)")

    improvement = (nested_results['retention'] - standard_results['retention']) * 100
    print(f"\nImprovement in retention: {improvement:+.2f}%")

    if nested_results['forgetting'] < standard_results['forgetting']:
        print("✓ Nested Learning shows less catastrophic forgetting!")
    else:
        print("✗ Standard model performed better")

    print("="*60)


if __name__ == '__main__':
    main()
