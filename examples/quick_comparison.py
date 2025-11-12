"""
Quick Comparison: Nested Learning vs Standard Training

A fast comparison script that demonstrates the key benefits of Nested Learning
on a simple dataset. Runs in minutes on CPU.

Features demonstrated:
- Multi-frequency parameter updates
- Training efficiency
- Generalization performance
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NestedMLP
from src.optimizers import NestedOptimizerBuilder
from src.training import NestedTrainer


def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=5, noise=0.1):
    """
    Generate synthetic classification dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of input features
        n_classes: Number of output classes
        noise: Noise level

    Returns:
        X_train, y_train, X_test, y_test
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate class centers
    centers = np.random.randn(n_classes, n_features) * 2

    # Generate samples
    X = []
    y = []

    samples_per_class = n_samples // n_classes
    for class_idx in range(n_classes):
        # Generate samples around class center
        class_samples = centers[class_idx] + np.random.randn(samples_per_class, n_features) * noise
        X.append(class_samples)
        y.extend([class_idx] * samples_per_class)

    X = np.vstack(X)
    y = np.array(y)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_test, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create train and test dataloaders."""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_standard(train_loader, test_loader, n_features, n_classes, device, epochs=50):
    """Train with standard approach."""
    print("\n" + "="*60)
    print("STANDARD TRAINING")
    print("="*60)

    # Create standard model
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_nested_optimizer=False
    )

    # Train
    start_time = time.time()
    history = trainer.fit(train_loader, test_loader, epochs=epochs)
    train_time = time.time() - start_time

    # Evaluate
    test_acc, test_loss = evaluate(model, test_loader, device)

    print(f"\n✓ Training complete in {train_time:.2f}s")
    print(f"✓ Test accuracy: {test_acc:.4f}")

    return model, history, train_time, test_acc


def train_nested(train_loader, test_loader, n_features, n_classes, device, epochs=50):
    """Train with Nested Learning."""
    print("\n" + "="*60)
    print("NESTED LEARNING")
    print("="*60)

    # Create nested model
    model = NestedMLP(
        input_dim=n_features,
        hidden_dims=[64, 32],
        output_dim=n_classes,
        dropout=0.1,
        use_norm=True
    ).to(device)

    # Build multi-frequency optimizer
    builder = NestedOptimizerBuilder(model, num_levels=3)
    builder.auto_assign_params('uniform')
    optimizer = builder.build(
        optimizer_types=['adam', 'sgd', 'sgd'],
        learning_rates=[0.01, 0.05, 0.1],
        frequencies=[1, 5, 25]
    )

    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_nested_optimizer=True
    )

    # Train
    start_time = time.time()
    history = trainer.fit(train_loader, test_loader, epochs=epochs)
    train_time = time.time() - start_time

    # Evaluate
    test_acc, test_loss = evaluate(model, test_loader, device)

    print(f"\n✓ Training complete in {train_time:.2f}s")
    print(f"✓ Test accuracy: {test_acc:.4f}")

    return model, history, train_time, test_acc


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    accuracy = correct / total
    loss = total_loss / total

    return accuracy, loss


def plot_comparison(standard_history, nested_history, standard_time, nested_time,
                   standard_acc, nested_acc, save_path='quick_comparison.png'):
    """Plot training comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs_standard = range(1, len(standard_history['train_loss']) + 1)
    epochs_nested = range(1, len(nested_history['train_loss']) + 1)

    # Plot 1: Training loss
    axes[0].plot(epochs_standard, standard_history['train_loss'],
                'r--', label='Standard', linewidth=2)
    axes[0].plot(epochs_nested, nested_history['train_loss'],
                'b-', label='Nested Learning', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Test accuracy
    axes[1].plot(epochs_standard, standard_history['val_accuracy'],
                'r--', label='Standard', linewidth=2)
    axes[1].plot(epochs_nested, nested_history['val_accuracy'],
                'b-', label='Nested Learning', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Summary comparison
    x = np.arange(2)
    width = 0.35

    # Training time
    ax3_1 = axes[2]
    bars1 = ax3_1.bar(x - width/2, [standard_time, nested_time], width,
                     label='Training Time (s)', color=['red', 'blue'], alpha=0.7)
    ax3_1.set_ylabel('Training Time (s)', color='black')
    ax3_1.set_title('Summary')
    ax3_1.set_xticks(x)
    ax3_1.set_xticklabels(['Standard', 'Nested'])

    # Test accuracy (secondary axis)
    ax3_2 = ax3_1.twinx()
    bars2 = ax3_2.bar(x + width/2, [standard_acc, nested_acc], width,
                     label='Test Accuracy', color=['orange', 'cyan'], alpha=0.7)
    ax3_2.set_ylabel('Test Accuracy', color='black')
    ax3_2.set_ylim([0, 1.0])

    # Add legends
    ax3_1.legend(loc='upper left')
    ax3_2.legend(loc='upper right')

    # Add value labels
    for i, (t, a) in enumerate(zip([standard_time, nested_time], [standard_acc, nested_acc])):
        ax3_1.text(i - width/2, t + 0.5, f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
        ax3_2.text(i + width/2, a + 0.02, f'{a:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.close()


def main():
    """Main function."""
    print("\n" + "="*60)
    print("QUICK COMPARISON: Nested Learning vs Standard Training")
    print("="*60)

    # Configuration
    n_samples = 2000
    n_features = 20
    n_classes = 5
    epochs = 50
    batch_size = 32

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Generate data
    print("\nGenerating synthetic dataset...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes
    )

    print(f"✓ Train samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features: {n_features}")
    print(f"✓ Classes: {n_classes}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size
    )

    # Train standard model
    standard_model, standard_history, standard_time, standard_acc = train_standard(
        train_loader, test_loader, n_features, n_classes, device, epochs
    )

    # Train nested model
    nested_model, nested_history, nested_time, nested_acc = train_nested(
        train_loader, test_loader, n_features, n_classes, device, epochs
    )

    # Plot comparison
    plot_comparison(
        standard_history, nested_history,
        standard_time, nested_time,
        standard_acc, nested_acc
    )

    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    print("\nStandard Training:")
    print(f"  Training time: {standard_time:.2f}s")
    print(f"  Test accuracy: {standard_acc:.4f}")
    print(f"  Final train loss: {standard_history['train_loss'][-1]:.4f}")

    print("\nNested Learning:")
    print(f"  Training time: {nested_time:.2f}s")
    print(f"  Test accuracy: {nested_acc:.4f}")
    print(f"  Final train loss: {nested_history['train_loss'][-1]:.4f}")

    print("\nComparison:")
    time_diff = nested_time - standard_time
    acc_diff = (nested_acc - standard_acc) * 100

    print(f"  Time difference: {time_diff:+.2f}s ({(time_diff/standard_time)*100:+.1f}%)")
    print(f"  Accuracy difference: {acc_diff:+.2f}%")

    if nested_acc > standard_acc:
        print("\n✓ Nested Learning achieved better accuracy!")
    elif abs(nested_acc - standard_acc) < 0.01:
        print("\n= Both approaches achieved similar accuracy")
    else:
        print("\n✗ Standard training achieved better accuracy")

    print("="*60)


if __name__ == '__main__':
    main()
