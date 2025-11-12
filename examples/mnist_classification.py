"""
MNIST Classification with Nested Learning

This example demonstrates multi-frequency training on MNIST using:
- NestedMLP with frequency-aware layers
- NestedOptimizer with 3 frequency levels
- NestedTrainer with callbacks
- Comparison with standard training

Results show how multi-frequency updates can improve training efficiency
and generalization on image classification tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NestedMLP
from src.optimizers import NestedOptimizerBuilder
from src.training import NestedTrainer, EarlyStopping, CheckpointCallback, LRSchedulerCallback


def prepare_data(batch_size=128, val_split=0.1, data_dir='./data'):
    """
    Prepare MNIST dataset with train/val/test splits.

    Args:
        batch_size: Batch size for data loaders
        val_split: Fraction of training data for validation
        data_dir: Directory to store dataset

    Returns:
        train_loader, val_loader, test_loader
    """
    print("Preparing MNIST dataset...")

    # Transform: Normalize to [-1, 1] and flatten
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])

    # Download datasets
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

    # Split train into train/val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"✓ Train: {train_size} samples")
    print(f"✓ Val: {val_size} samples")
    print(f"✓ Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def create_nested_model_and_optimizer(device='cpu'):
    """
    Create NestedMLP model with multi-frequency optimizer.

    Multi-frequency strategy:
    - Fast (every step): Output layer, biases - Quick task adaptation
    - Medium (every 10 steps): Hidden layers - General features
    - Slow (every 100 steps): Input layer - Stable feature extraction

    Args:
        device: Device to place model on

    Returns:
        model, optimizer
    """
    print("\nCreating Nested Learning model...")

    # Create NestedMLP (784 -> 256 -> 128 -> 10)
    model = NestedMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        dropout=0.2,
        use_norm=True
    ).to(device)

    # Build multi-frequency optimizer
    builder = NestedOptimizerBuilder(model, num_levels=3)

    # Auto-assign parameters uniformly across levels
    builder.auto_assign_params('uniform')

    # Set optimizer for each level
    # Fast: Adam with higher LR for quick adaptation
    # Medium/Slow: SGD with momentum for stable learning
    optimizer = builder.build(
        optimizer_types=['adam', 'sgd', 'sgd'],
        learning_rates=[0.001, 0.01, 0.05],
        frequencies=[1, 10, 100],
        optimizer_kwargs=[
            {},  # Adam defaults
            {'momentum': 0.9},  # SGD with momentum
            {'momentum': 0.9}
        ]
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model architecture: 784 -> 256 -> 128 -> 10")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Optimizer: 3-level multi-frequency")
    print(f"  - Level 0 (Fast, f=1): Adam, lr=0.001")
    print(f"  - Level 1 (Medium, f=10): SGD, lr=0.01")
    print(f"  - Level 2 (Slow, f=100): SGD, lr=0.05")

    return model, optimizer


def create_standard_model_and_optimizer(device='cpu'):
    """
    Create standard MLP for comparison.

    Args:
        device: Device to place model on

    Returns:
        model, optimizer
    """
    print("\nCreating standard model...")

    # Create standard MLP with same architecture
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    ).to(device)

    # Standard Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model architecture: 784 -> 256 -> 128 -> 10")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Optimizer: Adam, lr=0.001")

    return model, optimizer


def train_nested_learning(train_loader, val_loader, device='cpu', epochs=20):
    """
    Train model using Nested Learning approach.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Maximum number of epochs

    Returns:
        model, history
    """
    print("\n" + "="*60)
    print("Training with NESTED LEARNING")
    print("="*60)

    # Create model and optimizer
    model, optimizer = create_nested_model_and_optimizer(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.optimizers[0],  # Apply to fast level
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Create trainer with callbacks
    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_nested_optimizer=True
    )

    # Add callbacks
    trainer.callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=True
        ),
        CheckpointCallback(
            filepath='best_nested_mnist.pt',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=True
        ),
        LRSchedulerCallback(scheduler, metric='val_loss')
    ]

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )

    print(f"\n✓ Training complete!")
    print(f"✓ Best val accuracy: {max(history['val_accuracy']):.4f}")
    print(f"✓ Best val loss: {min(history['val_loss']):.4f}")

    return model, history


def train_standard(train_loader, val_loader, device='cpu', epochs=20):
    """
    Train model using standard approach for comparison.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Maximum number of epochs

    Returns:
        model, history
    """
    print("\n" + "="*60)
    print("Training with STANDARD APPROACH")
    print("="*60)

    # Create model and optimizer
    model, optimizer = create_standard_model_and_optimizer(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Create trainer with callbacks
    trainer = NestedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_nested_optimizer=False
    )

    # Add callbacks
    trainer.callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=True
        ),
        CheckpointCallback(
            filepath='best_standard_mnist.pt',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=True
        ),
        LRSchedulerCallback(scheduler, metric='val_loss')
    ]

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )

    print(f"\n✓ Training complete!")
    print(f"✓ Best val accuracy: {max(history['val_accuracy']):.4f}")
    print(f"✓ Best val loss: {min(history['val_loss']):.4f}")

    return model, history


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        test_accuracy, test_loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    test_accuracy = correct / total
    test_loss = total_loss / total

    return test_accuracy, test_loss


def plot_comparison(nested_history, standard_history, save_path='mnist_comparison.png'):
    """
    Plot training comparison between nested and standard approaches.

    Args:
        nested_history: Training history from nested learning
        standard_history: Training history from standard training
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs_nested = range(1, len(nested_history['train_loss']) + 1)
    epochs_standard = range(1, len(standard_history['train_loss']) + 1)

    # Training loss
    axes[0, 0].plot(epochs_nested, nested_history['train_loss'],
                    'b-', label='Nested Learning', linewidth=2)
    axes[0, 0].plot(epochs_standard, standard_history['train_loss'],
                    'r--', label='Standard', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation loss
    axes[0, 1].plot(epochs_nested, nested_history['val_loss'],
                    'b-', label='Nested Learning', linewidth=2)
    axes[0, 1].plot(epochs_standard, standard_history['val_loss'],
                    'r--', label='Standard', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training accuracy
    axes[1, 0].plot(epochs_nested, nested_history['train_accuracy'],
                    'b-', label='Nested Learning', linewidth=2)
    axes[1, 0].plot(epochs_standard, standard_history['train_accuracy'],
                    'r--', label='Standard', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation accuracy
    axes[1, 1].plot(epochs_nested, nested_history['val_accuracy'],
                    'b-', label='Nested Learning', linewidth=2)
    axes[1, 1].plot(epochs_standard, standard_history['val_accuracy'],
                    'r--', label='Standard', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")
    plt.close()


def print_summary(nested_results, standard_results):
    """
    Print summary comparison of results.

    Args:
        nested_results: (test_acc, test_loss, history) for nested
        standard_results: (test_acc, test_loss, history) for standard
    """
    nested_acc, nested_loss, nested_history = nested_results
    standard_acc, standard_loss, standard_history = standard_results

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print("\nNested Learning:")
    print(f"  Test Accuracy:  {nested_acc:.4f}")
    print(f"  Test Loss:      {nested_loss:.4f}")
    print(f"  Best Val Acc:   {max(nested_history['val_accuracy']):.4f}")
    print(f"  Epochs trained: {len(nested_history['train_loss'])}")

    print("\nStandard Training:")
    print(f"  Test Accuracy:  {standard_acc:.4f}")
    print(f"  Test Loss:      {standard_loss:.4f}")
    print(f"  Best Val Acc:   {max(standard_history['val_accuracy']):.4f}")
    print(f"  Epochs trained: {len(standard_history['train_loss'])}")

    print("\nImprovement:")
    acc_improvement = (nested_acc - standard_acc) * 100
    print(f"  Accuracy: {acc_improvement:+.2f}%")

    if nested_acc > standard_acc:
        print("  ✓ Nested Learning achieved better accuracy!")
    elif nested_acc < standard_acc:
        print("  ✗ Standard training achieved better accuracy")
    else:
        print("  = Both achieved same accuracy")

    print("="*60)


def main():
    """Main function to run MNIST classification experiment."""

    print("\n" + "="*60)
    print("MNIST Classification with Nested Learning")
    print("="*60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        batch_size=128,
        val_split=0.1
    )

    # Train with Nested Learning
    nested_model, nested_history = train_nested_learning(
        train_loader, val_loader, device, epochs=20
    )

    # Evaluate nested model
    print("\nEvaluating Nested Learning model on test set...")
    nested_acc, nested_loss = evaluate_model(nested_model, test_loader, device)
    print(f"✓ Test Accuracy: {nested_acc:.4f}")
    print(f"✓ Test Loss: {nested_loss:.4f}")

    # Train with standard approach
    standard_model, standard_history = train_standard(
        train_loader, val_loader, device, epochs=20
    )

    # Evaluate standard model
    print("\nEvaluating standard model on test set...")
    standard_acc, standard_loss = evaluate_model(standard_model, test_loader, device)
    print(f"✓ Test Accuracy: {standard_acc:.4f}")
    print(f"✓ Test Loss: {standard_loss:.4f}")

    # Plot comparison
    plot_comparison(nested_history, standard_history)

    # Print summary
    print_summary(
        (nested_acc, nested_loss, nested_history),
        (standard_acc, standard_loss, standard_history)
    )


if __name__ == '__main__':
    main()
