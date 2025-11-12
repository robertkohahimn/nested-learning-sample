"""
Simple example demonstrating Deep Momentum GD (DMGD) optimizer.

This example trains a simple neural network on a synthetic regression task
and compares DMGD with standard optimizers (SGD, Adam).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from src.optimizers.dmgd import DeepMomentumGD


def create_model(input_dim=10, hidden_dim=64, output_dim=1):
    """Create a simple MLP for regression."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def generate_data(n_samples=100, input_dim=10):
    """Generate synthetic regression data."""
    X = torch.randn(n_samples, input_dim)
    # True function: y = sum(X * weights) + noise
    weights = torch.randn(input_dim, 1)
    y = X @ weights + 0.1 * torch.randn(n_samples, 1)
    return X, y


def train_model(model, optimizer, X, y, num_epochs=100):
    """Train model and return loss history."""
    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        pred = model(X)
        loss = ((pred - y) ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    return losses


def main():
    print("="*60)
    print("Deep Momentum GD (DMGD) Example")
    print("="*60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_data(n_samples=200, input_dim=10)
    X_test, y_test = generate_data(n_samples=50, input_dim=10)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Experiment configurations
    experiments = {
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.01),
        'DMGD': lambda params: DeepMomentumGD(
            params,
            lr=0.01,
            momentum_hidden_dims=[32, 16],
            momentum_activation='relu'
        )
    }

    results = {}

    for name, optimizer_fn in experiments.items():
        print(f"\n2. Training with {name} optimizer...")

        # Create fresh model
        model = create_model()

        # Create optimizer
        optimizer = optimizer_fn(model.parameters())

        # Train
        losses = train_model(model, optimizer, X_train, y_train, num_epochs=100)

        # Evaluate
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = ((test_pred - y_test) ** 2).mean().item()

        results[name] = {
            'train_loss': losses[-1],
            'test_loss': test_loss,
            'losses': losses
        }

        print(f"   Final train loss: {losses[-1]:.4f}")
        print(f"   Test loss: {test_loss:.4f}")

    # Summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"{'Optimizer':<15} {'Train Loss':<15} {'Test Loss':<15}")
    print("-"*60)
    for name, result in results.items():
        print(f"{name:<15} {result['train_loss']:<15.4f} {result['test_loss']:<15.4f}")

    print("\nNotes:")
    print("- DMGD uses a learnable MLP to compute momentum")
    print("- Without meta-learning, DMGD may not outperform standard optimizers")
    print("- The power of DMGD comes from meta-learning across tasks")
    print("- See meta_learning_example.py for meta-learning demonstration")


if __name__ == '__main__':
    main()
