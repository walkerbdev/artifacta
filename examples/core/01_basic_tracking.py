"""
Basic Experiment Tracking with Artifacta
=========================================

This minimal example demonstrates the core Artifacta workflow:
1. Initialize a run with init()
2. Log metrics using Series primitive
3. Run automatically finishes when script exits

Perfect starting point for understanding Artifacta basics.

Requirements:
    pip install artifacta numpy

Usage:
    python examples/core/01_basic_tracking.py
"""

import time

import numpy as np

from artifacta import Series, init


def simulate_training(epochs=5):
    """Simulate a simple training loop.

    Returns training and validation loss for each epoch.
    """
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # Simulate decreasing loss with some noise
        train_loss = 1.0 / epoch + np.random.normal(0, 0.05)
        val_loss = 1.0 / epoch + np.random.normal(0, 0.08)

        train_losses.append(max(0, train_loss))
        val_losses.append(max(0, val_loss))

        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        time.sleep(0.2)  # Simulate training time

    return train_losses, val_losses


def main():
    print("=" * 60)
    print("Artifacta Basic Tracking Example")
    print("=" * 60)

    # =================================================================
    # 1. Initialize Artifacta run
    #    This creates a new experiment with configuration
    # =================================================================
    config = {
        "model": "SimpleNN",
        "learning_rate": 0.01,
        "epochs": 5,
    }

    run = init(
        project="getting-started",
        name="basic-example",
        config=config,
    )

    print("\nArtifacta run initialized")
    print("  Project: getting-started")
    print("  Run: basic-example")

    # =================================================================
    # 2. Run your experiment (training simulation)
    # =================================================================
    print("\nStarting training simulation...")
    train_losses, val_losses = simulate_training(epochs=config["epochs"])

    # =================================================================
    # 3. Log results using Series primitive
    #    Series is used for ordered data over a single dimension
    # =================================================================
    print("\nLogging training metrics...")

    run.log(
        "loss_curves",
        Series(
            index="epoch",
            fields={
                "train_loss": train_losses,
                "val_loss": val_losses,
            },
            index_values=list(range(1, config["epochs"] + 1)),
        ),
    )

    # =================================================================
    # 4. Run automatically finishes when script exits
    #    No need to call run.finish() manually!
    # =================================================================
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print("Metrics logged to Artifacta")
    print("Run will auto-finish when script exits")
    print("  View results in the Artifacta UI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
