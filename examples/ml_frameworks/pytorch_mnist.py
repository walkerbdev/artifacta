"""
PyTorch MNIST Training Example with Artifacta
==============================================

This example demonstrates Artifacta's logging capabilities for PyTorch training:

1. **Automatic checkpoint logging** via autolog() - tracks model checkpoints automatically
2. **Training metrics** via Series - tracks loss and accuracy over epochs
3. **Confusion matrix** via Matrix - visualizes classification performance
4. **Model artifact logging** - saves trained model with automatic metadata extraction
5. **Configuration tracking** - logs hyperparameters automatically

Key Artifacta Features Demonstrated:
- init() - Initialize experiment run with config
- autolog() - Enable automatic checkpoint logging
- Series - Log time-series metrics (loss, accuracy per epoch)
- Matrix - Log 2D data (confusion matrix)
- run.log_output() - Save model artifacts with metadata

Requirements:
    pip install artifacta torch torchvision scikit-learn

Usage:
    python examples/pytorch_mnist.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from artifacta import Matrix, Series, autolog, init, log


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification.

    Architecture:
    - Conv2D (1 -> 32 filters, 3x3 kernel)
    - Conv2D (32 -> 64 filters, 3x3 kernel)
    - Max Pool (2x2)
    - Dropout (0.25)
    - Fully Connected (9216 -> 128)
    - Dropout (0.5)
    - Fully Connected (128 -> 10 output classes)
    """

    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)  # 64 * 12 * 12 = 9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv layer 1: 28x28 -> 26x26 -> apply ReLU
        x = f.relu(self.conv1(x))
        # Conv layer 2: 26x26 -> 24x24 -> apply ReLU
        x = f.relu(self.conv2(x))
        # Max pool: 24x24 -> 12x12
        x = self.pool(x)
        # Dropout for regularization
        x = self.dropout1(x)
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        # FC layer 1: 9216 -> 128
        x = f.relu(self.fc1(x))
        x = self.dropout2(x)
        # FC layer 2: 128 -> 10 (output classes)
        x = self.fc2(x)
        # Log softmax for numerical stability with NLLLoss
        return f.log_softmax(x, dim=1)


def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train model for one epoch.

    Args:
        model: PyTorch model to train
        device: Device to train on (cpu or cuda)
        train_loader: DataLoader for training data
        optimizer: Optimizer for gradient descent
        epoch: Current epoch number (for logging)

    Returns:
        Average training loss for this epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = f.nll_loss(output, target)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, device, test_loader):
    """Evaluate model on test set.

    Args:
        model: PyTorch model to evaluate
        device: Device to evaluate on
        test_loader: DataLoader for test data

    Returns:
        Tuple of (avg_loss, accuracy, all_predictions, all_targets)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()

            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Store for confusion matrix
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_preds, all_targets


def main():
    """Main training function."""
    print("=" * 70)
    print("Artifacta PyTorch MNIST Example")
    print("=" * 70)

    # =================================================================
    # 1. Define hyperparameter search space (grid search)
    # =================================================================
    from itertools import product

    # Define parameter grid - typical grid search approach
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.05],
        "optimizer": ["Adam", "SGD"],
        "batch_size": [128],
        "epochs": [5],
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    configs = [dict(zip(keys, v)) for v in product(*values)]

    # Add metadata to each config
    for config in configs:
        config.update({
            "model": "SimpleCNN",
            "dataset": "MNIST",
        })

    print(f"\nGrid search: {len(configs)} configurations")
    print("  Parameter grid:")
    for key, values in param_grid.items():
        print(f"    {key}: {values}")

    # =================================================================
    # 2. Run experiments with different configurations
    # =================================================================
    for idx, config in enumerate(configs, 1):
        # Generate run name from config
        run_name = f"lr{config['learning_rate']}-{config['optimizer'].lower()}-bs{config['batch_size']}"

        print(f"\n{'=' * 70}")
        print(f"Run {idx}/{len(configs)}: {run_name}")
        print(f"{'=' * 70}")

        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Initialize Artifacta run with configuration
        run = init(
            project="mnist-classification",
            name=run_name,
            config=config,
        )
        print("\nArtifacta run initialized")

        # =================================================================
        # 3. Enable autolog for automatic checkpoint tracking
        #    This will log model checkpoints automatically during training
        # =================================================================
        autolog(framework="pytorch")

        # =================================================================
        # 4. Setup device (GPU if available, otherwise CPU)
        # =================================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        # =================================================================
        # 5. Load MNIST dataset
        #    Downloads to ./data directory on first run
        # =================================================================
        print("\nLoading MNIST dataset...")

        # Transform: Convert to tensor and normalize to [0, 1]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

        # Download and load training data
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

        # Download and load test data
        test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        # =================================================================
        # 6. Create model, optimizer, and move to device
        # =================================================================
        print("\nCreating model...")
        model = SimpleCNN().to(device)

        # Create optimizer based on config
        if config["optimizer"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        else:  # SGD
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # =================================================================
        # 7. Training loop
        # =================================================================
        print(f"\nTraining for {config['epochs']} epochs...")

        # Track metrics across epochs
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(1, config["epochs"] + 1):
            print(f"\nEpoch {epoch}/{config['epochs']}")
            print("-" * 70)

            # Train for one epoch
            train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)

            # Evaluate on test set
            val_loss, val_acc, _, _ = evaluate(model, device, test_loader)

            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # =================================================================
        # 8. Log training metrics as Series (time-series data)
        #    This creates interactive plots in the Artifacta UI
        # =================================================================
        print("\nLogging training metrics...")

        # Log loss curves
        log(
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

        # Log accuracy curves
        log(
            "accuracy_curves",
            Series(
                index="epoch",
                fields={
                    "train_accuracy": train_accuracies,
                    "val_accuracy": val_accuracies,
                },
                index_values=list(range(1, config["epochs"] + 1)),
            ),
        )

        # =================================================================
        # 9. Generate and log confusion matrix
        #    Shows which digits are confused with each other
        # =================================================================
        print("\nGenerating confusion matrix...")

        # Get predictions on full test set
        _, _, all_preds, all_targets = evaluate(model, device, test_loader)

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Log as Matrix primitive
        digit_labels = [str(i) for i in range(10)]
        log(
            "confusion_matrix",
            Matrix(
                rows=digit_labels,
                cols=digit_labels,
                values=cm.tolist(),
                metadata={"type": "confusion_matrix", "normalize": False},
            ),
        )

        print("  Confusion matrix shape:", cm.shape)

        # =================================================================
        # 10. Save and log the trained model
        #     Artifacta automatically extracts model metadata
        # =================================================================
        print("\nSaving model...")

        # Save model checkpoint
        model_path = "mnist_cnn.pt"
        torch.save(
            {
                "epoch": config["epochs"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1],
                "val_accuracy": val_accuracies[-1],
            },
            model_path,
        )

        # Log model as output artifact
        # Artifacta will automatically extract metadata (file size, format, etc.)
        run.log_output(
            model_path,
            name="trained_model",
            metadata={
                "framework": "pytorch",
                "model_type": "CNN",
                "final_val_accuracy": val_accuracies[-1],
                "final_val_loss": val_losses[-1],
            },
        )

        # =================================================================
        # 11. Final summary
        # =================================================================
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print("Final Results:")
        print(f"  Train Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"  Val Accuracy:   {val_accuracies[-1]:.2f}%")
        print(f"  Val Loss:       {val_losses[-1]:.4f}")
        print("\nAll metrics and artifacts logged to Artifacta")
        print("  View your results in the Artifacta UI!")
        print("=" * 70)

        # Finish the run
        run.finish()

    # All experiments complete
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
