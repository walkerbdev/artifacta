"""
Artifact creation helpers for e2e tests
Functions for creating fake model checkpoints, metadata, predictions, and logs
"""

import csv
import json
import os
import pickle
import tempfile

import numpy as np


def create_fake_model_checkpoint(model_type="pytorch"):
    """
    Create a fake model checkpoint file for testing artifacts.

    Args:
        model_type: Type of model ("pytorch", "keras", "sklearn", "onnx")

    Returns:
        Path to the created file
    """
    temp_dir = tempfile.gettempdir()

    if model_type == "pytorch":
        # Fake PyTorch checkpoint
        filepath = os.path.join(temp_dir, "model_checkpoint.pt")
        fake_state = {
            "epoch": 10,
            "model_state_dict": {"layer1.weight": [[0.1, 0.2], [0.3, 0.4]]},
            "optimizer_state_dict": {"lr": 0.001},
            "loss": 0.28,
        }
        with open(filepath, "wb") as f:
            pickle.dump(fake_state, f)

    elif model_type == "keras":
        # Fake Keras H5 model
        filepath = os.path.join(temp_dir, "model.h5")
        with open(filepath, "wb") as f:
            f.write(b"HDF5_FAKE_MODEL_DATA" + b"\x00" * 1000)

    elif model_type == "sklearn":
        # Fake sklearn pickle
        filepath = os.path.join(temp_dir, "model.pkl")
        fake_model = {"type": "RandomForest", "n_estimators": 100, "max_depth": 10}
        with open(filepath, "wb") as f:
            pickle.dump(fake_model, f)

    elif model_type == "onnx":
        # Fake ONNX model
        filepath = os.path.join(temp_dir, "model.onnx")
        with open(filepath, "wb") as f:
            f.write(b"ONNX_MODEL_FAKE" + b"\x00" * 500)

    return filepath


def create_model_metadata(config, metrics):
    """
    Create model metadata JSON file.

    Args:
        config: Training configuration dict
        metrics: Final metrics dict

    Returns:
        Path to JSON file
    """
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, "model_metadata.json")

    metadata = {
        "model_architecture": {
            "type": "ResNet50",
            "layers": 50,
            "parameters": 25_600_000,
        },
        "training_config": config,
        "final_metrics": metrics,
        "framework": "pytorch",
        "framework_version": "2.0.1",
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    return filepath


def create_predictions_csv(n_samples=100):
    """
    Create fake predictions CSV for testing.

    Args:
        n_samples: Number of prediction samples

    Returns:
        Path to CSV file
    """
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, "predictions.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_label", "predicted_label", "confidence"])

        for i in range(n_samples):
            true_label = np.random.choice(["cat", "dog", "bird", "fish"])
            # Mostly correct predictions
            predicted_label = (
                true_label
                if np.random.random() > 0.1
                else np.random.choice(["cat", "dog", "bird", "fish"])
            )
            confidence = np.random.uniform(0.7, 0.99)
            writer.writerow([f"sample_{i}", true_label, predicted_label, f"{confidence:.3f}"])

    return filepath


def create_training_log():
    """
    Create fake training log text file.

    Returns:
        Path to log file
    """
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, "training.log")

    log_content = """[2025-01-10 12:00:00] INFO: Starting training...
[2025-01-10 12:00:05] INFO: Epoch 1/10 - loss: 2.5 - accuracy: 0.30 - val_loss: 2.6 - val_accuracy: 0.28
[2025-01-10 12:00:10] INFO: Epoch 2/10 - loss: 1.8 - accuracy: 0.50 - val_loss: 1.9 - val_accuracy: 0.48
[2025-01-10 12:00:15] INFO: Epoch 3/10 - loss: 1.2 - accuracy: 0.65 - val_loss: 1.4 - val_accuracy: 0.62
[2025-01-10 12:00:20] INFO: Epoch 4/10 - loss: 0.9 - accuracy: 0.75 - val_loss: 1.1 - val_accuracy: 0.71
[2025-01-10 12:00:25] INFO: Epoch 5/10 - loss: 0.7 - accuracy: 0.82 - val_loss: 0.9 - val_accuracy: 0.78
[2025-01-10 12:00:30] INFO: Saving checkpoint...
[2025-01-10 12:00:35] INFO: Epoch 6/10 - loss: 0.5 - accuracy: 0.87 - val_loss: 0.75 - val_accuracy: 0.83
[2025-01-10 12:00:40] INFO: Epoch 7/10 - loss: 0.4 - accuracy: 0.90 - val_loss: 0.65 - val_accuracy: 0.86
[2025-01-10 12:00:45] INFO: Epoch 8/10 - loss: 0.35 - accuracy: 0.92 - val_loss: 0.6 - val_accuracy: 0.88
[2025-01-10 12:00:50] INFO: Epoch 9/10 - loss: 0.3 - accuracy: 0.94 - val_loss: 0.55 - val_accuracy: 0.89
[2025-01-10 12:00:55] INFO: Epoch 10/10 - loss: 0.28 - accuracy: 0.95 - val_loss: 0.52 - val_accuracy: 0.90
[2025-01-10 12:01:00] INFO: Training completed!
[2025-01-10 12:01:05] INFO: Final validation accuracy: 0.902
"""

    with open(filepath, "w") as f:
        f.write(log_content)

    return filepath
