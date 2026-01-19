"""
End-to-end tests for Computer Vision domain
Machine Learning - Image Classification with parameter sweep

Run with: pytest tests/domains/test_computer_vision.py -v
"""

import os
import time

import numpy as np
import pytest

import artifacta as ds
from tests.helpers import (
    create_fake_model_checkpoint,
    create_model_metadata,
    create_predictions_csv,
    create_training_log,
)
from tests.helpers.notebook_html import (
    create_computer_vision_notebook,
    create_experiment_summary_notebook,
)

# Path to example multi-file ML pipeline
ML_PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "../examples/ml_pipeline")


def run_parameter_sweep(project_name, base_config, param_variations, run_fn):
    """
    Helper to run multiple experiments with parameter variations.

    Args:
        project_name: Project name for the runs
        base_config: Base configuration dict
        param_variations: List of dicts with parameter overrides
        run_fn: Function(config, seed) that executes the run logic

    Returns:
        List of (run_id, config, results) tuples
    """
    results = []
    for idx, variation in enumerate(param_variations):
        config = {**base_config, **variation}
        # Use project name in run name to avoid collisions across different sweep tests
        run_name = f"{project_name}-run-{idx + 1}"
        ds.init(project=project_name, name=run_name, config=config)
        run = ds.get_run()
        # Pass seed to make results vary but be reproducible
        result = run_fn(config, seed=42 + idx)
        results.append((run.id, config, result))
        time.sleep(0.3)
    return results


@pytest.mark.e2e
def test_ml_classification(api_url):
    """Test 1: Machine Learning - Image Classification with parameter sweep"""

    def run_resnet_training(config, seed=42):
        run = ds.get_run()
        np.random.seed(seed)

        # Get config parameters
        lr = config.get("lr", 0.001)
        epochs = config.get("epochs", 10)

        # Vary training dynamics based on learning rate
        # Higher lr → faster initial convergence but potentially worse final accuracy
        convergence_rate = 2.0 + lr * 500  # Higher lr = faster convergence
        final_accuracy = 0.95 - (lr - 0.001) * 20  # Lower lr often gives better final accuracy
        final_accuracy = max(0.85, min(0.95, final_accuracy))  # Clamp between 0.85-0.95

        # Generate training curves
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        for epoch in range(1, epochs + 1):
            progress = epoch / epochs

            # Loss decreases based on learning rate
            t_loss = (
                2.5 * np.exp(-convergence_rate * progress) + 0.05 + np.random.uniform(-0.02, 0.02)
            )
            v_loss = t_loss * 1.15 + np.random.uniform(-0.05, 0.05)

            # Accuracy improves toward final_accuracy
            t_acc = final_accuracy * (1 - np.exp(-convergence_rate * progress)) + np.random.uniform(
                -0.02, 0.02
            )
            v_acc = t_acc - 0.03 + np.random.uniform(-0.02, 0.02)

            train_loss.append(float(t_loss))
            val_loss.append(float(v_loss))
            train_acc.append(float(t_acc))
            val_acc.append(float(v_acc))

        # Generate fake predictions for confusion matrix
        n_classes = 10
        n_samples = 150
        y_true = np.random.randint(0, n_classes, n_samples)
        # Add some noise to predictions based on final accuracy
        correct_ratio = final_accuracy
        y_pred = y_true.copy()
        n_incorrect = int(n_samples * (1 - correct_ratio))
        incorrect_indices = np.random.choice(n_samples, n_incorrect, replace=False)
        for idx in incorrect_indices:
            # Randomly assign incorrect class
            wrong_classes = [c for c in range(n_classes) if c != y_true[idx]]
            y_pred[idx] = np.random.choice(wrong_classes)

        # Series: Training loss over epochs
        loss_obj = ds.Series(
            index="epoch",
            fields={
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            index_values=list(range(1, epochs + 1)),
        )
        loss_obj._section = "Training Metrics"
        ds.log("training_loss", loss_obj)

        # Series: Training accuracy over epochs
        acc_obj = ds.Series(
            index="epoch",
            fields={
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            },
            index_values=list(range(1, epochs + 1)),
        )
        acc_obj._section = "Training Metrics"
        ds.log("training_accuracy", acc_obj)

        # Matrix: Confusion matrix - varies with final accuracy
        base_correct = int(150 * final_accuracy / 4)  # per class
        confusion_values = []
        for i in range(4):
            row = []
            for j in range(4):
                if i == j:
                    row.append(base_correct + np.random.randint(-5, 5))
                else:
                    row.append(np.random.randint(1, 8))
            confusion_values.append(row)

        cm_obj = ds.Matrix(
            rows=["cat", "dog", "bird", "fish"],
            cols=["cat", "dog", "bird", "fish"],
            values=confusion_values,
        )
        cm_obj._section = "Validation Results"
        cm_obj._metadata = {"accuracy": float(final_accuracy)}
        ds.log("confusion_matrix", cm_obj)

        # Distribution: Class-wise prediction confidence
        confidence_mean = 0.7 + final_accuracy * 0.2  # Higher accuracy = higher confidence
        confidence_values = np.clip(
            np.random.beta(8, 2, 20) * confidence_mean + (1 - confidence_mean), 0.5, 0.99
        ).tolist()

        dist_obj = ds.Distribution(
            values=confidence_values,
            groups=["cat"] * 5 + ["dog"] * 5 + ["bird"] * 5 + ["fish"] * 5,
        )
        dist_obj._section = "Validation Results"
        ds.log("prediction_confidence", dist_obj)

        # Log multi-file ML pipeline (directory with code, config, utils)
        run.log_input(ML_PIPELINE_PATH)

        # Log model checkpoint with rich metadata
        model_path = create_fake_model_checkpoint("pytorch")
        run.log_output(
            model_path,
            metadata={
                "framework": "PyTorch",
                "architecture": config["architecture"],
                "optimizer": config["optimizer"],
                "learning_rate": config["lr"],
                "train": {
                    "accuracy": float(final_accuracy),
                    "loss": float(val_loss[-1]),
                    "epochs": config["epochs"],
                },
                "model_size_mb": 45.3,
                "author": "CV Team",
            },
        )

        # Log model metadata (auto-detects as "metadata" from .json extension)
        metadata_path = create_model_metadata(
            config=config,
            metrics={
                "final_val_accuracy": float(final_accuracy),
                "final_val_loss": float(val_loss[-1]),
            },
        )
        run.log_output(metadata_path)

        # Log predictions with dataset info
        predictions_path = create_predictions_csv(n_samples=150)
        run.log_output(
            predictions_path,
            metadata={
                "samples": 150,
                "classes": ["cat", "dog"],
                "format": "CSV",
                "confidence": {"mean": 0.87, "min": 0.45, "max": 0.99},
            },
        )

        # Log training log as output (auto-detects as "log" from .log extension)
        log_path = create_training_log()
        run.log_output(log_path)

        # Return results for notebook generation
        return {
            "final_accuracy": final_accuracy,
            "final_loss": float(val_loss[-1]),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    base_config = {"architecture": "ResNet50", "epochs": 10}
    param_variations = [
        {"optimizer": "Adam", "lr": 0.001},
        {"optimizer": "Adam", "lr": 0.0001},
        {"optimizer": "SGD", "lr": 0.01},
    ]

    # Run parameter sweep
    sweep_results = run_parameter_sweep(
        "computer-vision", base_config, param_variations, run_resnet_training
    )

    # Extract configs and results
    configs = [r[1] for r in sweep_results]
    results = [r[2] for r in sweep_results]

    # Create experiment summary notebook
    metrics_summary = [
        {
            "optimizer": config["optimizer"],
            "learning_rate": config["lr"],
            "accuracy": result["final_accuracy"],
            "loss": result["final_loss"],
        }
        for config, result in zip(configs, results)
    ]

    # Find best config
    best_idx = max(range(len(results)), key=lambda i: results[i]["final_accuracy"])
    best_config = configs[best_idx]

    create_experiment_summary_notebook(
        project_id="computer-vision",
        experiment_name="ResNet50 CIFAR-10",
        metrics_summary=metrics_summary,
        best_config=best_config,
        api_url=api_url,
    )

    # Create detailed computer vision notebook with confusion matrix for best run
    best_result = results[best_idx]
    labels = [f"Class_{i}" for i in range(10)]

    create_computer_vision_notebook(
        project_id="computer-vision",
        metrics={
            "Test Accuracy": best_result["final_accuracy"],
            "Test Loss": best_result["final_loss"],
            "Precision": 0.92,
            "Recall": 0.91,
            "F1 Score": 0.915,
        },
        confusion_data=(best_result["y_true"], best_result["y_pred"], labels),
        api_url=api_url,
    )
