"""
End-to-end tests for PyTorch domain
Binary Classification and Regression with multiple runs

Run with: pytest tests/domains/test_pytorch.py -v
"""

import os
import tempfile
import time

import numpy as np
import pytest
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

import artifacta as ds
from tests.helpers import (
    generate_csv_data,
    generate_synthetic_images,
)


@pytest.mark.e2e
def test_classification_runs():
    """Test multiple classification training runs with epoch-by-epoch metrics"""
    # Vary learning rate AND batch size to test multi-parameter sweeps
    scenarios = [
        ("lr=0.01_bs=16", 0.85, 50, 0.01, 16),
        ("lr=0.001_bs=32", 0.92, 50, 0.001, 32),
        ("lr=0.0001_bs=64", 0.78, 50, 0.0001, 64),
        ("lr=0.00001_bs=128", 0.68, 50, 0.00001, 128),
    ]

    for run_name, target_accuracy, epochs, learning_rate, batch_size in scenarios:
        print(f"\nCreating: {run_name}")
        config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": "adam",
            "epochs": epochs,
        }
        ds.init(project="binary-classification", name=run_name, config=config)
        run = ds.get_run()

        # Accumulate all epochs
        metrics_history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
            "precision": [],
            "val_precision": [],
            "recall": [],
            "val_recall": [],
            "f1": [],
            "val_f1": [],
        }

        # Simulate training with realistic learning curves
        for epoch in range(epochs):
            progress = (epoch + 1) / epochs

            # Training metrics improve over time
            train_loss = 0.7 * np.exp(-3 * progress) + 0.05 + np.random.uniform(-0.02, 0.02)
            val_loss = train_loss * 1.15 + np.random.uniform(-0.05, 0.05)

            # Accuracy improves toward target
            accuracy = min(
                target_accuracy, 0.5 + (target_accuracy - 0.5) * (1 - np.exp(-2 * progress))
            )
            accuracy += np.random.uniform(-0.02, 0.02)
            val_accuracy = accuracy - 0.03 + np.random.uniform(-0.02, 0.02)

            # Other classification metrics
            precision = accuracy + np.random.uniform(-0.05, 0.02)
            recall = accuracy + np.random.uniform(-0.03, 0.03)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

            val_precision = val_accuracy + np.random.uniform(-0.05, 0.02)
            val_recall = val_accuracy + np.random.uniform(-0.03, 0.03)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

            # Accumulate metrics
            metrics_history["loss"].append(train_loss)
            metrics_history["val_loss"].append(val_loss)
            metrics_history["accuracy"].append(accuracy)
            metrics_history["val_accuracy"].append(val_accuracy)
            metrics_history["precision"].append(precision)
            metrics_history["val_precision"].append(val_precision)
            metrics_history["recall"].append(recall)
            metrics_history["val_recall"].append(val_recall)
            metrics_history["f1"].append(f1)
            metrics_history["val_f1"].append(val_f1)

        # Log final evaluation metrics (after training complete)
        # Generate synthetic predictions
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_scores = np.clip(y_true + np.random.normal(0, 0.3, n_samples), 0, 1)
        y_pred = (y_scores > 0.5).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_obj = ds.Matrix(
            rows=["Negative", "Positive"],
            cols=["Predicted Neg", "Predicted Pos"],
            values=cm.tolist(),
        )
        cm_obj._section = "Model Evaluation"
        ds.log("Confusion Matrix", cm_obj)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_obj = ds.Curve(
            x=[float(f) for f in fpr],
            y=[float(t) for t in tpr],
            x_label="False Positive Rate",
            y_label="True Positive Rate",
            baseline="diagonal",
            metric={"name": "AUC", "value": float(roc_auc)},
        )
        roc_obj._section = "Model Evaluation"
        ds.log("ROC Curve", roc_obj)

        # Precision-Recall Curve
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_arr, precision_arr)
        pr_obj = ds.Curve(
            x=[float(r) for r in recall_arr],
            y=[float(p) for p in precision_arr],
            x_label="Recall",
            y_label="Precision",
            baseline="horizontal",
            metric={"name": "AP", "value": float(pr_auc)},
        )
        pr_obj._section = "Model Evaluation"
        ds.log("Precision-Recall Curve", pr_obj)

        # Log metrics as separate series (different scales)
        # Loss metrics
        loss_obj = ds.Series(
            index="epoch",
            fields={
                "train_loss": metrics_history["loss"],
                "val_loss": metrics_history["val_loss"],
            },
            index_values=list(range(epochs)),
        )
        loss_obj._section = "Training Metrics"
        ds.log("Loss", loss_obj)

        # Accuracy metrics
        acc_obj = ds.Series(
            index="epoch",
            fields={
                "train_accuracy": metrics_history["accuracy"],
                "val_accuracy": metrics_history["val_accuracy"],
            },
            index_values=list(range(epochs)),
        )
        acc_obj._section = "Training Metrics"
        ds.log("Accuracy", acc_obj)

        # Precision
        prec_obj = ds.Series(
            index="epoch",
            fields={
                "precision": metrics_history["precision"],
                "val_precision": metrics_history["val_precision"],
            },
            index_values=list(range(epochs)),
        )
        prec_obj._section = "Training Metrics"
        ds.log("precision", prec_obj)

        # Recall
        recall_obj = ds.Series(
            index="epoch",
            fields={
                "recall": metrics_history["recall"],
                "val_recall": metrics_history["val_recall"],
            },
            index_values=list(range(epochs)),
        )
        recall_obj._section = "Training Metrics"
        ds.log("recall", recall_obj)

        # F1 Score
        f1_obj = ds.Series(
            index="epoch",
            fields={"f1": metrics_history["f1"], "val_f1": metrics_history["val_f1"]},
            index_values=list(range(epochs)),
        )
        f1_obj._section = "Training Metrics"
        ds.log("f1", f1_obj)

        # Generate and log sample images as a single artifact (directory)
        images_dir = os.path.join(tempfile.gettempdir(), f"training_images_{run.id}")
        os.makedirs(images_dir, exist_ok=True)
        images = generate_synthetic_images(count=5)
        for filename, filepath in images:
            # Move images to the directory
            dest = os.path.join(images_dir, filename)
            os.rename(filepath, dest)
        run.log_input(images_dir)

        # Log classification training code artifact (directory)
        code_path = os.path.join(os.path.dirname(__file__), "../fixtures/code/pytorch")
        run.log_input(code_path)

        print(f"  Completed: {run_name} (final acc: {accuracy:.3f})")
        time.sleep(0.3)


@pytest.mark.e2e
def test_regression_runs():
    """Test multiple regression training runs with epoch-by-epoch metrics"""
    # Vary batch size to demonstrate hyperparameter sweep
    scenarios = [
        ("batch=16", 0.78, 50, 16),
        ("batch=32", 0.85, 50, 32),
        ("batch=64", 0.80, 50, 64),
        ("batch=128", 0.72, 50, 128),
    ]

    for run_name, target_r2, epochs, batch_size in scenarios:
        print(f"\nCreating: {run_name}")
        config = {
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "optimizer": "adam",
            "epochs": epochs,
        }
        ds.init(project="regression", name=run_name, config=config)
        run = ds.get_run()

        # Accumulate all epochs
        metrics_history = {
            "loss": [],
            "val_loss": [],
            "r2": [],
            "val_r2": [],
            "mse": [],
            "val_mse": [],
            "mae": [],
            "val_mae": [],
        }

        # Simulate training
        for epoch in range(epochs):
            progress = (epoch + 1) / epochs

            # Loss decreases over time
            train_loss = 10.0 * np.exp(-2 * progress) + 0.5 + np.random.uniform(-0.1, 0.1)
            val_loss = train_loss * 1.2 + np.random.uniform(-0.2, 0.2)

            # R² improves toward target
            r2 = min(target_r2, -0.5 + (target_r2 + 0.5) * (1 - np.exp(-2 * progress)))
            r2 += np.random.uniform(-0.03, 0.03)
            val_r2 = r2 - 0.05 + np.random.uniform(-0.03, 0.03)

            # MSE and MAE decrease
            mse = train_loss**2
            mae = train_loss * 0.8
            val_mse = val_loss**2
            val_mae = val_loss * 0.8

            # Accumulate metrics
            metrics_history["loss"].append(train_loss)
            metrics_history["val_loss"].append(val_loss)
            metrics_history["r2"].append(r2)
            metrics_history["val_r2"].append(val_r2)
            metrics_history["mse"].append(mse)
            metrics_history["val_mse"].append(val_mse)
            metrics_history["mae"].append(mae)
            metrics_history["val_mae"].append(val_mae)

        # Log final evaluation metrics (after training complete)
        # Generate synthetic regression data
        n_samples = 100
        true_vals = np.random.uniform(0, 100, n_samples)
        noise_level = (1 - r2) * 20
        predicted_vals = true_vals + np.random.normal(0, noise_level, n_samples)

        # Actual vs Predicted scatter
        scatter_data = [
            {"actual": float(a), "predicted": float(p)} for a, p in zip(true_vals, predicted_vals)
        ]
        scatter_obj = ds.Scatter(
            points=scatter_data,
            x_label="Actual",
            y_label="Predicted",
        )
        scatter_obj._section = "Model Performance"
        ds.log("Actual vs Predicted", scatter_obj)

        # Residuals
        residuals = true_vals - predicted_vals
        residual_data = [
            {"predicted": float(p), "residual": float(r)} for p, r in zip(predicted_vals, residuals)
        ]
        residual_obj = ds.Scatter(
            points=residual_data,
            x_label="Predicted Value",
            y_label="Residual",
        )
        residual_obj._section = "Model Performance"
        ds.log("Residuals", residual_obj)

        # Log metrics as separate series (different scales)
        # Loss metrics
        loss_obj = ds.Series(
            index="epoch",
            fields={
                "train_loss": metrics_history["loss"],
                "val_loss": metrics_history["val_loss"],
            },
            index_values=list(range(epochs)),
        )
        loss_obj._section = "Training Metrics"
        ds.log("Loss", loss_obj)

        # R² metrics
        r2_obj = ds.Series(
            index="epoch",
            fields={"train_r2": metrics_history["r2"], "val_r2": metrics_history["val_r2"]},
            index_values=list(range(epochs)),
        )
        r2_obj._section = "Training Metrics"
        ds.log("R² Score", r2_obj)

        # MSE metrics
        mse_obj = ds.Series(
            index="epoch",
            fields={
                "train_mse": metrics_history["mse"],
                "val_mse": metrics_history["val_mse"],
            },
            index_values=list(range(epochs)),
        )
        mse_obj._section = "Training Metrics"
        ds.log("MSE", mse_obj)

        # MAE metrics
        mae_obj = ds.Series(
            index="epoch",
            fields={
                "train_mae": metrics_history["mae"],
                "val_mae": metrics_history["val_mae"],
            },
            index_values=list(range(epochs)),
        )
        mae_obj._section = "Training Metrics"
        ds.log("MAE", mae_obj)

        # Generate and log training data CSV as artifact
        csv_filename, csv_path = generate_csv_data("training_samples.csv", rows=20)
        run.log_output(csv_path)

        # Log regression training code artifact
        code_path = os.path.join(os.path.dirname(__file__), "../fixtures/code/pytorch")
        run.log_input(code_path)

        print(f"  Completed: {run_name} (final R²: {r2:.3f})")
        time.sleep(0.3)
