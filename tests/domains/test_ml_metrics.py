"""
End-to-end tests for ML Metrics domain
ROC/PR Curves and Embeddings/t-SNE with parameter sweeps

Run with: pytest tests/domains/test_ml_metrics.py -v
"""

import os
import time

import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import artifacta as ds


def run_parameter_sweep(project_name, base_config, param_variations, run_fn):
    """
    Helper to run multiple experiments with parameter variations.

    Args:
        project_name: Project name for the runs
        base_config: Base configuration dict
        param_variations: List of dicts with parameter overrides
        run_fn: Function(config, seed) that executes the run logic
    """
    for idx, variation in enumerate(param_variations):
        config = {**base_config, **variation}
        # Use project name in run name to avoid collisions across different sweep tests
        run_name = f"{project_name}-run-{idx + 1}"
        ds.init(project=project_name, name=run_name, config=config)
        # Pass seed to make results vary but be reproducible
        run_fn(config, seed=42 + idx)
        time.sleep(0.3)


@pytest.mark.e2e
def test_ml_roc_pr_curves():
    """Test 9: ML Classification - ROC & PR Curves with parameter sweep"""

    def run_ml_classification(config, seed=42):
        run = ds.get_run()

        # Generate synthetic binary classification data
        np.random.seed(seed)
        y_true = np.random.binomial(1, 0.3, 500)
        y_scores = np.clip(y_true + np.random.normal(0, 0.4, 500), 0, 1)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_curve_obj = ds.Curve(
            x=fpr.tolist(),
            y=tpr.tolist(),
            x_label="False Positive Rate",
            y_label="True Positive Rate",
            baseline="diagonal",
            metric={"name": "AUC", "value": float(roc_auc)},
        )
        roc_curve_obj._section = "Model Evaluation"
        ds.log("roc_curve", roc_curve_obj)

        # Precision-Recall Curve
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_arr, precision_arr)
        pr_curve_obj = ds.Curve(
            x=recall_arr.tolist(),
            y=precision_arr.tolist(),
            x_label="Recall",
            y_label="Precision",
            baseline="horizontal",
            metric={"name": "AUC", "value": float(pr_auc)},
        )
        pr_curve_obj._section = "Model Evaluation"
        ds.log("pr_curve", pr_curve_obj)

        # Log ML classification code artifact
        code_path = os.path.join(os.path.dirname(__file__), "../fixtures/code/ml_metrics")
        run.log_input(code_path)

    base_config = {"model": "logistic_regression", "solver": "lbfgs"}
    param_variations = [
        {"C": 1.0, "max_iter": 100},
        {"C": 0.1, "max_iter": 100},
        {"C": 1.0, "max_iter": 200},
    ]

    run_parameter_sweep(
        "ml-classification-sweep", base_config, param_variations, run_ml_classification
    )


@pytest.mark.e2e
def test_embeddings():
    """Test 10: Embeddings - t-SNE Clustering with parameter sweep"""

    def run_embeddings(config, seed=42):
        run = ds.get_run()

        # Scatter: 2D embedding visualization
        np.random.seed(seed)
        n_points = 100
        clusters = np.random.choice(["technology", "sports", "politics"], n_points)

        # Generate clustered 2D points
        points = []
        for cluster in clusters:
            if cluster == "technology":
                x, y = np.random.normal(-2, 0.5), np.random.normal(2, 0.5)
            elif cluster == "sports":
                x, y = np.random.normal(2, 0.5), np.random.normal(2, 0.5)
            else:
                x, y = np.random.normal(0, 0.5), np.random.normal(-2, 0.5)
            points.append({"x": float(x), "y": float(y), "label": cluster})

        scatter_obj = ds.Scatter(
            points=points,
            x_label="t-SNE Dimension 1",
            y_label="t-SNE Dimension 2",
        )
        scatter_obj._section = "Embedding Visualization"
        ds.log("embedding_tsne", scatter_obj)

        # Log embedding visualization code artifact
        code_path = os.path.join(os.path.dirname(__file__), "../fixtures/code/ml_metrics")
        run.log_input(code_path)

    base_config = {"window_size": 5, "min_count": 2}
    param_variations = [
        {"embedding_dim": 100, "perplexity": 30},
        {"embedding_dim": 50, "perplexity": 30},
        {"embedding_dim": 100, "perplexity": 50},
    ]

    run_parameter_sweep("embeddings-sweep", base_config, param_variations, run_embeddings)
