"""
All Artifacta Primitives Demonstration
======================================

This example demonstrates all 7 Artifacta data primitives in one place:
1. Series - Ordered data over a single dimension (time series, epochs)
2. Distribution - Value collections with optional grouping (histograms)
3. Matrix - 2D relationships (confusion matrices, heatmaps)
4. Table - Generic tabular data (measurements, logs)
5. Curve - Pure X-Y relationships (ROC curves, dose-response)
6. Scatter - Unordered point clouds (embeddings, correlations)
7. BarChart - Categorical comparisons (model performance)

Each primitive is domain-agnostic and works for any field (ML, physics, finance, etc.).

Requirements:
    pip install artifacta numpy

Usage:
    python examples/core/02_all_primitives.py
"""

import numpy as np

from artifacta import BarChart, Curve, Distribution, Matrix, Scatter, Series, Table, init


def main():
    print("=" * 70)
    print("Artifacta - All Primitives Demonstration")
    print("=" * 70)

    # Initialize run
    run = init(
        project="primitives-demo",
        name="all-primitives",
        config={"description": "Showcase all 7 data primitives"},
    )

    print("\nArtifacta run initialized\n")

    # =================================================================
    # 1. Series - Ordered data over single dimension
    #    Use cases: Training loss, stock prices, temperature over time
    # =================================================================
    print("Logging: Series (training metrics over epochs)")

    run.log(
        "training_metrics",
        Series(
            index="epoch",
            fields={
                "train_loss": [0.8, 0.5, 0.3, 0.2, 0.15],
                "val_loss": [0.9, 0.6, 0.4, 0.3, 0.25],
                "accuracy": [0.6, 0.75, 0.85, 0.90, 0.93],
            },
            index_values=[1, 2, 3, 4, 5],
            metadata={"description": "Training progress over 5 epochs"},
        ),
    )

    # =================================================================
    # 2. Distribution - Values with optional grouping
    #    Use cases: A/B testing, prediction distributions, response times
    # =================================================================
    print("Logging: Distribution (A/B test results)")

    # Simulate A/B test conversion rates
    control_conversions = np.random.binomial(1, 0.05, 100)
    variant_conversions = np.random.binomial(1, 0.07, 100)

    all_values = np.concatenate([control_conversions, variant_conversions])
    all_groups = ["Control"] * 100 + ["Variant"] * 100

    run.log(
        "ab_test_distribution",
        Distribution(
            values=all_values.tolist(),
            groups=all_groups,
            metadata={
                "description": "Conversion outcomes by variant",
                "control_rate": float(control_conversions.mean()),
                "variant_rate": float(variant_conversions.mean()),
            },
        ),
    )

    # =================================================================
    # 3. Matrix - 2D relationships
    #    Use cases: Confusion matrices, correlation matrices, heatmaps
    # =================================================================
    print("Logging: Matrix (confusion matrix)")

    run.log(
        "confusion_matrix",
        Matrix(
            rows=["Cat", "Dog", "Bird"],
            cols=["Cat", "Dog", "Bird"],
            values=[
                [85, 10, 5],  # Cat predictions
                [8, 87, 5],  # Dog predictions
                [7, 8, 85],  # Bird predictions
            ],
            metadata={"type": "confusion_matrix", "total_samples": 300},
        ),
    )

    # =================================================================
    # 4. Table - Generic tabular data
    #    Use cases: Event logs, measurements, multi-variable data
    # =================================================================
    print("Logging: Table (experiment measurements)")

    run.log(
        "measurements",
        Table(
            columns=[
                {"name": "experiment_id", "type": "string"},
                {"name": "temperature_C", "type": "float"},
                {"name": "pressure_bar", "type": "float"},
                {"name": "yield_pct", "type": "float"},
            ],
            data=[
                ["exp_001", 25.0, 1.0, 78.5],
                ["exp_002", 30.0, 1.2, 82.3],
                ["exp_003", 35.0, 1.5, 85.1],
                ["exp_004", 40.0, 1.8, 79.2],
            ],
            metadata={"description": "Chemical reaction optimization data"},
        ),
    )

    # =================================================================
    # 5. Curve - Pure X-Y relationships (not time-indexed)
    #    Use cases: ROC curves, dose-response, calibration curves
    # =================================================================
    print("Logging: Curve (ROC curve)")

    # Simulate ROC curve data
    fpr = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    tpr = [0.0, 0.6, 0.75, 0.85, 0.92, 0.97, 1.0]
    auc = 0.92

    run.log(
        "roc_curve",
        Curve(
            x=fpr,
            y=tpr,
            x_label="False Positive Rate",
            y_label="True Positive Rate",
            baseline="diagonal",  # Show y=x diagonal reference line
            metric={"name": "AUC", "value": auc},
            metadata={"description": "ROC curve for binary classifier"},
        ),
    )

    # =================================================================
    # 6. Scatter - Unordered point clouds
    #    Use cases: Feature correlations, embeddings (t-SNE, UMAP)
    # =================================================================
    print("Logging: Scatter (feature correlation)")

    # Generate correlated features
    np.random.seed(42)
    feature1 = np.random.randn(50)
    feature2 = feature1 * 0.8 + np.random.randn(50) * 0.3  # Correlated

    points = [
        {"x": float(x), "y": float(y), "label": "data_point"}
        for x, y in zip(feature1, feature2)
    ]

    run.log(
        "feature_correlation",
        Scatter(
            points=points,
            x_label="Feature 1",
            y_label="Feature 2",
            metadata={
                "description": "Correlation between two features",
                "correlation": float(np.corrcoef(feature1, feature2)[0, 1]),
            },
        ),
    )

    # =================================================================
    # 7. BarChart - Categorical comparisons
    #    Use cases: Model performance, metrics by group
    # =================================================================
    print("Logging: BarChart (model comparison)")

    run.log(
        "model_comparison",
        BarChart(
            categories=["LogisticReg", "RandomForest", "XGBoost", "NeuralNet"],
            groups={
                "Accuracy": [0.85, 0.91, 0.94, 0.93],
                "F1-Score": [0.82, 0.89, 0.92, 0.91],
                "Precision": [0.84, 0.90, 0.93, 0.92],
            },
            x_label="Model",
            y_label="Score",
            metadata={"description": "Classification performance comparison"},
        ),
    )

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("All 7 Primitives Logged Successfully!")
    print("=" * 70)
    print("Series      - Training metrics over epochs")
    print("Distribution - A/B test conversion outcomes")
    print("Matrix      - Confusion matrix")
    print("Table       - Experiment measurements")
    print("Curve       - ROC curve")
    print("Scatter     - Feature correlation")
    print("BarChart    - Model performance comparison")
    print("\nView all visualizations in the Artifacta UI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
