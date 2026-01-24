"""
Scikit-learn Classification with Artifacta Autolog
==================================================

This example demonstrates Artifacta's autolog integration for scikit-learn:
1. **Automatic parameter logging** - Model hyperparameters logged automatically
2. **Automatic metric logging** - Accuracy, precision, recall, F1 automatically computed
3. **Automatic model logging** - Trained model saved as artifact
4. **ROC/PR curves** - Binary classification performance visualization
5. **Confusion matrix** - Multi-class classification analysis
6. **Feature importance** - For tree-based models

Key Artifacta Features:
- autolog() - Enable automatic logging for sklearn
- Curve - ROC and Precision-Recall curves
- Matrix - Confusion matrix
- BarChart - Feature importance

Requirements:
    pip install artifacta scikit-learn numpy

Usage:
    python examples/ml_frameworks/sklearn_classification.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from artifacta import BarChart, Curve, Matrix, autolog, init


def main():
    print("=" * 70)
    print("Artifacta - Scikit-learn Classification Example")
    print("=" * 70)

    # =================================================================
    # 1. Initialize Artifacta run with configuration
    # =================================================================
    config = {
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "dataset": "synthetic_binary",
        "n_samples": 2000,
        "n_features": 20,
    }

    run = init(
        project="sklearn-demo",
        name="rf-binary-classification",
        config=config,
    )

    print("\nArtifacta run initialized")

    # =================================================================
    # 2. Enable autolog for automatic parameter/metric/model logging
    #    This will automatically log model parameters and training metrics
    # =================================================================
    autolog(framework="sklearn")

    # =================================================================
    # 3. Create synthetic binary classification dataset
    # =================================================================
    print("\nGenerating synthetic dataset...")

    X, y = make_classification(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=config["random_state"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["random_state"]
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X.shape[1]}")

    # =================================================================
    # 4. Train Random Forest classifier
    #    Autolog automatically captures:
    #    - Model parameters (n_estimators, max_depth, etc.)
    #    - Training metrics (accuracy, precision, recall, F1)
    #    - Trained model artifact
    # =================================================================
    print("\nTraining Random Forest classifier...")

    clf = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        random_state=config["random_state"],
    )

    clf.fit(X_train, y_train)

    print("  Training complete!")

    # Get predictions and probabilities
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

    # =================================================================
    # 5. Log ROC Curve (Receiver Operating Characteristic)
    #    Shows trade-off between true positive rate and false positive rate
    # =================================================================
    print("\nLogging ROC curve...")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    run.log(
        "roc_curve",
        Curve(
            x=fpr.tolist(),
            y=tpr.tolist(),
            x_label="False Positive Rate",
            y_label="True Positive Rate",
            baseline="diagonal",
            metric={"name": "AUC-ROC", "value": float(auc)},
            metadata={
                "description": "ROC curve for binary classification",
                "interpretation": "Higher AUC (closer to 1.0) indicates better performance",
            },
        ),
    )

    print(f"  AUC-ROC: {auc:.4f}")

    # =================================================================
    # 6. Log Precision-Recall Curve
    #    Shows trade-off between precision and recall
    #    More informative than ROC for imbalanced datasets
    # =================================================================
    print("\nLogging Precision-Recall curve...")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    # Calculate Average Precision (area under PR curve)
    from sklearn.metrics import average_precision_score

    avg_precision = average_precision_score(y_test, y_proba)

    run.log(
        "precision_recall_curve",
        Curve(
            x=recall.tolist(),
            y=precision.tolist(),
            x_label="Recall",
            y_label="Precision",
            metric={"name": "Average Precision", "value": float(avg_precision)},
            metadata={
                "description": "Precision-Recall curve",
                "interpretation": "Higher average precision indicates better performance",
            },
        ),
    )

    print(f"  Average Precision: {avg_precision:.4f}")

    # =================================================================
    # 7. Log Confusion Matrix
    #    Shows how predictions compare to actual labels
    # =================================================================
    print("\nLogging confusion matrix...")

    cm = confusion_matrix(y_test, y_pred)

    run.log(
        "confusion_matrix",
        Matrix(
            rows=["Negative (0)", "Positive (1)"],
            cols=["Predicted Negative", "Predicted Positive"],
            values=cm.tolist(),
            metadata={
                "type": "confusion_matrix",
                "total_samples": int(cm.sum()),
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1]),
            },
        ),
    )

    # =================================================================
    # 8. Log Feature Importance (for tree-based models)
    #    Shows which features contribute most to predictions
    # =================================================================
    print("\nLogging feature importance...")

    importances = clf.feature_importances_
    # Get top 10 most important features
    top_indices = np.argsort(importances)[-10:][::-1]

    run.log(
        "feature_importance",
        BarChart(
            categories=[f"Feature_{i}" for i in top_indices],
            groups={"Importance": importances[top_indices].tolist()},
            x_label="Feature",
            y_label="Importance",
            metadata={"description": "Top 10 most important features"},
        ),
    )

    # =================================================================
    # 9. Final Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("Model Performance:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Test Accuracy: {clf.score(X_test, y_test):.4f}")
    print("\nAll metrics and artifacts logged to Artifacta")
    print("  Automatically logged:")
    print("    - Model parameters (autolog)")
    print("    - Training metrics (autolog)")
    print("    - Trained model (autolog)")
    print("  Manually logged:")
    print("    - ROC curve")
    print("    - Precision-Recall curve")
    print("    - Confusion matrix")
    print("    - Feature importance")
    print("\nView results in the Artifacta UI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
