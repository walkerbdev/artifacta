"""
XGBoost Regression with Artifacta Autolog
==========================================

This example demonstrates Artifacta's autolog integration for XGBoost:
1. **Automatic parameter logging** - XGBoost hyperparameters logged automatically
2. **Automatic metric logging** - Training and validation metrics per iteration
3. **Automatic model logging** - Trained booster saved as artifact
4. **Feature importance** - Which features matter most
5. **Prediction analysis** - Predicted vs actual scatter plot
6. **Hyperparameter sweep** - Test multiple configurations

Key Artifacta Features:
- autolog() - Enable automatic logging for XGBoost
- Series - Training/validation curves over iterations
- Scatter - Predicted vs actual values
- BarChart - Feature importance visualization

Requirements:
    pip install artifacta xgboost scikit-learn numpy

Usage:
    python examples/ml_frameworks/xgboost_regression.py
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from artifacta import BarChart, Scatter, Series, autolog, init


def main():
    print("=" * 70)
    print("Artifacta - XGBoost Regression Example")
    print("=" * 70)

    # =================================================================
    # 1. Define hyperparameter grid for sweep
    # =================================================================
    configs = [
        {
            "name": "shallow-fast",
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 50,
        },
        {
            "name": "medium-balanced",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 100,
        },
        {
            "name": "deep-slow",
            "max_depth": 10,
            "learning_rate": 0.01,
            "n_estimators": 150,
        },
    ]

    print(f"\nRunning hyperparameter sweep with {len(configs)} configurations\n")

    # =================================================================
    # 2. Generate synthetic regression dataset once
    # =================================================================
    print("Generating synthetic regression dataset...")

    X, y = make_regression(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X.shape[1]}")

    # =================================================================
    # 3. Run experiments for each configuration
    # =================================================================
    for idx, config in enumerate(configs, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {idx}/{len(configs)}: {config['name']}")
        print(f"{'=' * 70}")

        # Initialize Artifacta run
        run_config = {
            "model": "XGBRegressor",
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "n_estimators": config["n_estimators"],
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        run = init(
            project="xgboost-demo",
            name=f"xgb-{config['name']}",
            config=run_config,
        )

        print("\nConfiguration:")
        print(f"  max_depth: {config['max_depth']}")
        print(f"  learning_rate: {config['learning_rate']}")
        print(f"  n_estimators: {config['n_estimators']}")

        # =================================================================
        # 4. Enable autolog
        #    Automatically logs: parameters, per-iteration metrics, model
        # =================================================================
        autolog(framework="xgboost")

        # =================================================================
        # 5. Train XGBoost model
        #    Autolog captures training/validation metrics automatically
        # =================================================================
        print("\nTraining XGBoost model...")

        model = xgb.XGBRegressor(
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            n_estimators=config["n_estimators"],
            objective="reg:squarederror",
            random_state=42,
            eval_metric="rmse",
        )

        # Fit with validation set for eval metrics
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

        print("  Training complete!")

        # =================================================================
        # 6. Make predictions and calculate metrics
        # =================================================================
        print("\nEvaluating model...")

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")

        # =================================================================
        # 7. Log predictions vs actual (Scatter plot)
        #    Points should fall on y=x diagonal for perfect predictions
        # =================================================================
        print("\nLogging prediction analysis...")

        points = [
            {"x": float(actual), "y": float(pred)}
            for actual, pred in zip(y_test, y_pred)
        ]

        run.log(
            "predictions_vs_actual",
            Scatter(
                points=points,
                x_label="Actual Values",
                y_label="Predicted Values",
                metadata={
                    "description": "Predictions vs actual on test set",
                    "rmse": float(rmse),
                    "r2": float(r2),
                },
            ),
        )

        # =================================================================
        # 8. Log feature importance
        #    Shows which features contribute most to predictions
        # =================================================================
        print("\nLogging feature importance...")

        importance_scores = model.feature_importances_
        # Get top 10 features
        top_indices = np.argsort(importance_scores)[-10:][::-1]

        run.log(
            "feature_importance",
            BarChart(
                categories=[f"Feature_{i}" for i in top_indices],
                groups={"Importance": importance_scores[top_indices].tolist()},
                x_label="Feature",
                y_label="Importance Score",
                metadata={"description": "Top 10 most important features"},
            ),
        )

        # =================================================================
        # 9. Log final metrics as Series (for comparison)
        # =================================================================
        run.log(
            "final_metrics",
            Series(
                index="metric",
                fields={
                    "value": [float(rmse), float(mae), float(r2)],
                },
                index_values=["RMSE", "MAE", "R²"],
            ),
        )

        # Finish run
        run.finish()

        print(f"\nExperiment {config['name']} complete")

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 70)
    print("Hyperparameter Sweep Complete!")
    print("=" * 70)
    print(f"Trained {len(configs)} XGBoost models")
    print("  Configurations tested:")
    for config in configs:
        print(f"    - {config['name']}: depth={config['max_depth']}, lr={config['learning_rate']}")
    print("\nAll metrics and artifacts logged to Artifacta")
    print("  Automatically logged (per run):")
    print("    - XGBoost parameters (autolog)")
    print("    - Training/validation curves (autolog)")
    print("    - Trained model (autolog)")
    print("  Manually logged (per run):")
    print("    - Prediction scatter plot")
    print("    - Feature importance")
    print("    - Final metrics")
    print("\nView and compare all runs in the Artifacta UI!")
    print("Use the comparison view to find the best configuration.")
    print("=" * 70)


if __name__ == "__main__":
    main()
