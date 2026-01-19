"""
TensorFlow/Keras Regression Example with Artifacta
===================================================

This example demonstrates Artifacta's logging capabilities for TensorFlow/Keras training:

1. **Automatic checkpoint logging** via ds.autolog() - tracks model checkpoints automatically
2. **Training curves** via ds.Series - tracks train/validation loss over epochs
3. **Prediction visualization** via ds.Scatter - plots predicted vs actual values
4. **Residual analysis** via ds.Distribution - analyzes prediction errors
5. **Model artifact logging** - saves trained model with metadata

Key Artifacta Features Demonstrated:
- ds.init() - Initialize experiment run with config
- ds.autolog() - Enable automatic checkpoint logging for Keras
- ds.Series - Log time-series metrics (loss curves)
- ds.Scatter - Log 2D scatter plots (predictions vs actual)
- ds.Distribution - Log value distributions (residuals)
- run.log_output() - Save model artifacts

Requirements:
    pip install artifacta tensorflow scikit-learn numpy

Usage:
    python examples/tensorflow_regression.py
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import artifacta as ds

# Import TensorFlow/Keras
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Error: TensorFlow is not installed.")
    print("Please install it with: pip install tensorflow")
    exit(1)


def create_synthetic_data(n_samples=1000, n_features=10, noise=10.0, random_state=42):
    """Create synthetic regression dataset.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (x_train, x_test, y_train, y_test, scaler_x, scaler_y)
    """
    print("\n✓ Generating synthetic regression data...")

    # Generate regression data
    # make_regression creates a random regression problem
    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,  # 8 out of 10 features are informative
        noise=noise,
        random_state=random_state,
    )

    print(f"  Dataset shape: {x.shape}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Split into train/test sets (80/20 split)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    # Standardize features (zero mean, unit variance)
    # This is important for neural network training
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)

    # Standardize targets for better training stability
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")

    return x_train, x_test, y_train, y_test, scaler_x, scaler_y


def create_model(input_dim, hidden_dim=64, learning_rate=0.001):
    """Create a simple feedforward neural network for regression.

    Architecture:
    - Input layer (input_dim features)
    - Dense layer (hidden_dim neurons, ReLU activation)
    - Dropout (0.2) for regularization
    - Dense layer (hidden_dim // 2 neurons, ReLU activation)
    - Dropout (0.2)
    - Dense layer (hidden_dim // 4 neurons, ReLU activation)
    - Output layer (1 neuron, linear activation)

    Args:
        input_dim: Number of input features
        hidden_dim: Number of neurons in first hidden layer
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled Keras model
    """
    print("\n✓ Creating neural network model...")

    model = keras.Sequential(
        [
            # Input layer
            layers.Input(shape=(input_dim,)),
            # Hidden layer 1: hidden_dim neurons with ReLU activation
            layers.Dense(hidden_dim, activation="relu", name="dense_1"),
            layers.Dropout(0.2, name="dropout_1"),  # Dropout for regularization
            # Hidden layer 2: hidden_dim // 2 neurons with ReLU activation
            layers.Dense(hidden_dim // 2, activation="relu", name="dense_2"),
            layers.Dropout(0.2, name="dropout_2"),
            # Hidden layer 3: hidden_dim // 4 neurons with ReLU activation
            layers.Dense(hidden_dim // 4, activation="relu", name="dense_3"),
            # Output layer: Single neuron for regression
            layers.Dense(1, activation="linear", name="output"),
        ]
    )

    # Compile model with Adam optimizer and MSE loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",  # Mean Squared Error for regression
        metrics=["mae"],  # Also track Mean Absolute Error
    )

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Count parameters
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")

    return model


def main():
    """Main training function."""
    print("=" * 70)
    print("Artifacta TensorFlow Regression Example")
    print("=" * 70)

    # =================================================================
    # 1. Configuration variations - run 3 experiments with different hyperparameters
    # =================================================================
    configs = [
        {
            "hidden_dim": 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "n_samples": 1000,
            "n_features": 10,
            "noise": 10.0,
            "optimizer": "Adam",
            "loss": "mse",
            "model": "FeedForwardNN",
            "name": "small-network",
        },
        {
            "hidden_dim": 64,
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 50,
            "n_samples": 1000,
            "n_features": 10,
            "noise": 10.0,
            "optimizer": "Adam",
            "loss": "mse",
            "model": "FeedForwardNN",
            "name": "medium-network",
        },
        {
            "hidden_dim": 128,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "n_samples": 1000,
            "n_features": 10,
            "noise": 10.0,
            "optimizer": "Adam",
            "loss": "mse",
            "model": "FeedForwardNN",
            "name": "large-network",
        },
    ]

    # =================================================================
    # 2. Run experiments with different configurations
    # =================================================================
    for idx, config in enumerate(configs, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {idx}/3: {config['name']}")
        print(f"{'=' * 70}")

        print("\nConfiguration:")
        for key, value in config.items():
            if key != "name":
                print(f"  {key}: {value}")

        # =================================================================
        # 3. Initialize Artifacta run
        #    This automatically logs config and environment info
        # =================================================================
        run = ds.init(
            project="regression-demo",
            name=f"tensorflow-regression-{config['name']}",
            config={k: v for k, v in config.items() if k != "name"},
        )
        print("\n✓ Artifacta run initialized")

        # =================================================================
        # 4. Enable autolog for automatic checkpoint tracking
        #    This will log model checkpoints and metrics automatically
        # =================================================================
        ds.autolog(framework="tensorflow")

        # =================================================================
        # 5. Generate synthetic regression dataset
        # =================================================================
        x_train, x_test, y_train, y_test, scaler_x, scaler_y = create_synthetic_data(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            noise=config["noise"],
            random_state=42,
        )

        # =================================================================
        # 6. Create and compile model
        # =================================================================
        model = create_model(
            input_dim=config["n_features"],
            hidden_dim=config["hidden_dim"],
            learning_rate=config["learning_rate"],
        )

        # =================================================================
        # 7. Train the model
        #    Keras automatically tracks metrics during training
        # =================================================================
        print("\n✓ Training model...")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch size: {config['batch_size']}")
        print("-" * 70)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            verbose=1,  # Show progress bar
        )

        print("\n✓ Training complete!")

        # =================================================================
        # 8. Log training curves as Series
        #    Shows how loss decreases over epochs
        # =================================================================
        print("\n✓ Logging training metrics...")

        # Extract metrics from training history
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        train_mae = history.history["mae"]
        val_mae = history.history["val_mae"]

        # Log loss curves
        ds.log(
            "loss_curves",
            ds.Series(
                index="epoch",
                fields={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                index_values=list(range(1, len(train_loss) + 1)),
            ),
        )

        # Log MAE (Mean Absolute Error) curves
        ds.log(
            "mae_curves",
            ds.Series(
                index="epoch",
                fields={
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                },
                index_values=list(range(1, len(train_mae) + 1)),
            ),
        )

        # =================================================================
        # 9. Make predictions on test set
        # =================================================================
        print("\n✓ Generating predictions...")

        y_pred = model.predict(x_test, verbose=0).flatten()

        print(f"  Predictions generated for {len(y_pred)} test samples")

        # =================================================================
        # 10. Log predictions vs actual as Scatter plot
        #     Visualizes model performance - points should fall on diagonal
        # =================================================================
        print("\n✓ Logging prediction scatter plot...")

        # Create scatter plot data
        # Each point represents one test sample
        scatter_points = [
            {"x": float(actual), "y": float(pred), "label": "prediction"}
            for actual, pred in zip(y_test, y_pred)
        ]

        ds.log(
            "predictions_vs_actual",
            ds.Scatter(
                points=scatter_points,
                x_label="Actual Values",
                y_label="Predicted Values",
                metadata={
                    "description": "Predictions vs actual values on test set",
                    "ideal": "Points should fall on y=x diagonal for perfect predictions",
                },
            ),
        )

        # =================================================================
        # 11. Calculate and log residuals (prediction errors)
        #     Residual = Actual - Predicted
        #     Good model should have residuals centered around 0
        # =================================================================
        print("\n✓ Analyzing residuals...")

        residuals = y_test - y_pred

        # Calculate residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        residual_min = np.min(residuals)
        residual_max = np.max(residuals)

        print(f"  Residual mean: {residual_mean:.4f} (should be close to 0)")
        print(f"  Residual std:  {residual_std:.4f}")
        print(f"  Residual range: [{residual_min:.4f}, {residual_max:.4f}]")

        # Log residual distribution
        ds.log(
            "residual_distribution",
            ds.Distribution(
                values=residuals.tolist(),
                metadata={
                    "description": "Prediction errors (actual - predicted)",
                    "mean": float(residual_mean),
                    "std": float(residual_std),
                    "ideal": "Centered around 0 with small spread",
                },
            ),
        )

        # =================================================================
        # 12. Calculate final metrics
        # =================================================================
        print("\n✓ Calculating final metrics...")

        # Mean Squared Error
        mse = np.mean((y_test - y_pred) ** 2)
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        # Mean Absolute Error
        mae = np.mean(np.abs(y_test - y_pred))
        # R² Score (coefficient of determination)
        # R² = 1 means perfect predictions, R² = 0 means no better than mean baseline
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2_score:.4f}")

        # Log final metrics as Series (single point)
        ds.log(
            "final_metrics",
            ds.Series(
                index="metric",
                fields={
                    "value": [mse, rmse, mae, r2_score],
                },
                index_values=["MSE", "RMSE", "MAE", "R²"],
            ),
        )

        # =================================================================
        # 13. Save and log the trained model
        #     Keras models are saved in .keras format (recommended)
        # =================================================================
        print("\n✓ Saving model...")

        model_path = "regression_model.keras"
        model.save(model_path)

        # Log model as output artifact
        # Artifacta will automatically extract metadata
        run.log_output(
            model_path,
            name="trained_model",
            metadata={
                "framework": "tensorflow/keras",
                "model_type": "FeedForwardNN",
                "final_val_loss": float(val_loss[-1]),
                "final_val_mae": float(val_mae[-1]),
                "r2_score": float(r2_score),
                "test_rmse": float(rmse),
            },
        )

        # =================================================================
        # 14. Final summary
        # =================================================================
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print("Final Results:")
        print(f"  MSE:         {mse:.4f}")
        print(f"  RMSE:        {rmse:.4f}")
        print(f"  MAE:         {mae:.4f}")
        print(f"  R² Score:    {r2_score:.4f}")
        print("\nInterpretation:")
        print(f"  - R² = {r2_score:.4f} means the model explains {r2_score * 100:.1f}% of variance")
        print(f"  - Average prediction error (MAE): {mae:.4f} standard units")
        print("\n✓ All metrics and artifacts logged to Artifacta")
        print("  View your results in the Artifacta UI!")
        print("=" * 70)

        # Finish the run
        run.finish()

    # All experiments complete
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
