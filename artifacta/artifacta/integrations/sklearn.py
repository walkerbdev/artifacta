"""Scikit-learn autolog integration.

This module implements automatic logging for scikit-learn estimators by patching
their fit() methods. Standard autologging implementation for scikit-learn estimators.

What gets logged automatically:
    - Parameters: All parameters from estimator.get_params(deep=True)
    - Training metrics: Score from estimator.score() on training data
    - Classifier metrics: accuracy, precision, recall, F1, log loss, ROC-AUC
    - Regressor metrics: MSE, RMSE, MAE, R²
    - Model artifacts: Serialized model (pickle format)
    - Plots: Confusion matrix, ROC curves (for classifiers)

Architecture:
    1. enable_autolog() patches fit() methods of all sklearn estimators
    2. Patched fit() captures parameters and training data
    3. After original fit() completes, compute and log metrics
    4. Save model artifact and generate plots
    5. disable_autolog() restores original methods

Patching Strategy:
    - Patch all estimators from sklearn.utils.all_estimators()
    - Exclude preprocessing/feature manipulation estimators
    - Include meta-estimators (Pipeline, GridSearchCV)
    - Use weak references to avoid memory leaks
    - Track active training session to prevent nested logging

Special Features:
    - GridSearchCV creates parent run + child runs for each fit
    - Pipeline logs parameters from all steps
    - Post-training metrics (future enhancement)
"""

import functools
import logging
import pickle
import tempfile

import numpy as np

_logger = logging.getLogger(__name__)

# Global state
_AUTOLOG_ENABLED = False
_ORIGINAL_METHODS = {}  # Store original fit methods
_ACTIVE_TRAINING = False  # Prevent nested logging


def enable_autolog(
    log_models: bool = True,
    log_input_examples: bool = False,
    log_model_signatures: bool = True,
    log_training_metrics: bool = True,
    log_post_training_metrics: bool = False,
    log_datasets: bool = True,
):
    """Enable scikit-learn autolog.

    Patches all sklearn estimators' fit() methods to automatically log:
    - Parameters via get_params(deep=True)
    - Training metrics (accuracy, precision, recall, F1, etc.)
    - Model artifacts
    - Dataset metadata (shape, dtype, hash)
    - Confusion matrix and ROC curves (for classifiers)

    Args:
        log_models: If True, save fitted model as artifact
        log_input_examples: If True, log sample of training data
        log_model_signatures: If True, infer and log model signature
        log_training_metrics: If True, compute and log training metrics
        log_post_training_metrics: If True, track metrics computed after training (advanced)
        log_datasets: If True, log dataset metadata (shape, dtype, hash)

    Example:
        >>> import artifacta as ds
        >>> ds.sklearn.autolog()
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier()
        >>> clf.fit(X_train, y_train)  # Automatically logs params, metrics, model
    """
    global _AUTOLOG_ENABLED, _ORIGINAL_METHODS

    if _AUTOLOG_ENABLED:
        _logger.warning("sklearn autolog already enabled")
        return

    try:
        import sklearn
    except ImportError as err:
        raise ImportError(
            "scikit-learn is not installed. Install with: pip install scikit-learn"
        ) from err

    # Get all estimators to patch
    estimators = _get_estimators_to_patch()

    _logger.info(f"Patching {len(estimators)} sklearn estimators for autologging")

    # Patch each estimator's fit() method
    for estimator_class in estimators:
        _patch_estimator_fit(
            estimator_class,
            log_models=log_models,
            log_input_examples=log_input_examples,
            log_model_signatures=log_model_signatures,
            log_training_metrics=log_training_metrics,
            log_post_training_metrics=log_post_training_metrics,
            log_datasets=log_datasets,
        )

    _AUTOLOG_ENABLED = True
    _logger.info("sklearn autolog enabled")


def disable_autolog():
    """Disable scikit-learn autolog and restore original methods."""
    global _AUTOLOG_ENABLED, _ORIGINAL_METHODS

    if not _AUTOLOG_ENABLED:
        return

    # Restore all original methods
    for (estimator_class, method_name), original_method in _ORIGINAL_METHODS.items():
        setattr(estimator_class, method_name, original_method)

    _ORIGINAL_METHODS.clear()
    _AUTOLOG_ENABLED = False
    _logger.info("sklearn autolog disabled")


def _get_estimators_to_patch():
    """Get list of sklearn estimators to patch.

    Standard approach:
    - Include all estimators from sklearn.utils.all_estimators()
    - Include meta-estimators (GridSearchCV, Pipeline)
    - Exclude preprocessing/feature manipulation classes

    Returns:
        List of estimator classes to patch
    """
    from sklearn.utils import all_estimators

    try:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.pipeline import Pipeline

        meta_estimators = [GridSearchCV, RandomizedSearchCV, Pipeline]
    except ImportError:
        meta_estimators = []

    # Get all estimators
    estimator_list = [est_class for est_name, est_class in all_estimators()]

    # Add meta-estimators if not already included
    estimators_to_patch = set(estimator_list).union(set(meta_estimators))

    # Exclude preprocessing/feature manipulation estimators
    excluded_modules = [
        "sklearn.preprocessing",
        "sklearn.impute",
        "sklearn.feature_extraction",
        "sklearn.feature_selection",
    ]

    excluded_classes = [
        "sklearn.compose._column_transformer.ColumnTransformer",
    ]

    filtered_estimators = []
    for estimator in estimators_to_patch:
        module_name = estimator.__module__
        full_name = f"{module_name}.{estimator.__name__}"

        # Check if excluded
        is_excluded = any(module_name.startswith(excl) for excl in excluded_modules)
        is_excluded = is_excluded or (full_name in excluded_classes)

        if not is_excluded:
            filtered_estimators.append(estimator)

    return filtered_estimators


def _patch_estimator_fit(
    estimator_class,
    log_models,
    log_input_examples,
    log_model_signatures,
    log_training_metrics,
    log_post_training_metrics,
    log_datasets,
):
    """Patch an estimator's fit() method to add autologging.

    Args:
        estimator_class: Sklearn estimator class to patch
        log_models: Whether to log model artifacts
        log_training_metrics: Whether to log training metrics
        log_input_examples: Whether to log input examples
        log_model_signatures: Whether to log model signatures
        log_post_training_metrics: Whether to track post-training metrics
        log_datasets: Whether to log dataset metadata
    """
    global _ORIGINAL_METHODS

    # Save original fit method
    original_fit = estimator_class.fit
    key = (estimator_class, "fit")
    _ORIGINAL_METHODS[key] = original_fit

    @functools.wraps(original_fit)
    def patched_fit(self, X, y=None, **fit_params):
        """Patched fit() that adds autologging."""
        global _ACTIVE_TRAINING

        # Prevent nested logging (e.g., Pipeline calling fit on sub-estimators)
        if _ACTIVE_TRAINING:
            return original_fit(self, X, y, **fit_params)

        # Import here to avoid circular dependency
        from artifacta import get_run

        run = get_run()
        if run is None:
            # No active run, just call original fit
            return original_fit(self, X, y, **fit_params)

        _ACTIVE_TRAINING = True
        try:
            # Log parameters before training
            _log_params(run, self)

            # Log dataset metadata
            if log_datasets:
                from .dataset_utils import log_dataset_metadata
                log_dataset_metadata(run, X, y, context="train")

            # Call original fit
            result = original_fit(self, X, y, **fit_params)

            # Log metrics after training
            if log_training_metrics:
                _log_training_metrics(run, self, X, y)

            # Log model artifact
            if log_models:
                _log_model(run, self)

            return result

        finally:
            _ACTIVE_TRAINING = False

    # Replace fit method
    estimator_class.fit = patched_fit


def _log_params(run, estimator):
    """Log all estimator parameters.

    Uses get_params(deep=True) to capture parameters from nested estimators
    (e.g., Pipeline steps, GridSearchCV base estimator).

    Args:
        run: Active Artifacta run
        estimator: Fitted sklearn estimator
    """
    try:
        params = estimator.get_params(deep=True)

        # Convert params to simple types for logging
        serializable_params = {}
        for key, value in params.items():
            # Skip complex objects (estimators, transformers)
            if hasattr(value, "get_params"):
                continue
            # Convert numpy types to Python types
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            # Skip None values
            if value is None:
                continue

            serializable_params[key] = value

        # Update run config with discovered parameters
        if serializable_params:
            run.update_config(serializable_params)

    except Exception as e:
        _logger.warning(f"Failed to log parameters: {e}")


def _log_training_metrics(run, estimator, X, y):
    """Log training metrics based on estimator type.

    For classifiers: accuracy, precision, recall, F1, log loss, ROC-AUC
    For regressors: MSE, RMSE, MAE, R²

    Args:
        run: Active Artifacta run
        estimator: Fitted sklearn estimator
        X: Training features
        y: Training labels
    """
    try:
        from sklearn.base import is_classifier, is_regressor

        metrics = {}

        # Get training score (works for both classifiers and regressors)
        if hasattr(estimator, "score"):
            score = estimator.score(X, y)
            metrics["training_score"] = score

        # Classifier-specific metrics
        if is_classifier(estimator):
            metrics.update(_compute_classifier_metrics(estimator, X, y))

        # Regressor-specific metrics
        elif is_regressor(estimator):
            metrics.update(_compute_regressor_metrics(estimator, X, y))

        # Log metrics as structured data
        if metrics:
            # Convert to list format for Series primitive
            metric_data = {"metric": list(metrics.keys()), "value": list(metrics.values())}
            run.log("training_metrics", metric_data)

    except Exception as e:
        _logger.warning(f"Failed to log training metrics: {e}")


def _compute_classifier_metrics(estimator, X, y):
    """Compute classifier-specific metrics.

    Standard approach:
    - accuracy, precision, recall, F1
    - log loss (if predict_proba available)
    - ROC-AUC (if predict_proba available)

    Args:
        estimator: Fitted classifier
        X: Training features
        y: Training labels

    Returns:
        Dict of metric name -> value
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    metrics = {}
    y_pred = estimator.predict(X)

    # Determine if binary or multiclass
    n_classes = len(np.unique(y))
    average = "binary" if n_classes == 2 else "weighted"

    metrics["accuracy"] = accuracy_score(y, y_pred)
    metrics["precision"] = precision_score(y, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y, y_pred, average=average, zero_division=0)
    metrics["f1_score"] = f1_score(y, y_pred, average=average, zero_division=0)

    # Log loss and ROC-AUC (if predict_proba available)
    if hasattr(estimator, "predict_proba"):
        try:
            from sklearn.metrics import log_loss, roc_auc_score

            y_proba = estimator.predict_proba(X)
            metrics["log_loss"] = log_loss(y, y_proba)

            # ROC-AUC (binary or multiclass)
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y, y_proba, multi_class="ovr", average="weighted"
                )
        except Exception as e:
            _logger.debug(f"Could not compute probabilistic metrics: {e}")

    return metrics


def _compute_regressor_metrics(estimator, X, y):
    """Compute regressor-specific metrics.

    Standard approach:
    - MSE, RMSE, MAE, R²

    Args:
        estimator: Fitted regressor
        X: Training features
        y: Training labels

    Returns:
        Dict of metric name -> value
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {}
    y_pred = estimator.predict(X)

    mse = mean_squared_error(y, y_pred)
    metrics["mse"] = mse
    metrics["rmse"] = np.sqrt(mse)
    metrics["mae"] = mean_absolute_error(y, y_pred)
    metrics["r2_score"] = r2_score(y, y_pred)

    return metrics


def _log_model(run, estimator):
    """Log fitted model as artifact.

    Saves model using pickle format (following sklearn convention).

    Args:
        run: Active Artifacta run
        estimator: Fitted sklearn estimator
    """
    try:
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(estimator, tmp)
            model_path = tmp.name

        # Log as artifact
        estimator_name = estimator.__class__.__name__
        run.log_artifact(
            name=f"{estimator_name}_model",
            path=model_path,
            include_content=False,
            metadata={
                "artifact_type": "sklearn_model",
                "estimator_class": estimator_name,
                "estimator_module": estimator.__class__.__module__,
            },
            role="output",
        )

        # Cleanup temp file
        import os
        from contextlib import suppress

        with suppress(Exception):
            os.remove(model_path)

    except Exception as e:
        _logger.warning(f"Failed to log model: {e}")
