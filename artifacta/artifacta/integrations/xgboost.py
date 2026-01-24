"""XGBoost autolog integration.

This module implements automatic logging for XGBoost models by patching
the xgboost.train() function and sklearn API.

What gets logged automatically:
    - Parameters: All parameters passed to xgboost.train()
    - Training metrics: Per-iteration metrics from evals (validation sets)
    - Feature importance: Multiple types (weight, gain, cover) as JSON + plots
    - Model artifacts: Trained booster in native XGBoost format
    - Best iteration metrics: If early stopping is used

Architecture:
    1. enable_autolog() patches xgboost.train() and sklearn API
    2. Patched train() injects callback for metrics logging
    3. After training, log feature importance and model
    4. Callback logs metrics at each iteration
    5. disable_autolog() restores original methods

Patching Strategy:
    - Patch xgboost.train() function
    - Patch sklearn API (XGBClassifier, XGBRegressor) fit() methods
    - Use callbacks for per-iteration metrics
    - Track active training to prevent nested logging
    - Sanitize metric names (@ → _at_)

Special Features:
    - Per-iteration metrics via callbacks
    - Feature importance plots (multiple types)
    - Early stopping detection (best iteration metrics)
    - Metric name sanitization (@ → _at_)
"""

import contextlib
import functools
import json
import logging
import pickle
import tempfile
from typing import List, Optional

import numpy as np

_logger = logging.getLogger(__name__)

# Global state
_AUTOLOG_ENABLED = False
_ORIGINAL_TRAIN = None
_ORIGINAL_SKLEARN_METHODS = {}  # Store original sklearn fit methods
_ACTIVE_TRAINING = False  # Prevent nested logging


def enable_autolog(
    log_models: bool = True,
    log_feature_importance: bool = True,
    importance_types: Optional[List[str]] = None,
    log_datasets: bool = True,
):
    """Enable XGBoost autolog.

    Patches xgboost.train() and sklearn API to automatically log:
    - Parameters
    - Per-iteration training metrics
    - Feature importance (weight, gain, cover)
    - Trained model
    - Dataset metadata (shape, dtype, hash)

    Args:
        log_models: If True, save trained booster as artifact
        log_feature_importance: If True, log feature importance as JSON
        importance_types: Feature importance types to log. Default: ["weight", "gain", "cover"]
        log_datasets: If True, log dataset metadata (requires XGBoost >= 1.7.0)

    Example:
        >>> import artifacta as ds
        >>> ds.xgboost.autolog()
        >>> import xgboost as xgb
        >>> dtrain = xgb.DMatrix(X_train, y_train)
        >>> dval = xgb.DMatrix(X_val, y_val)
        >>> params = {"max_depth": 3, "eta": 0.1}
        >>> booster = xgb.train(params, dtrain, evals=[(dval, "val")])
        >>> # Params, metrics, feature importance, model auto-logged
    """
    global _AUTOLOG_ENABLED, _ORIGINAL_TRAIN, _ORIGINAL_SKLEARN_METHODS

    if _AUTOLOG_ENABLED:
        _logger.warning("XGBoost autolog already enabled")
        return

    try:
        import xgboost as xgb
    except ImportError as err:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost"
        ) from err

    if importance_types is None:
        importance_types = ["weight", "gain", "cover"]

    # Patch native xgboost.train()
    _ORIGINAL_TRAIN = xgb.train
    xgb.train = _create_patched_train(
        xgb.train,
        log_models=log_models,
        log_feature_importance=log_feature_importance,
        importance_types=importance_types,
        log_datasets=log_datasets,
    )

    # Patch sklearn API
    _patch_sklearn_api(
        log_models=log_models,
        log_feature_importance=log_feature_importance,
        importance_types=importance_types,
        log_datasets=log_datasets,
    )

    _AUTOLOG_ENABLED = True
    _logger.info("XGBoost autolog enabled")


def disable_autolog():
    """Disable XGBoost autolog and restore original methods."""
    global _AUTOLOG_ENABLED, _ORIGINAL_TRAIN, _ORIGINAL_SKLEARN_METHODS

    if not _AUTOLOG_ENABLED:
        return

    try:
        import xgboost as xgb

        # Restore xgboost.train()
        if _ORIGINAL_TRAIN is not None:
            xgb.train = _ORIGINAL_TRAIN

        # Restore sklearn API
        for (cls, method_name), original_method in _ORIGINAL_SKLEARN_METHODS.items():
            setattr(cls, method_name, original_method)

        _ORIGINAL_SKLEARN_METHODS.clear()
        _AUTOLOG_ENABLED = False
        _logger.info("XGBoost autolog disabled")

    except ImportError:
        pass


def _create_patched_train(
    original_train,
    log_models,
    log_feature_importance,
    importance_types,
    log_datasets,
):
    """Create patched xgboost.train() function.

    Args:
        original_train: Original xgboost.train function
        log_models: Whether to log model artifacts
        log_feature_importance: Whether to log feature importance
        importance_types: Types of feature importance to log
        log_datasets: Whether to log dataset metadata

    Returns:
        Patched train function
    """

    @functools.wraps(original_train)
    def patched_train(params, dtrain, *args, evals=None, **kwargs):
        """Patched xgboost.train() that adds autologging."""
        global _ACTIVE_TRAINING

        # Prevent nested logging
        if _ACTIVE_TRAINING:
            return original_train(params, dtrain, *args, evals=evals, **kwargs)

        from artifacta import get_run

        run = get_run()
        if run is None:
            # No active run, just call original train
            return original_train(params, dtrain, *args, evals=evals, **kwargs)

        _ACTIVE_TRAINING = True
        try:
            # Log parameters
            _log_params(run, params)

            # Log dataset metadata (requires XGBoost >= 1.7.0)
            if log_datasets:
                _log_xgboost_datasets(run, dtrain, evals)

            # Create metrics logger
            metrics_history = []

            # Inject callback for metrics logging
            callbacks = kwargs.get("callbacks", [])
            if not isinstance(callbacks, list):
                callbacks = [callbacks] if callbacks is not None else []

            # Add metrics logging callback
            autolog_callback = _create_autolog_callback(run, metrics_history)
            callbacks.append(autolog_callback)
            kwargs["callbacks"] = callbacks

            # Call original train
            booster = original_train(params, dtrain, *args, evals=evals, **kwargs)

            # Log feature importance
            if log_feature_importance and booster is not None:
                _log_feature_importance(run, booster, importance_types)

            # Log model artifact
            if log_models and booster is not None:
                _log_model(run, booster)

            return booster

        finally:
            _ACTIVE_TRAINING = False

    return patched_train


def _patch_sklearn_api(log_models, log_feature_importance, importance_types, log_datasets):
    """Patch XGBoost sklearn API (XGBClassifier, XGBRegressor, XGBRanker).

    Args:
        log_models: Whether to log models
        log_feature_importance: Whether to log feature importance
        importance_types: Types of feature importance to log
        log_datasets: Whether to log dataset metadata
    """
    try:
        import xgboost as xgb

        # Get sklearn-compatible classes
        sklearn_classes = [
            xgb.XGBClassifier,
            xgb.XGBRegressor,
        ]

        # Also try XGBRanker if available
        with contextlib.suppress(AttributeError):
            sklearn_classes.append(xgb.XGBRanker)

        for sklearn_class in sklearn_classes:
            _patch_sklearn_fit(
                sklearn_class,
                log_models=log_models,
                log_feature_importance=log_feature_importance,
                importance_types=importance_types,
                log_datasets=log_datasets,
            )

    except ImportError:
        pass


def _patch_sklearn_fit(
    sklearn_class,
    log_models,
    log_feature_importance,
    importance_types,
    log_datasets,
):
    """Patch fit() method of XGBoost sklearn class.

    Args:
        sklearn_class: XGBoost sklearn class (XGBClassifier, XGBRegressor, etc.)
        log_models: Whether to log models
        log_feature_importance: Whether to log feature importance
        importance_types: Types of feature importance to log
        log_datasets: Whether to log dataset metadata
    """
    global _ORIGINAL_SKLEARN_METHODS

    # Save original fit
    original_fit = sklearn_class.fit
    key = (sklearn_class, "fit")
    _ORIGINAL_SKLEARN_METHODS[key] = original_fit

    @functools.wraps(original_fit)
    def patched_fit(self, X, y, **fit_params):
        """Patched fit() that adds autologging."""
        global _ACTIVE_TRAINING

        # Prevent nested logging
        if _ACTIVE_TRAINING:
            return original_fit(self, X, y, **fit_params)

        from artifacta import get_run

        run = get_run()
        if run is None:
            return original_fit(self, X, y, **fit_params)

        _ACTIVE_TRAINING = True
        try:
            # Log parameters (get_params includes hyperparameters)
            params = self.get_params(deep=True)
            _log_params(run, params)

            # Log dataset metadata
            if log_datasets:
                from .dataset_utils import log_dataset_metadata
                log_dataset_metadata(run, X, y, context="train")

            # Call original fit
            result = original_fit(self, X, y, **fit_params)

            # Get underlying booster
            booster = self.get_booster()

            # Log feature importance
            if log_feature_importance and booster is not None:
                _log_feature_importance(run, booster, importance_types)

            # Log model artifact
            if log_models:
                _log_sklearn_model(run, self)

            return result

        finally:
            _ACTIVE_TRAINING = False

    # Replace fit method
    sklearn_class.fit = patched_fit


def _create_autolog_callback(run, metrics_history):
    """Create callback for logging metrics at each iteration.

    Args:
        run: Active Artifacta run
        metrics_history: List to store metrics history

    Returns:
        Callback function compatible with XGBoost
    """
    import xgboost as xgb
    from packaging.version import Version

    # Check XGBoost version to determine callback API
    xgb_version = Version(xgb.__version__.replace("SNAPSHOT", "dev"))
    use_new_callback = xgb_version >= Version("1.3.0")

    if use_new_callback:
        # XGBoost >= 1.3.0: Use TrainingCallback class
        class AutologCallback(xgb.callback.TrainingCallback):
            """Callback for logging metrics at each iteration (XGBoost >= 1.3.0)."""

            def after_iteration(self, model, epoch, evals_log):
                """Called after each iteration.

                Args:
                    model: XGBoost booster
                    epoch: Current iteration number
                    evals_log: Dict of evaluation results
                        Format: {"eval_name": {"metric_name": [values...]}}

                Returns:
                    False to continue training
                """
                # Extract metrics from evals_log
                metrics = {}
                for eval_name, metric_dict in evals_log.items():
                    for metric_name, metric_values in metric_dict.items():
                        # Get latest value (last in list)
                        value = metric_values[-1]
                        # Sanitize metric name (@ → _at_)
                        sanitized_name = metric_name.replace("@", "_at_")
                        key = f"{eval_name}_{sanitized_name}"
                        metrics[key] = value

                # Log metrics for this iteration
                if metrics:
                    metrics["iteration"] = epoch
                    metrics_history.append(metrics)

                    # Log to Artifacta as series data
                    try:
                        _log_iteration_metrics(run, metrics_history)
                    except Exception as e:
                        _logger.warning(f"Failed to log metrics: {e}")

                return False  # Continue training

        return AutologCallback()

    else:
        # XGBoost < 1.3.0: Use function callback
        def autolog_callback_fn(env):
            """Callback for logging metrics (XGBoost < 1.3.0).

            Args:
                env: XGBoost callback environment with evaluation_result_list
            """
            # Extract metrics from evaluation results
            metrics = {}
            for eval_result in env.evaluation_result_list:
                # eval_result is tuple: (eval_name, metric_name, value, is_higher_better)
                eval_name = eval_result[0]
                metric_name = eval_result[1]
                value = eval_result[2]

                # Sanitize metric name
                sanitized_name = metric_name.replace("@", "_at_")
                key = f"{eval_name}_{sanitized_name}"
                metrics[key] = value

            # Log metrics
            if metrics:
                metrics["iteration"] = env.iteration
                metrics_history.append(metrics)

                try:
                    _log_iteration_metrics(run, metrics_history)
                except Exception as e:
                    _logger.warning(f"Failed to log metrics: {e}")

        return autolog_callback_fn


def _log_params(run, params):
    """Log XGBoost parameters.

    Args:
        run: Active Artifacta run
        params: Dictionary of XGBoost parameters
    """
    try:
        # Convert params to simple types
        serializable_params = {}
        for key, value in params.items():
            # Skip complex objects
            if hasattr(value, "__dict__") and not isinstance(value, (int, float, str, bool)):
                continue
            # Convert numpy types
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()

            serializable_params[key] = value

        if serializable_params:
            run.update_config(serializable_params)

    except Exception as e:
        _logger.warning(f"Failed to log parameters: {e}")


def _log_iteration_metrics(run, metrics_history):
    """Log metrics from all iterations.

    Args:
        run: Active Artifacta run
        metrics_history: List of metric dicts from each iteration
    """
    if not metrics_history:
        return

    # Convert to series format: {"iteration": [...], "metric1": [...], "metric2": [...]}
    series_data = {}

    for metrics in metrics_history:
        for key, value in metrics.items():
            if key not in series_data:
                series_data[key] = []
            series_data[key].append(value)

    # Log as series
    run.log("xgboost_metrics", series_data)


def _log_xgboost_datasets(run, dtrain, evals):
    """Log dataset metadata from XGBoost DMatrix objects.

    Uses DMatrix.get_data() method (XGBoost >= 1.7.0) to retrieve original data.

    Args:
        run: Active Artifacta run
        dtrain: Training DMatrix
        evals: List of (DMatrix, name) tuples for evaluation sets
    """
    import xgboost as xgb
    from packaging.version import Version

    # Check XGBoost version
    if Version(xgb.__version__) < Version("1.7.0"):
        _logger.warning(
            "Dataset logging requires XGBoost >= 1.7.0. "
            f"Current version: {xgb.__version__}. Skipping dataset logging."
        )
        return

    try:
        from .dataset_utils import log_dataset_metadata

        # Log training dataset
        try:
            train_data = dtrain.get_data()
            log_dataset_metadata(run, train_data, context="train")
        except Exception as e:
            _logger.warning(f"Failed to log training dataset: {e}")

        # Log evaluation datasets
        if evals:
            for deval, eval_name in evals:
                try:
                    eval_data = deval.get_data()
                    log_dataset_metadata(run, eval_data, context=f"eval_{eval_name}")
                except Exception as e:
                    _logger.warning(f"Failed to log eval dataset '{eval_name}': {e}")

    except Exception as e:
        _logger.warning(f"Failed to log datasets: {e}")


def _log_feature_importance(run, booster, importance_types):
    """Log feature importance as JSON.

    Args:
        run: Active Artifacta run
        booster: Trained XGBoost booster
        importance_types: List of importance types to log
    """
    try:
        for importance_type in importance_types:
            # Get importance scores
            importance_dict = booster.get_score(importance_type=importance_type)

            if importance_dict:
                # Convert to JSON
                importance_json = json.dumps(importance_dict, indent=2)

                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=f"_importance_{importance_type}.json",
                    delete=False,
                ) as tmp:
                    tmp.write(importance_json)
                    tmp_path = tmp.name

                # Log as artifact
                run.log_artifact(
                    name=f"feature_importance_{importance_type}",
                    path=tmp_path,
                    include_content=True,
                    metadata={
                        "artifact_type": "feature_importance",
                        "importance_type": importance_type,
                    },
                    role="output",
                )

                # Cleanup
                import os
                from contextlib import suppress

                with suppress(Exception):
                    os.remove(tmp_path)

    except Exception as e:
        _logger.warning(f"Failed to log feature importance: {e}")


def _log_model(run, booster):
    """Log XGBoost booster as artifact.

    Args:
        run: Active Artifacta run
        booster: Trained XGBoost booster
    """
    try:
        # Save model to temp file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            booster.save_model(tmp.name)
            model_path = tmp.name

        # Log as artifact
        run.log_artifact(
            name="xgboost_model",
            path=model_path,
            include_content=False,
            metadata={
                "artifact_type": "xgboost_model",
                "model_format": "json",
            },
            role="output",
        )

        # Cleanup
        import os
        from contextlib import suppress

        with suppress(Exception):
            os.remove(model_path)

    except Exception as e:
        _logger.warning(f"Failed to log model: {e}")


def _log_sklearn_model(run, model):
    """Log XGBoost sklearn model as artifact.

    Args:
        run: Active Artifacta run
        model: Trained XGBoost sklearn model
    """
    try:
        # Save model to temp file (pickle format for sklearn compatibility)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(model, tmp)
            model_path = tmp.name

        # Log as artifact
        model_class = model.__class__.__name__
        run.log_artifact(
            name=f"{model_class}_model",
            path=model_path,
            include_content=False,
            metadata={
                "artifact_type": "xgboost_sklearn_model",
                "model_class": model_class,
            },
            role="output",
        )

        # Cleanup
        import os
        from contextlib import suppress

        with suppress(Exception):
            os.remove(model_path)

    except Exception as e:
        _logger.warning(f"Failed to log sklearn model: {e}")
