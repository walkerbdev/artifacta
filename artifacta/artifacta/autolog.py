"""Automatic logging integration for ML frameworks.

This module provides a unified autolog() interface that automatically integrates
Artifacta with popular ML frameworks (scikit-learn, XGBoost, LightGBM, PyTorch Lightning, TensorFlow/Keras).
Once enabled, Artifacta automatically captures model checkpoints, parameters, metrics, and
framework-specific metadata without requiring explicit logging calls in user code.

Architecture:
    The module acts as a facade/dispatcher:

    1. User calls autolog() with optional framework parameter
    2. If framework=None, auto-detect via import checks
    3. Dispatch to framework-specific integration module
    4. Integration module patches framework classes (callbacks, hooks)
    5. Track enabled state in _AUTOLOG_ENABLED global dict

Auto-Detection Algorithm:
    The _detect_framework() function uses import-based detection:
    1. Try import sklearn -> return "sklearn"
    2. Try import xgboost -> return "xgboost"
    3. Try import lightgbm -> return "lightgbm"
    4. Try import pytorch_lightning -> return "pytorch"
    5. Try import tensorflow -> return "tensorflow"
    6. If all fail -> raise RuntimeError

    Why this order:
        - Traditional ML frameworks (sklearn, xgboost, lightgbm) checked first (most common)
        - PyTorch Lightning checked after (more specific than PyTorch)
        - TensorFlow checked last
        - Import-based detection is fast and reliable (no version parsing)

Integration Strategy (per framework):

    Scikit-learn:
        - Patches fit() methods of all sklearn estimators
        - Logs parameters via get_params(deep=True)
        - Computes and logs training metrics (accuracy, precision, recall, F1, etc.)
        - Saves fitted model as pickle artifact

    XGBoost:
        - Patches xgboost.train() and sklearn API
        - Uses callbacks to log metrics per iteration
        - Logs feature importance plots
        - Handles early stopping

    LightGBM:
        - Patches lightgbm.train() and sklearn API
        - Uses callbacks to log metrics per iteration
        - Logs feature importance plots
        - Handles early stopping

    PyTorch Lightning:
        - Registers a global callback in CALLBACK_REGISTRY
        - Callback hooks into on_save_checkpoint()
        - Automatically logs checkpoint file as artifact
        - Includes metadata: epoch, global_step, checkpoint path

    TensorFlow/Keras:
        - Patches tf.keras.callbacks.ModelCheckpoint class
        - Wraps on_epoch_end() to intercept checkpoint saves
        - Automatically logs checkpoint file as artifact
        - Includes metadata: epoch, model architecture, optimizer config

State Management:
    - _AUTOLOG_ENABLED: Global dict tracking which frameworks are enabled
    - Prevents duplicate patching (idempotent)
    - Allows selective disabling via disable()

Why separate integration modules:
    - Keeps framework dependencies optional (import only when needed)
    - Each framework has unique patching strategy
    - Easy to add new frameworks without modifying core autolog
    - Allows framework-specific configuration options

Design Philosophy:
    - Zero-friction: One call (autolog()) enables all automatic logging
    - Framework-agnostic: Same API works across different ML frameworks
    - Non-invasive: No changes to user training code required
    - Fail-safe: Missing framework dependencies raise clear errors
    - Reversible: disable() removes all patches and restores original behavior
"""

from typing import Optional

_AUTOLOG_ENABLED = {}


def autolog(
    framework: Optional[str] = None,
    log_models: bool = True,
    log_metrics: bool = True,
    log_params: bool = True,
):
    """Enable automatic logging for ML frameworks.

    Automatically logs parameters, metrics, and models during training.
    Works with scikit-learn, XGBoost, LightGBM, PyTorch Lightning, and TensorFlow/Keras.

    Args:
        framework: Framework name ("sklearn", "xgboost", "lightgbm", "pytorch", "tensorflow")
                  or None for auto-detection
        log_models: Automatically log trained models as artifacts
        log_metrics: Automatically log training metrics
        log_params: Automatically log model parameters/hyperparameters

    Example:
        >>> import artifacta as ds
        >>> ds.autolog()  # Auto-detect framework
        >>>
        >>> # Scikit-learn
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier()
        >>> clf.fit(X_train, y_train)  # Params, metrics, model auto-logged
        >>>
        >>> # PyTorch Lightning
        >>> trainer = pl.Trainer(...)
        >>> trainer.fit(model)  # Checkpoints auto-logged every epoch
        >>>
        >>> # TensorFlow/Keras
        >>> model.fit(X_train, y_train)  # Checkpoints auto-logged every epoch
    """
    global _AUTOLOG_ENABLED

    # Auto-detect framework if not specified
    if framework is None:
        framework = _detect_framework()

    if framework == "sklearn":
        from artifacta.integrations import sklearn

        sklearn.enable_autolog(
            log_models=log_models,
            log_training_metrics=log_metrics,
        )
        _AUTOLOG_ENABLED["sklearn"] = True

    elif framework == "xgboost":
        from artifacta.integrations import xgboost

        xgboost.enable_autolog(log_models=log_models)
        _AUTOLOG_ENABLED["xgboost"] = True

    elif framework == "lightgbm":
        # TODO: Implement LightGBM autolog
        raise NotImplementedError("LightGBM autolog coming soon")

    elif framework == "pytorch":
        from artifacta.integrations import pytorch_lightning

        pytorch_lightning.enable_autolog(log_checkpoints=log_models)
        _AUTOLOG_ENABLED["pytorch"] = True

    elif framework == "tensorflow":
        from artifacta.integrations import tensorflow

        tensorflow.enable_autolog(log_checkpoints=log_models)
        _AUTOLOG_ENABLED["tensorflow"] = True

    else:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported: 'sklearn', 'xgboost', 'lightgbm', 'pytorch', 'tensorflow'"
        )

    print(f"Artifacta autolog enabled for {framework}")


def disable():
    """Disable autolog for all frameworks."""
    global _AUTOLOG_ENABLED

    if "sklearn" in _AUTOLOG_ENABLED:
        from artifacta.integrations import sklearn

        sklearn.disable_autolog()

    if "xgboost" in _AUTOLOG_ENABLED:
        from artifacta.integrations import xgboost

        xgboost.disable_autolog()

    if "lightgbm" in _AUTOLOG_ENABLED:
        # TODO: Implement LightGBM disable
        pass

    if "pytorch" in _AUTOLOG_ENABLED:
        from artifacta.integrations import pytorch_lightning

        pytorch_lightning.disable_autolog()

    if "tensorflow" in _AUTOLOG_ENABLED:
        from artifacta.integrations import tensorflow

        tensorflow.disable_autolog()

    _AUTOLOG_ENABLED = {}
    print("Artifacta autolog disabled")


def _detect_framework():
    """Auto-detect which ML framework is installed.

    Priority order: sklearn, xgboost, lightgbm, pytorch, tensorflow
    """
    try:
        import sklearn  # noqa: F401

        return "sklearn"
    except ImportError:
        pass

    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except ImportError:
        pass

    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass

    try:
        import pytorch_lightning  # noqa: F401

        return "pytorch"
    except ImportError:
        pass

    try:
        import tensorflow  # noqa: F401

        return "tensorflow"
    except ImportError:
        pass

    raise RuntimeError(
        "Could not detect ML framework. "
        "Install scikit-learn, xgboost, lightgbm, pytorch-lightning, or tensorflow, "
        "or specify framework explicitly."
    )
