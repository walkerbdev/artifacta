"""Automatic logging for ML frameworks.

Enables automatic checkpoint logging and metadata extraction for
PyTorch Lightning, TensorFlow/Keras, and other ML frameworks.
"""

from typing import Optional

_AUTOLOG_ENABLED = {}


def autolog(
    framework: Optional[str] = None,
    log_checkpoints: bool = True,
):
    """Enable automatic logging for ML frameworks.

    Automatically logs model checkpoints as artifacts during training.
    All checkpoints are logged with metadata (epoch, step, framework info).

    Args:
        framework: "pytorch", "tensorflow", or None (auto-detect)
        log_checkpoints: Automatically log model checkpoints

    Example:
        >>> import artifacta as ds
        >>> ds.autolog()  # Auto-detect framework and log checkpoints
        >>>
        >>> # Disable checkpoint logging
        >>> ds.autolog(log_checkpoints=False)
        >>>
        >>> # PyTorch Lightning
        >>> trainer = pl.Trainer(...)
        >>> trainer.fit(model)  # Checkpoints auto-logged every epoch
        >>>
        >>> # TensorFlow/Keras
        >>> model.fit(X_train, y_train)  # Checkpoints auto-logged every epoch

    Note:
        Autolog only captures checkpoints. Use ds.log() to log metrics for visualization.
    """
    global _AUTOLOG_ENABLED

    # Auto-detect framework if not specified
    if framework is None:
        framework = _detect_framework()

    if framework == "pytorch":
        from artifacta.integrations import pytorch_lightning

        pytorch_lightning.enable_autolog(log_checkpoints=log_checkpoints)
        _AUTOLOG_ENABLED["pytorch"] = True

    elif framework == "tensorflow":
        from artifacta.integrations import tensorflow

        tensorflow.enable_autolog(log_checkpoints=log_checkpoints)
        _AUTOLOG_ENABLED["tensorflow"] = True

    else:
        raise ValueError(f"Unsupported framework: {framework}. Supported: 'pytorch', 'tensorflow'")

    print(f"✓ Artifacta autolog enabled for {framework}")


def disable():
    """Disable autolog for all frameworks."""
    global _AUTOLOG_ENABLED

    if "pytorch" in _AUTOLOG_ENABLED:
        from artifacta.integrations import pytorch_lightning

        pytorch_lightning.disable_autolog()

    if "tensorflow" in _AUTOLOG_ENABLED:
        from artifacta.integrations import tensorflow

        tensorflow.disable_autolog()

    _AUTOLOG_ENABLED = {}
    print("✓ Artifacta autolog disabled")


def _detect_framework():
    """Auto-detect which ML framework is installed."""
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
        "Install pytorch-lightning or tensorflow, or specify framework explicitly."
    )
