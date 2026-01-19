"""TensorFlow/Keras autolog integration."""

import os
import tempfile

_AUTOLOG_ENABLED = False
_ORIGINAL_FIT = None


try:
    import tensorflow as tf

    CallbackBase = tf.keras.callbacks.Callback
except ImportError:
    # Fallback for when tensorflow not installed
    class CallbackBase:
        """Fallback callback base class when TensorFlow not installed."""

        pass


class ArtifactaCheckpointCallback(CallbackBase):
    """Auto-logs TensorFlow/Keras checkpoints to Artifacta.

    Hooks into Keras training callbacks to automatically
    upload model checkpoints as artifacts with rich metadata.
    """

    def __init__(self):
        """Initialize callback."""
        super().__init__()
        self.checkpoints_logged = []

    def on_epoch_end(self, epoch, logs=None):
        """Log checkpoint after each epoch."""
        from artifacta import get_run

        run = get_run()
        if run is None:
            return  # No active run

        # logs parameter required by Keras callback interface but not used
        _ = logs

        # Save checkpoint to temp file
        with tempfile.NamedTemporaryFile(suffix=f"-epoch{epoch}.keras", delete=False) as tmp:
            checkpoint_path = tmp.name

        # Save model
        self.model.save(checkpoint_path)

        # Log as artifact with metadata
        artifact_name = f"checkpoint_epoch{epoch}"
        run.log_artifact(
            name=artifact_name,
            path=checkpoint_path,
            include_content=False,
            metadata={
                "artifact_type": "model_checkpoint",
                "framework": "tensorflow",
                "epoch": epoch,
            },
            role="output",
        )

        self.checkpoints_logged.append({"epoch": epoch})

        # Cleanup temp file
        from contextlib import suppress

        with suppress(Exception):
            os.remove(checkpoint_path)


def enable_autolog(log_checkpoints: bool = True):
    """Enable TensorFlow/Keras autolog.

    Monkey-patches keras Model.fit() to inject Artifacta callbacks.
    """
    global _AUTOLOG_ENABLED, _ORIGINAL_FIT

    if _AUTOLOG_ENABLED:
        return  # Already enabled

    try:
        import tensorflow as tf
    except ImportError as err:
        raise ImportError(
            "tensorflow is not installed. Install with: pip install tensorflow"
        ) from err

    # Save original fit method
    _ORIGINAL_FIT = tf.keras.Model.fit

    def patched_fit(self, *args, callbacks=None, **kwargs):
        """Inject Artifacta callback into fit()."""
        # Always inject callback if checkpoints enabled
        # The callback itself checks for active run
        if log_checkpoints:
            ds_callback = ArtifactaCheckpointCallback()

            callbacks = [ds_callback] if callbacks is None else list(callbacks) + [ds_callback]

        # Call original fit with modified callbacks
        return _ORIGINAL_FIT(self, *args, callbacks=callbacks, **kwargs)

    # Replace Model.fit with our patched version
    tf.keras.Model.fit = patched_fit
    _AUTOLOG_ENABLED = True


def disable_autolog():
    """Disable TensorFlow/Keras autolog."""
    global _AUTOLOG_ENABLED, _ORIGINAL_FIT

    if not _AUTOLOG_ENABLED:
        return

    try:
        import tensorflow as tf

        # Restore original fit method
        if _ORIGINAL_FIT is not None:
            tf.keras.Model.fit = _ORIGINAL_FIT

        _AUTOLOG_ENABLED = False
    except ImportError:
        pass  # Already unloaded
