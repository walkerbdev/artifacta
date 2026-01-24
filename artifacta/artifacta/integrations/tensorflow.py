"""TensorFlow/Keras autolog integration via monkey-patching.

This module implements automatic logging for TensorFlow/Keras by monkey-patching
the Model.fit() method to inject a custom callback. The callback automatically
logs parameters, metrics, model checkpoints, and final models.

What is logged:
    - Parameters: epochs, batch_size, optimizer name, optimizer config (lr, etc.)
    - Metrics: All metrics from logs dict (loss, accuracy, val_loss, etc.) per epoch
    - Checkpoints: Model checkpoints saved during training
    - Final model: Trained model saved at end of training

Architecture:
    1. enable_autolog() patches tf.keras.Model.fit to inject callback
    2. Patched fit() adds ArtifactaAutologCallback to callbacks list
    3. Callback hooks into training lifecycle to log params/metrics/models
    4. disable_autolog() restores original fit() method

Monkey-Patching Strategy:
    Why patch Model.fit() instead of using model.fit(callbacks=[...]):
        - User doesn't need to modify their code at all
        - Works with existing training scripts (zero friction)
        - Callback is injected automatically for all fit() calls
        - User can still pass their own callbacks, ours is appended

    Patch implementation:
        1. Save reference to original tf.keras.Model.fit in _ORIGINAL_FIT
        2. Define patched_fit() that:
           a. Creates ArtifactaCheckpointCallback instance
           b. Appends to callbacks list (or creates list if None)
           c. Calls original fit() with modified callbacks
        3. Replace tf.keras.Model.fit with patched_fit
        4. Set _AUTOLOG_ENABLED flag to prevent double-patching

Checkpoint Logging Flow:
    1. on_epoch_end() is called by Keras after each epoch
    2. Get current Artifacta run via get_run() (returns None if no active run)
    3. Create temporary file with epoch number in filename (.keras extension)
    4. Call self.model.save() to save complete model (architecture + weights)
    5. Log checkpoint as artifact with metadata:
       - artifact_type: "model_checkpoint"
       - framework: "tensorflow"
       - epoch: Current epoch number
    6. Cleanup temporary file (best effort, suppress exceptions)

Keras Callback Interface:
    - on_epoch_end(epoch, logs): Standard Keras callback method
    - epoch: Integer epoch number (0-indexed)
    - logs: Dictionary of metrics (e.g., loss, accuracy) - not currently used
    - self.model: Reference to Keras model being trained

Model Saving:
    - Uses model.save() which saves complete model in Keras format
    - .keras extension is the new standard format (replaces .h5)
    - Includes architecture, weights, optimizer state, training config
    - Can be loaded later with tf.keras.models.load_model()

Error Handling:
    - Import artifacta inside callback (avoids circular dependency)
    - Check if run is None (user might not have called artifacta.init())
    - Return early if no active run (fail silently, don't crash training)
    - Suppress exceptions during temp file cleanup (best effort)

State Management:
    - _AUTOLOG_ENABLED: Global flag tracking whether patching is active
    - _ORIGINAL_FIT: Reference to original fit() method for restoration
    - Both are module-level globals for persistence across calls

Comparison to PyTorch Lightning:
    - Similar patching strategy but targets different method (fit vs __init__)
    - Keras callbacks are simpler (no trainer object, just model reference)
    - Checkpoint format differs (.keras vs .ckpt)
    - Both use temporary files and cleanup strategy
"""

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


class ArtifactaAutologCallback(CallbackBase):
    """Auto-logs TensorFlow/Keras params, metrics, checkpoints, and models to Artifacta.

    Hooks into Keras training callbacks to automatically log:
    - Parameters: epochs, batch_size, optimizer config
    - Metrics: loss, accuracy, validation metrics (per epoch)
    - Checkpoints: Model checkpoints during training
    - Final model: Trained model at end
    """

    def __init__(self, epochs=None, batch_size=None, log_checkpoints=True, log_models=True):
        """Initialize callback.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            log_checkpoints: Whether to log model checkpoints during training
            log_models: Whether to log final trained model
        """
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_checkpoints = log_checkpoints
        self.log_models = log_models
        self.checkpoints_logged = []
        self._params_logged = False

    def on_train_begin(self, logs=None):
        """Log parameters when training begins."""
        from artifacta import get_run

        run = get_run()
        if run is None or self._params_logged:
            return

        # Build parameter dictionary
        params = {}
        if self.epochs is not None:
            params["epochs"] = self.epochs
        if self.batch_size is not None:
            params["batch_size"] = self.batch_size

        # Get optimizer info from model
        if hasattr(self.model, "optimizer") and self.model.optimizer is not None:
            optimizer = self.model.optimizer
            params["optimizer_name"] = optimizer.__class__.__name__

            # Extract optimizer config (lr, etc.)
            if hasattr(optimizer, "get_config"):
                config = optimizer.get_config()
                # Add common optimizer hyperparameters
                for key in ["learning_rate", "lr", "beta_1", "beta_2", "epsilon", "decay", "momentum"]:
                    if key in config:
                        params[key] = float(config[key]) if isinstance(config[key], (int, float)) else config[key]

        # Update run config with discovered parameters
        if params:
            run.update_config(params)
            self._params_logged = True

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics and checkpoints after each epoch."""
        from artifacta import get_run

        run = get_run()
        if run is None:
            return

        # Log metrics from logs dict as Series
        # logs contains: loss, accuracy, val_loss, val_accuracy, etc.
        if logs:
            # Convert metrics to Series format (epoch-indexed)
            series_data = {"index_values": [epoch]}
            for key, value in logs.items():
                series_data[key] = [float(value)]

            run.log("training_metrics", series_data)

        # Log checkpoint
        if self.log_checkpoints:
            with tempfile.NamedTemporaryFile(suffix=f"-epoch{epoch}.keras", delete=False) as tmp:
                checkpoint_path = tmp.name

            self.model.save(checkpoint_path)

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

            from contextlib import suppress
            with suppress(Exception):
                os.remove(checkpoint_path)

    def on_train_end(self, logs=None):
        """Log final trained model."""
        from artifacta import get_run

        run = get_run()
        if run is None or not self.log_models:
            return

        # Save final model
        with tempfile.NamedTemporaryFile(suffix="-final.keras", delete=False) as tmp:
            model_path = tmp.name

        self.model.save(model_path)

        run.log_artifact(
            name="model",
            path=model_path,
            include_content=False,
            metadata={
                "artifact_type": "model",
                "framework": "tensorflow",
            },
            role="output",
        )

        from contextlib import suppress
        with suppress(Exception):
            os.remove(model_path)


def enable_autolog(log_checkpoints: bool = True, log_models: bool = True):
    """Enable TensorFlow/Keras autolog.

    Automatically logs parameters, metrics, checkpoints, and models for all Model.fit() calls.

    Args:
        log_checkpoints: Whether to log model checkpoints during training
        log_models: Whether to log final trained model

    What is logged:
        - Parameters: epochs, batch_size, optimizer_name, learning_rate, etc.
        - Metrics: All metrics from logs dict (loss, val_loss, accuracy, etc.)
        - Checkpoints: Model checkpoints saved during training (if log_checkpoints=True)
        - Final model: Trained model at end of training (if log_models=True)
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

    def patched_fit(self, *args, callbacks=None, epochs=None, batch_size=None, **kwargs):
        """Inject Artifacta callback into fit()."""
        # Extract epochs and batch_size from kwargs if not in args
        if epochs is None and 'epochs' in kwargs:
            epochs = kwargs['epochs']
        if batch_size is None and 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']

        # Always inject callback - it checks for active run internally
        autolog_callback = ArtifactaAutologCallback(
            epochs=epochs,
            batch_size=batch_size,
            log_checkpoints=log_checkpoints,
            log_models=log_models
        )

        callbacks = [autolog_callback] if callbacks is None else list(callbacks) + [autolog_callback]

        # Call original fit with modified callbacks
        return _ORIGINAL_FIT(self, *args, callbacks=callbacks, epochs=epochs, batch_size=batch_size, **kwargs)

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
