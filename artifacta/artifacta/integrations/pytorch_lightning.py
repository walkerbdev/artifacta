"""PyTorch Lightning autolog integration via monkey-patching.

This module implements automatic logging for PyTorch Lightning by monkey-patching
the Trainer.__init__ method to inject a custom callback. The callback automatically
logs parameters, metrics, model checkpoints, and final models.

What is logged:
    - Parameters: max_epochs, optimizer name, optimizer hyperparameters (lr, etc.)
    - Metrics: All metrics in trainer.callback_metrics (loss, accuracy, val_loss, etc.)
    - Checkpoints: Model checkpoints saved during training
    - Final model: Trained model saved at end of training

Architecture:
    1. enable_autolog() patches pl.Trainer.__init__ to inject callback
    2. Patched __init__ adds ArtifactaAutologCallback to callbacks list
    3. Callback hooks into training lifecycle to log params/metrics/models
    4. disable_autolog() restores original __init__ method

Monkey-Patching Strategy:
    Why patch __init__ instead of using Trainer(callbacks=[...]):
        - User doesn't need to modify their code at all
        - Works with existing training scripts (zero friction)
        - Callback is injected automatically for all Trainer instances
        - User can still pass their own callbacks, ours is appended

    Patch implementation:
        1. Save reference to original pl.Trainer.__init__ in _ORIGINAL_TRAINER_INIT
        2. Define patched_init() that:
           a. Creates ArtifactaCheckpointCallback instance
           b. Appends to callbacks list (or creates list if None)
           c. Calls original __init__ with modified callbacks
        3. Replace pl.Trainer.__init__ with patched_init
        4. Set _CALLBACK_INJECTED flag to prevent double-patching

Checkpoint Logging Flow:
    1. on_train_epoch_end() is called by PyTorch Lightning after each epoch
    2. Get current Artifacta run via get_run() (returns None if no active run)
    3. Create temporary file with epoch number in filename
    4. Call trainer.save_checkpoint() to save model state
    5. Log checkpoint as artifact with metadata:
       - artifact_type: "model_checkpoint"
       - framework: "pytorch_lightning"
       - epoch: Current epoch number
       - global_step: Total training steps
       - model_class: Name of LightningModule class
    6. Cleanup temporary file (best effort, suppress exceptions)

Temporary File Strategy:
    We use tempfile.NamedTemporaryFile instead of fixed paths because:
    - Avoids conflicts between concurrent runs
    - Automatic cleanup on most platforms
    - No need to invent unique filenames
    - Manual cleanup at end ensures no temp file leaks

Error Handling:
    - Import artifacta inside callback (avoids circular dependency)
    - Check if run is None (user might not have called artifacta.init())
    - Return early if no active run (fail silently, don't crash training)
    - Suppress exceptions during temp file cleanup (best effort)

State Management:
    - _CALLBACK_INJECTED: Global flag tracking whether patching is active
    - _ORIGINAL_TRAINER_INIT: Reference to original __init__ for restoration
    - Both are module-level globals for persistence across calls

Why daemon pattern (not using Python's built-in callback registration):
    PyTorch Lightning doesn't have a global callback registry that applies
    to all Trainer instances. Patching __init__ is the cleanest way to
    inject callbacks universally without requiring user code changes.
"""

import os
import tempfile

_CALLBACK_INJECTED = False
_ORIGINAL_TRAINER_INIT = None


try:
    import pytorch_lightning as pl

    CallbackBase = pl.Callback
except ImportError:
    # Fallback for when pytorch_lightning not installed
    class CallbackBase:
        """Fallback callback base class when PyTorch Lightning not installed."""

        pass


def _get_optimizer_name(optimizer):
    """Get optimizer class name, handling LightningOptimizer wrapper."""
    try:
        import pytorch_lightning as pl
        from packaging.version import Version

        if Version(pl.__version__) >= Version("1.1.0"):
            from pytorch_lightning.core.optimizer import LightningOptimizer
            if isinstance(optimizer, LightningOptimizer):
                return optimizer._optimizer.__class__.__name__
    except (ImportError, AttributeError):
        pass

    return optimizer.__class__.__name__


class ArtifactaAutologCallback(CallbackBase):
    """Auto-logs PyTorch Lightning params, metrics, checkpoints, and models to Artifacta.

    Hooks into PyTorch Lightning's training loop to automatically log:
    - Parameters: epochs, optimizer config
    - Metrics: loss, accuracy, validation metrics (per epoch)
    - Checkpoints: Model checkpoints during training
    - Final model: Trained model at end
    """

    def __init__(self, log_checkpoints=True, log_models=True):
        """Initialize callback.

        Args:
            log_checkpoints: Whether to log model checkpoints during training
            log_models: Whether to log final trained model
        """
        self.log_checkpoints = log_checkpoints
        self.log_models = log_models
        self.checkpoints_logged = []
        self._params_logged = False

    def on_train_start(self, trainer, pl_module):
        """Log parameters when training begins."""
        from artifacta import get_run

        run = get_run()
        if run is None or self._params_logged:
            return

        # Build parameter dictionary
        params = {"epochs": trainer.max_epochs}

        # Add optimizer info (first optimizer if multiple)
        if hasattr(trainer, "optimizers") and trainer.optimizers:
            optimizer = trainer.optimizers[0]
            params["optimizer_name"] = _get_optimizer_name(optimizer)

            # Add optimizer hyperparameters (lr, weight_decay, etc.)
            if hasattr(optimizer, "defaults"):
                params.update(optimizer.defaults)

        # Update run config with discovered parameters
        run.update_config(params)
        self._params_logged = True

    def on_train_epoch_end(self, trainer, pl_module):
        """Log metrics and checkpoints after each epoch."""
        from artifacta import get_run

        run = get_run()
        if run is None:
            return

        # Log metrics from trainer.callback_metrics as Series
        # This includes loss, accuracy, val_loss, val_accuracy, etc.
        if trainer.callback_metrics:
            # Convert metrics to Series format (epoch-indexed)
            series_data = {"index_values": [pl_module.current_epoch]}
            for key, value in trainer.callback_metrics.items():
                series_data[key] = [float(value)]

            run.log("training_metrics", series_data)

        # Log checkpoint
        if self.log_checkpoints:
            with tempfile.NamedTemporaryFile(
                suffix=f"-epoch{trainer.current_epoch}.ckpt", delete=False
            ) as tmp:
                checkpoint_path = tmp.name

            trainer.save_checkpoint(checkpoint_path)

            artifact_name = f"checkpoint_epoch{trainer.current_epoch}"
            run.log_artifact(
                name=artifact_name,
                path=checkpoint_path,
                include_content=False,
                metadata={
                    "artifact_type": "model_checkpoint",
                    "framework": "pytorch_lightning",
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "model_class": pl_module.__class__.__name__,
                },
                role="output",
            )

            self.checkpoints_logged.append({"epoch": trainer.current_epoch})

            from contextlib import suppress
            with suppress(Exception):
                os.remove(checkpoint_path)

    def on_train_end(self, trainer, pl_module):
        """Log final trained model."""
        from artifacta import get_run

        run = get_run()
        if run is None or not self.log_models:
            return

        # Save final model
        with tempfile.NamedTemporaryFile(suffix="-final.ckpt", delete=False) as tmp:
            model_path = tmp.name

        trainer.save_checkpoint(model_path)

        run.log_artifact(
            name="model",
            path=model_path,
            include_content=False,
            metadata={
                "artifact_type": "model",
                "framework": "pytorch_lightning",
                "model_class": pl_module.__class__.__name__,
            },
            role="output",
        )

        from contextlib import suppress
        with suppress(Exception):
            os.remove(model_path)


def enable_autolog(log_checkpoints: bool = True, log_models: bool = True):
    """Enable PyTorch Lightning autolog.

    Automatically logs parameters, metrics, checkpoints, and models for all Trainer.fit() calls.

    Args:
        log_checkpoints: Whether to log model checkpoints during training
        log_models: Whether to log final trained model

    What is logged:
        - Parameters: max_epochs, optimizer_name, learning rate, etc.
        - Metrics: All metrics in trainer.callback_metrics (loss, val_loss, accuracy, etc.)
        - Checkpoints: Model checkpoints saved during training (if log_checkpoints=True)
        - Final model: Trained model at end of training (if log_models=True)
    """
    global _CALLBACK_INJECTED, _ORIGINAL_TRAINER_INIT

    if _CALLBACK_INJECTED:
        return  # Already enabled

    try:
        import pytorch_lightning as pl
    except ImportError as err:
        raise ImportError(
            "pytorch-lightning is not installed. Install with: pip install pytorch-lightning"
        ) from err

    # Save original __init__
    _ORIGINAL_TRAINER_INIT = pl.Trainer.__init__

    def patched_init(self, *args, callbacks=None, **kwargs):
        """Inject Artifacta callback into trainer."""
        # Always inject callback - it checks for active run internally
        autolog_callback = ArtifactaAutologCallback(
            log_checkpoints=log_checkpoints,
            log_models=log_models
        )

        callbacks = [autolog_callback] if callbacks is None else list(callbacks) + [autolog_callback]

        # Call original init with modified callbacks
        _ORIGINAL_TRAINER_INIT(self, *args, callbacks=callbacks, **kwargs)

    # Replace Trainer.__init__ with our patched version
    pl.Trainer.__init__ = patched_init
    _CALLBACK_INJECTED = True


def disable_autolog():
    """Disable PyTorch Lightning autolog."""
    global _CALLBACK_INJECTED, _ORIGINAL_TRAINER_INIT

    if not _CALLBACK_INJECTED:
        return

    try:
        import pytorch_lightning as pl

        # Restore original __init__
        if _ORIGINAL_TRAINER_INIT is not None:
            pl.Trainer.__init__ = _ORIGINAL_TRAINER_INIT

        _CALLBACK_INJECTED = False
    except ImportError:
        pass  # Already unloaded
