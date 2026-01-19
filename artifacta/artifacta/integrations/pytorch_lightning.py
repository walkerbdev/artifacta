"""PyTorch Lightning autolog integration."""

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


class ArtifactaCheckpointCallback(CallbackBase):
    """Auto-logs PyTorch Lightning checkpoints to Artifacta.

    Hooks into PyTorch Lightning's training loop to automatically
    upload model checkpoints as artifacts with rich metadata.
    """

    def __init__(self):
        """Initialize callback."""
        self.checkpoints_logged = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Log checkpoint after each epoch."""
        # Import here to avoid circular dependency
        from artifacta import get_run

        run = get_run()
        if run is None:
            return  # No active run

        # Save checkpoint to temp file
        with tempfile.NamedTemporaryFile(
            suffix=f"-epoch{trainer.current_epoch}.ckpt", delete=False
        ) as tmp:
            checkpoint_path = tmp.name

        trainer.save_checkpoint(checkpoint_path)

        # Log as artifact with metadata
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

        # Cleanup temp file
        from contextlib import suppress

        with suppress(Exception):
            os.remove(checkpoint_path)


def enable_autolog(log_checkpoints: bool = True):
    """Enable PyTorch Lightning autolog.

    Injects Artifacta callbacks into all future Trainer instances.
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
        # Always inject callback if checkpoints enabled
        # The callback itself checks for active run
        if log_checkpoints:
            ds_callback = ArtifactaCheckpointCallback()

            callbacks = [ds_callback] if callbacks is None else list(callbacks) + [ds_callback]

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
