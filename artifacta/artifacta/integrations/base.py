"""Base callback interface for framework integrations.

This module defines the abstract base class for framework-specific callbacks.
It provides a common interface that all integration modules (PyTorch Lightning,
TensorFlow, etc.) can extend to implement checkpoint logging and other hooks.

Architecture:
    - ArtifactaCallback: Abstract base class with common interface
    - Concrete implementations: In pytorch_lightning.py, tensorflow.py, etc.
    - Each concrete class overrides on_checkpoint_save() with framework-specific logic

Why abstract base class:
    - Enforces consistent interface across all framework integrations
    - Allows type checking and polymorphism
    - Documents expected callback methods
    - Makes it easy to add new hooks (on_train_start, on_train_end, etc.)

Future extensions:
    Could add more hook methods:
    - on_train_start(self): Called at training start
    - on_train_end(self): Called at training end
    - on_batch_end(self, batch_metrics): Called after each batch
    - on_metric_log(self, metrics): Called when metrics are logged
"""

from abc import ABC, abstractmethod


class ArtifactaCallback(ABC):
    """Base class for Artifacta framework callbacks."""

    def __init__(self, run):
        """Initialize callback.

        Args:
            run: Artifacta run object.
        """
        self.run = run  # Artifacta run object

    @abstractmethod
    def on_checkpoint_save(self, filepath: str, metrics: dict):
        """Called when a checkpoint is saved."""
        pass
