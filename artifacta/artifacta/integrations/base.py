"""Base callback interface for framework integrations."""

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
