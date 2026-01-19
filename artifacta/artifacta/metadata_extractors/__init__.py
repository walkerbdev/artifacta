"""Metadata extractors for model files."""

from .torch import extract_pytorch_metadata

__all__ = [
    "extract_pytorch_metadata",
    "extract_metadata",
]


def extract_metadata(filepath: str) -> dict:
    """Automatically extract metadata from model file based on extension.

    Args:
        filepath: Path to model file

    Returns:
        dict with extracted metadata, or empty dict if extraction fails
    """
    filepath_lower = filepath.lower()

    if filepath_lower.endswith((".pt", ".pth", ".ckpt")):
        return extract_pytorch_metadata(filepath)

    # Unknown format or no extractor - return empty dict
    return {}
