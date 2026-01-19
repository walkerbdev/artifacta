"""PyTorch model metadata extractor."""

import os


def extract_pytorch_metadata(filepath: str) -> dict:
    """Extract metadata from PyTorch checkpoint file (.pt, .pth, .ckpt).

    Args:
        filepath: Path to PyTorch checkpoint file

    Returns:
        dict with extracted metadata (parameter count, file size, layers, etc.)
    """
    try:
        import torch
    except ImportError:
        return {}

    if not os.path.exists(filepath):
        return {}

    try:
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        metadata = {
            "file_size_bytes": os.path.getsize(filepath),
        }

        # Handle different checkpoint formats
        state_dict = None

        if isinstance(checkpoint, dict):
            # PyTorch Lightning format: has 'state_dict' key
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

                # Extract epoch if available
                if "epoch" in checkpoint:
                    metadata["saved_epoch"] = checkpoint["epoch"]

                # Extract global_step if available
                if "global_step" in checkpoint:
                    metadata["saved_global_step"] = checkpoint["global_step"]

                # Extract hyperparameters if available
                if "hyper_parameters" in checkpoint:
                    metadata["saved_hyperparameters"] = checkpoint["hyper_parameters"]

            # Standard PyTorch format: has 'model_state_dict' key
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

                # Extract epoch/loss if available
                if "epoch" in checkpoint:
                    metadata["saved_epoch"] = checkpoint["epoch"]
                if "loss" in checkpoint:
                    metadata["saved_loss"] = float(checkpoint["loss"])

            # Raw state_dict saved as dict
            else:
                # Check if it looks like a state_dict (has tensor values)
                if checkpoint and any(hasattr(v, "numel") for v in checkpoint.values()):
                    state_dict = checkpoint

        # If checkpoint is an OrderedDict, it's likely a raw state_dict
        elif hasattr(checkpoint, "items"):
            state_dict = checkpoint

        # Extract metadata from state_dict
        if state_dict is not None:
            metadata.update(_extract_from_state_dict(state_dict))

        return metadata

    except Exception:
        # Failed to load - return empty dict
        return {}


def _extract_from_state_dict(state_dict: dict) -> dict:
    """Extract metadata from PyTorch state_dict."""
    total_params = 0
    layer_names = []

    for name, param in state_dict.items():
        if hasattr(param, "numel"):
            total_params += param.numel()

            # Extract layer name (remove .weight, .bias suffixes)
            layer_name = name.rsplit(".", 1)[0] if "." in name else name
            if layer_name and layer_name not in layer_names:
                layer_names.append(layer_name)

    metadata = {
        "total_parameters": total_params,
        "num_layers": len(layer_names),
    }

    # Include first few layer names for inspection
    if layer_names:
        metadata["layer_names_sample"] = layer_names[:10]

    return metadata
