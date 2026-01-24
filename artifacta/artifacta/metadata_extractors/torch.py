"""PyTorch model metadata extractor for checkpoint files.

Automatically extracts metadata from PyTorch model checkpoints (.pt, .pth, .ckpt)
to provide insights into model architecture, training progress, and hyperparameters.

Supported checkpoint formats:
- PyTorch Lightning: {"state_dict": {...}, "epoch": N, "hyper_parameters": {...}}
- Standard PyTorch: {"model_state_dict": {...}, "epoch": N, "loss": X}
- Raw state_dict: OrderedDict of layer tensors

Extraction strategy:
1. Load checkpoint with torch.load (CPU, no weights_only for compatibility)
2. Detect checkpoint format by inspecting dict keys
3. Extract training metadata (epoch, loss, hyperparameters)
4. Analyze state_dict for parameter count and layer names
5. Return structured metadata dict

Performance:
- Loads on CPU to avoid GPU memory usage
- Works even if training was done on CUDA
- Handles large checkpoints efficiently
"""

import os


def extract_pytorch_metadata(filepath: str) -> dict:
    """Extract metadata from PyTorch checkpoint file (.pt, .pth, .ckpt).

    Detection algorithm:
    1. Check if PyTorch is installed (graceful failure if not)
    2. Load checkpoint with map_location="cpu" to avoid GPU dependency
    3. Inspect checkpoint structure to determine format:
       - Has "state_dict" key? → PyTorch Lightning format
       - Has "model_state_dict" key? → Standard PyTorch format
       - Has tensor values? → Raw state_dict
    4. Extract training metadata if available (epoch, loss, hyperparameters)
    5. Analyze state_dict for model architecture:
       - Count total parameters (sum of tensor.numel() across all layers)
       - Extract unique layer names (remove .weight/.bias suffixes)
       - Sample first 10 layer names for inspection

    Why map_location="cpu":
    - Avoids CUDA out of memory errors when extracting metadata
    - Works even if checkpoint was saved on GPU
    - Metadata extraction doesn't need GPU compute

    Why weights_only=False:
    - Some checkpoints contain non-tensor objects (hyperparameters, optimizers)
    - We need full checkpoint structure for metadata extraction
    - Security is less of a concern for user's own checkpoints

    Args:
        filepath: Path to PyTorch checkpoint file

    Returns:
        dict with extracted metadata:
        - file_size_bytes: Size of checkpoint file
        - total_parameters: Total number of model parameters
        - num_layers: Number of unique layers
        - layer_names_sample: First 10 layer names
        - saved_epoch: Training epoch when checkpoint was saved (if available)
        - saved_global_step: Global training step (PyTorch Lightning)
        - saved_loss: Training loss at checkpoint (if available)
        - saved_hyperparameters: Model hyperparameters (PyTorch Lightning)

    Returns empty dict if:
    - PyTorch is not installed
    - File doesn't exist
    - Checkpoint can't be loaded (corrupted, incompatible format)
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
    """Extract metadata from PyTorch state_dict.

    Algorithm:
    1. Iterate through all entries in state_dict
    2. For each tensor parameter:
       - Count parameters using tensor.numel() (number of elements)
       - Extract layer name by removing suffix (.weight, .bias, etc.)
       - Track unique layer names (avoid duplicates)
    3. Return parameter count, layer count, and sample names

    Why remove suffixes:
    - A single layer has multiple entries (weight, bias, running_mean, etc.)
    - We want to count unique layers, not unique parameters
    - Example: "conv1.weight" and "conv1.bias" → one layer "conv1"

    Why limit to 10 layer names:
    - Prevents overwhelming metadata for large models
    - Provides enough info for inspection without bloat
    - Full layer list can be extracted from checkpoint if needed

    Args:
        state_dict: PyTorch state_dict (dict mapping param names to tensors)

    Returns:
        dict with:
        - total_parameters: Total parameter count across all layers
        - num_layers: Number of unique layers
        - layer_names_sample: First 10 unique layer names
    """
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
