"""Utility functions for artifacta."""

import json


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary into dot-notation key-value pairs.

    This is fully agnostic - works with ANY nested structure.

    Examples:
        {"a": {"b": 1}} -> {"a.b": "1"}
        {"model": {"layers": [64, 32]}} -> {"model.layers": "[64, 32]"}
        {"training": {"optimizer": {"type": "adam"}}} -> {"training.optimizer.type": "adam"}

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator for nested keys (default: ".")

    Returns:
        Flattened dictionary with string values
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # Recursively flatten nested dicts
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to JSON string
            items.append((new_key, json.dumps(v)))
        else:
            # Store primitive values as strings
            items.append((new_key, str(v)))

    return dict(items)
