"""Utility functions for data transformation and serialization.

This module provides helper functions for common data transformations needed
throughout the Artifacta codebase, particularly for configuration flattening
and JSON serialization.
"""

import json


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary into dot-notation key-value pairs.

    This function recursively traverses a nested dictionary structure and converts
    it into a flat dictionary with dot-separated keys. This is useful for storing
    hierarchical configurations in flat key-value stores (databases, tags, etc.).

    Flattening Algorithm:
        1. Iterate over each key-value pair in the input dictionary
        2. Construct new_key by joining parent_key with current key using separator
        3. Handle three value types:
           a. Dict -> Recursively flatten and extend items list
           b. List -> Convert to JSON string representation
           c. Primitive (str, int, float, bool) -> Convert to string
        4. Accumulate all (key, value) pairs in items list
        5. Convert items list to dictionary and return

    Why convert everything to strings:
        - Database storage: Most key-value stores require string values
        - Tag systems: Tags are typically string-based for consistency
        - JSON serialization: Lists as JSON strings preserve structure
        - Type preservation: Can be reversed by parsing JSON for lists

    Recursion termination:
        - Base case: Value is not a dict -> store as string
        - Recursive case: Value is a dict -> call flatten_dict with current key as parent

    Edge cases:
        - Empty dict: Returns empty dict
        - Empty list: Returns "[]" as value
        - None values: Converted to string "None"
        - Nested empty dicts: Recursion handles naturally

    Examples:
        Simple nesting:
            {"a": {"b": 1}} -> {"a.b": "1"}

        Lists are JSON-encoded:
            {"model": {"layers": [64, 32]}} -> {"model.layers": "[64, 32]"}

        Deep nesting:
            {"training": {"optimizer": {"type": "adam"}}}
            -> {"training.optimizer.type": "adam"}

        Mixed types:
            {"config": {"lr": 0.01, "layers": [64, 32], "name": "model"}}
            -> {"config.lr": "0.01", "config.layers": "[64, 32]", "config.name": "model"}

    Args:
        d: Dictionary to flatten (can be nested arbitrarily deep)
        parent_key: Prefix for keys (used internally for recursion, default "")
        sep: Separator for nested keys (default ".", can use "/" or "_")

    Returns:
        Flattened dictionary with string keys and string values
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
