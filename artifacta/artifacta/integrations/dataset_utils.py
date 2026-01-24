"""Dataset metadata logging utilities.

This module provides shared utilities for logging dataset metadata across
different ML frameworks (sklearn, xgboost, lightgbm, etc.).

Dataset metadata logged includes:
- Shape and size of features and targets
- Data types
- Hash/digest for change detection
- Column names (for pandas DataFrames)
- Context (train/eval/test)

This enables:
- Reproducibility: Detect when training data changes via hash
- Debugging: Verify data shapes match expectations
- Comparison: Check if same data used across experiments
"""

import hashlib
import logging
from typing import Any, Dict, Optional

import numpy as np

_logger = logging.getLogger(__name__)


def log_dataset_metadata(run, X, y=None, context="train"):
    """Log dataset metadata to Artifacta run.

    Logs shape, dtype, size, and hash of features and targets.

    Args:
        run: Artifacta run object
        X: Features (numpy array, pandas DataFrame, scipy sparse matrix)
        y: Targets (numpy array, pandas Series) - optional
        context: Dataset context - "train", "eval", "test", etc.

    Example:
        >>> import artifacta as ds
        >>> run = ds.init(project="test")
        >>> log_dataset_metadata(run, X_train, y_train, context="train")
        >>> log_dataset_metadata(run, X_test, y_test, context="test")

    Logged metadata:
        {
          "context": "train",
          "features_shape": (1000, 20),
          "features_size": 20000,
          "features_nbytes": 160000,
          "features_dtype": "float64",
          "features_digest": "sha256:abc123...",
          "columns": ["age", "income", ...],  # If pandas DataFrame
          "targets_shape": (1000,),
          "targets_size": 1000,
          "targets_nbytes": 8000,
          "targets_dtype": "int64",
          "targets_digest": "sha256:def456..."
        }
    """
    try:
        import json

        metadata = _extract_dataset_metadata(X, y, context)
        if metadata:
            # Convert to JSON string
            metadata_json = json.dumps(metadata, indent=2, default=str)

            # Log as virtual artifact (no file needed)
            run._log_virtual_artifact(
                name=f"dataset_{context}.json",
                type="dataset_metadata",
                content_str=metadata_json,
                mime_type="application/json"
            )
    except Exception as e:
        _logger.warning(f"Failed to log dataset metadata: {e}")


def _extract_dataset_metadata(X, y=None, context="train") -> Optional[Dict[str, Any]]:
    """Extract metadata from features and targets.

    Args:
        X: Features
        y: Targets (optional)
        context: Dataset context

    Returns:
        Dictionary of metadata, or None if extraction fails
    """

    # Convert X to numpy array and extract metadata
    X_array, columns = _to_numpy_array(X)
    if X_array is None:
        return None

    # Log features metadata
    metadata = {
        "context": context,
        "features_shape": list(X_array.shape),
        "features_size": int(X_array.size),
        "features_nbytes": int(X_array.nbytes),
        "features_dtype": str(X_array.dtype),
        "features_digest": _compute_hash(X_array),
    }

    # Add column names if available
    if columns is not None:
        metadata["columns"] = columns

    # Log targets metadata if provided
    if y is not None:
        y_array, _ = _to_numpy_array(y)
        if y_array is not None:
            metadata.update({
                "targets_shape": list(y_array.shape),
                "targets_size": int(y_array.size),
                "targets_nbytes": int(y_array.nbytes),
                "targets_dtype": str(y_array.dtype),
                "targets_digest": _compute_hash(y_array),
            })

    return metadata


def _to_numpy_array(data):
    """Convert various data types to numpy array.

    Args:
        data: Input data (numpy, pandas, scipy sparse, etc.)

    Returns:
        Tuple of (numpy_array, columns) where columns is None or list of column names
    """
    import pandas as pd
    from scipy.sparse import issparse

    columns = None

    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        return data.values, columns

    # Handle pandas Series
    elif isinstance(data, pd.Series):
        return data.values, None

    # Handle numpy array
    elif isinstance(data, np.ndarray):
        return data, None

    # Handle scipy sparse matrix
    elif issparse(data):
        return data.toarray(), None

    # Handle list
    elif isinstance(data, list):
        return np.array(data), None

    # Unknown type
    else:
        _logger.warning(f"Unsupported data type for dataset logging: {type(data)}")
        return None, None


def _compute_hash(array: np.ndarray) -> str:
    """Compute SHA256 hash of numpy array.

    Args:
        array: Numpy array

    Returns:
        SHA256 hash as hex string
    """
    try:
        return hashlib.sha256(array.tobytes()).hexdigest()
    except Exception as e:
        _logger.warning(f"Failed to compute hash: {e}")
        return "unknown"
