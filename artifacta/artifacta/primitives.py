"""Data primitives for structured logging.

Universal data schema that supports any domain:
- ML training
- A/B testing
- Physics simulations
- Financial data
- Genomics
- Analytics
- Robotics
- And more...
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Series:
    """Ordered data with single index dimension.

    Use cases:
    - ML training: loss/accuracy over epochs
    - Physics: temperature over time
    - Finance: stock prices over time
    - A/B testing: conversion rates over funnel steps

    Example:
        Series(
            index="epoch",
            fields={
                "train_loss": [0.5, 0.3, 0.2],
                "val_loss": [0.6, 0.4, 0.3]
            }
        )
    """

    index: str
    fields: Dict[str, List[float]]
    index_values: Optional[List] = None  # Optional: explicit index values (can be categorical)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "index": self.index,
            "fields": {
                k: list(v) if isinstance(v, np.ndarray) else v for k, v in self.fields.items()
            },
        }
        if self.index_values is not None:
            d["index_values"] = (
                list(self.index_values)
                if isinstance(self.index_values, np.ndarray)
                else self.index_values
            )
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Distribution:
    """Values with optional grouping.

    Use cases:
    - A/B testing: conversion rates by variant
    - ML: prediction distributions
    - Analytics: response times by server

    Example:
        Distribution(
            values=[0.12, 0.15, 0.18, 0.11],
            groups=["control", "variant_a", "variant_a", "control"]
        )
    """

    values: List[float]
    groups: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = {"values": list(self.values) if isinstance(self.values, np.ndarray) else self.values}
        if self.groups is not None:
            d["groups"] = list(self.groups) if isinstance(self.groups, np.ndarray) else self.groups
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Matrix:
    """2D relationships.

    Use cases:
    - ML: confusion matrix
    - Finance: portfolio correlation
    - Genomics: gene expression heatmap
    - Analytics: user cohort retention

    Example:
        Matrix(
            rows=["cat", "dog", "bird"],
            cols=["cat", "dog", "bird"],
            values=[[95, 3, 2], [1, 88, 11], [4, 8, 88]],
            metadata={"type": "confusion_matrix"}
        )
    """

    rows: List[str]
    cols: List[str]
    values: List[List[float]]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "rows": self.rows,
            "cols": self.cols,
            "values": [[float(v) for v in row] for row in self.values],
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Table:
    """Generic tabular data.

    Use cases:
    - Finance: multi-ticker time series
    - Analytics: user event logs
    - Experiments: arbitrary measurements

    Example:
        Table(
            columns=[
                {"name": "timestamp", "type": "datetime"},
                {"name": "value", "type": "float"}
            ],
            data=[[datetime(...), 1.5], [datetime(...), 2.3]]
        )
    """

    columns: List[Dict]  # [{"name": str, "type": str}]
    data: List[List]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = {"columns": self.columns, "data": self.data}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Curve:
    """Pure X-Y relationship (not time-indexed).

    Use cases:
    - ML: ROC curves, PR curves
    - Optimization: Pareto fronts
    - Biology: Dose-response curves
    - Economics: Supply-demand curves

    Example:
        Curve(
            x=[0, 0.2, 0.5, 1.0],
            y=[0, 0.6, 0.9, 1.0],
            x_label="False Positive Rate",
            y_label="True Positive Rate",
            baseline="diagonal",
            metric={"name": "AUC", "value": 0.95}
        )
    """

    x: List[float]
    y: List[float]
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    baseline: Optional[str] = None  # "diagonal", "horizontal", "vertical"
    metric: Optional[Dict] = None  # {"name": "AUC", "value": 0.95}
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "x": list(self.x) if isinstance(self.x, np.ndarray) else self.x,
            "y": list(self.y) if isinstance(self.y, np.ndarray) else self.y,
        }
        if self.x_label:
            d["x_label"] = self.x_label
        if self.y_label:
            d["y_label"] = self.y_label
        if self.baseline:
            d["baseline"] = self.baseline
        if self.metric:
            d["metric"] = self.metric
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Scatter:
    """Unordered point cloud (no sequential relationship).

    Use cases:
    - ML: Feature correlations, embeddings (t-SNE, UMAP)
    - Physics: Particle positions
    - Biology: Cell populations

    Example:
        Scatter(
            points=[
                {"x": 1.2, "y": 3.4, "label": "cluster_A", "size": 10},
                {"x": 2.1, "y": 1.5, "label": "cluster_B", "size": 15}
            ],
            x_label="PC1",
            y_label="PC2"
        )
    """

    points: List[
        Dict
    ]  # Each: {"x": float, "y": float, "z": float (opt), "label": str (opt), "size": float (opt)}
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    z_label: Optional[str] = None  # For 3D scatter
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = {"points": self.points}
        if self.x_label:
            d["x_label"] = self.x_label
        if self.y_label:
            d["y_label"] = self.y_label
        if self.z_label:
            d["z_label"] = self.z_label
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class BarChart:
    """Categorical bar chart for comparing values across discrete categories.

    Use cases:
    - ML: Model performance comparison, feature importance
    - A/B Testing: Conversion rates by variant
    - Analytics: Metrics by category/group
    - Business: Sales by region/product

    Example:
        BarChart(
            categories=["model_A", "model_B", "model_C"],
            groups={
                "accuracy": [0.85, 0.92, 0.88],
                "f1_score": [0.83, 0.90, 0.86]
            },
            x_label="Model",
            y_label="Score"
        )
    """

    categories: List[str]  # Category labels for x-axis
    groups: Dict[str, List[float]]  # Group name -> values for each category
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    stacked: bool = False  # If True, stack bars instead of grouping side-by-side
    horizontal: bool = False  # If True, horizontal bars instead of vertical
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "categories": self.categories,
            "groups": self.groups,
            "stacked": self.stacked,
            "horizontal": self.horizontal,
        }
        if self.x_label:
            d["x_label"] = self.x_label
        if self.y_label:
            d["y_label"] = self.y_label
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# Type mapping for internal use
PRIMITIVE_TYPES = {
    Series: "series",
    Distribution: "distribution",
    Matrix: "matrix",
    Table: "table",
    Curve: "curve",
    Scatter: "scatter",
    BarChart: "barchart",
}


def auto_convert(data):
    """Auto-detect and convert plain Python types to primitives.

    Makes it easy for users - they just log dicts/lists/arrays and we handle conversion.

    Examples:
        # Series from dict with multiple fields
        {"epoch": [1,2,3], "loss": [0.5,0.3,0.2]} -> Series

        # Series from dict with x/y
        {"x": [1,2,3], "y": [0.5,0.3,0.2]} -> Series

        # Distribution from 1D array/list
        [0.1, 0.2, 0.15, ...] -> Distribution

        # Matrix from 2D array/list
        [[1,2], [3,4]] -> Matrix

    Args:
        data: Plain Python dict, list, or numpy array

    Returns:
        One of the primitive types (Series, Distribution, Matrix, etc.)
    """
    # Already a primitive? Return as-is
    if type(data) in PRIMITIVE_TYPES:
        return data

    # Convert numpy arrays to lists for easier detection
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            # 1D array -> Distribution
            return Distribution(values=data.tolist())
        elif data.ndim == 2:
            # 2D array -> Matrix
            return Matrix(data=data.tolist())
        elif data.ndim >= 3:
            # 3D+ arrays not supported
            raise ValueError(f"3D+ arrays are not supported. Got shape {data.shape}")

    # Dict -> Series (most common case for metrics)
    if isinstance(data, dict):
        # Assume Series
        # Detect index field (first non-list field, or "x", "epoch", "step", "time")
        index_candidates = ["x", "epoch", "step", "time", "iteration"]
        index_key = None
        index_values = None
        fields = {}

        for key, value in data.items():
            if key in index_candidates and isinstance(value, (list, tuple, np.ndarray)):
                index_key = key
                index_values = (
                    list(value) if isinstance(value, (list, tuple, np.ndarray)) else value
                )
            elif isinstance(value, (list, tuple, np.ndarray)):
                fields[key] = list(value) if isinstance(value, (list, tuple, np.ndarray)) else value

        # If no index found, use first field as index
        if not index_key and fields:
            first_key = list(data.keys())[0]
            if isinstance(data[first_key], (list, tuple, np.ndarray)):
                index_key = first_key
                index_values = list(data[first_key])
                fields = {
                    k: list(v) if isinstance(v, (list, tuple, np.ndarray)) else v
                    for k, v in data.items()
                    if k != first_key
                }

        # Default index name
        if not index_key:
            index_key = "index"

        return Series(index=index_key, fields=fields, index_values=index_values)

    # List -> Distribution or Matrix depending on shape
    if isinstance(data, (list, tuple)):
        # Empty list
        if not data:
            return Distribution(values=[])

        # Check if nested (2D)
        if isinstance(data[0], (list, tuple, np.ndarray)):
            # 2D list -> Matrix
            return Matrix(
                data=[
                    list(row) if isinstance(row, (list, tuple, np.ndarray)) else row for row in data
                ]
            )
        else:
            # 1D list -> Distribution
            return Distribution(values=list(data))

    # Fallback: wrap in Distribution
    return Distribution(values=[data])
