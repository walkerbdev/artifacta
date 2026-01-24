"""
API client functions for e2e tests
Functions for creating runs, logging data, and managing artifacts using artifacta SDK
"""

import os

import artifacta as ds


def create_run(api_url, project, name, config=None):
    """Create a new run using artifacta SDK

    Args:
        api_url: Base API URL from fixture (ignored, SDK uses env vars)
        project: Project name
        name: Run name
        config: Optional config dict
    """
    # Initialize run with SDK - it auto-logs provenance
    run = ds.init(project=project, name=name, config=config)
    return run.id


def log_structured_data(api_url, run_id, name, primitive_type, data, section=None, metadata=None):
    """Log structured data primitive using artifacta SDK"""
    # Map primitive_type string to actual artifacta primitive class
    primitive_map = {
        "series": ds.Series,
        "scatter": ds.Scatter,
        "distribution": ds.Distribution,
        "matrix": ds.Matrix,
        "curve": ds.Curve,
        "table": ds.Table,
    }

    primitive_class = primitive_map.get(primitive_type)
    if not primitive_class:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

    # Handle Table special case: convert 'rows' to 'data' and fix columns format
    if primitive_type == "table" and "rows" in data:
        cols = data.get("columns", [])
        col_types = data.get("column_types", ["string"] * len(cols))

        # Convert simple column names to column objects if needed
        if cols and isinstance(cols[0], str):
            data["columns"] = [{"name": name, "type": typ} for name, typ in zip(cols, col_types)]

        # Rename 'rows' to 'data'
        data["data"] = data.pop("rows")
        data.pop("column_types", None)  # Remove column_types

    # Create primitive instance and log it
    primitive = primitive_class(**data)
    if section:
        primitive._section = section
    if metadata:
        primitive._metadata = metadata

    ds.log(name, primitive)
    print(f"Logged {primitive_type}: {name} [{section or 'General'}]")


def finish_run(api_url, run_id):
    """Mark run as finished using artifacta SDK"""
    ds.finish()


def log_artifact(api_url, run_id, filepath, role=None, include_content=True):
    """Log artifact using artifacta SDK

    Args:
        api_url: Base API URL (ignored, SDK uses env vars)
        run_id: Run identifier (ignored, SDK uses current run)
        filepath: Path to artifact file or directory
        role: 'input' or 'output' (default: 'output')
        include_content: If True, inline text file contents
    """
    run = ds.get_run()
    if not run:
        raise RuntimeError("No active run. Call create_run() first.")

    # Use filename as the artifact name
    filename = os.path.basename(filepath)

    # Log artifact using SDK with role
    run.log_artifact(filename, filepath, include_content=include_content, role=role or "output")

    print(f"Logged artifact: {filename}")
