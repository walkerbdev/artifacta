"""Test utilities for verifying autolog functionality.

Provides helper functions to inspect what was logged by autolog integrations
without needing to query the tracking server or database directly.
"""

import json
from unittest.mock import patch


class MockHTTPEmitter:
    """Mock HTTPEmitter that captures emitted data for testing.

    Replaces the real HTTPEmitter to capture all logged data without
    needing a running tracking server.
    """

    def __init__(self, run_id):
        self.run_id = run_id
        self.emitted_data = []
        self.emitted_artifacts = []
        self.init_called = False
        self.enabled = True

    def emit_init(self, data):
        """Capture run initialization."""
        self.init_called = True
        self.emitted_data.append(("init", data))

    def emit_structured_data(self, data):
        """Capture structured data (metrics, params)."""
        self.emitted_data.append(("structured_data", data))

    def emit_artifact(self, data, content=None, role=None):
        """Capture artifact metadata and content."""
        # Store both metadata and content
        artifact_with_content = {**data, "content": content, "role": role}
        self.emitted_artifacts.append(artifact_with_content)

    def close(self):
        """Close emitter (no-op for mock)."""
        pass


def get_logged_params(run):
    """Extract logged parameters from a run.

    Args:
        run: Artifacta run object

    Returns:
        Dictionary of logged parameters
    """
    # First check run.config (new approach using update_config())
    if hasattr(run, 'config') and run.config:
        return dict(run.config)

    # Fallback: check emitted_data (old approach using run.log())
    if not hasattr(run.http_emitter, 'emitted_data'):
        return {}

    params = {}
    for event_type, data in run.http_emitter.emitted_data:
        if event_type == "structured_data":
            if data.get("name") in ["parameters", "xgboost_params", "sklearn_params"]:
                # Extract params from structured data
                data_dict = data.get("data", {})
                # Handle Series format: {"index_values": [{"param1": val1, ...}]}
                if "index_values" in data_dict:
                    index_values = data_dict["index_values"]
                    if isinstance(index_values, list) and len(index_values) > 0:
                        params.update(index_values[0])
                # Handle other formats
                elif "params" in data_dict:
                    param_list = data_dict["params"]
                    if isinstance(param_list, list) and len(param_list) > 0:
                        params.update(param_list[0])

    return params


def get_logged_metrics(run):
    """Extract logged metrics from a run.

    Args:
        run: Artifacta run object

    Returns:
        Dictionary of metric name -> values
    """
    if not hasattr(run.http_emitter, 'emitted_data'):
        return {}

    metrics = {}
    for event_type, data in run.http_emitter.emitted_data:
        if event_type == "structured_data":
            name = data.get("name", "")
            if "metric" in name.lower() or name in ["xgboost_metrics", "sklearn_metrics"]:
                # Extract metrics from structured data
                data_dict = data.get("data", {})

                # Handle Series format with index and fields
                # {"index": "metric", "fields": {"value": [...]}, "index_values": ["metric1", "metric2"]}
                if "index" in data_dict and "fields" in data_dict:
                    index_values = data_dict.get("index_values", [])
                    fields = data_dict.get("fields", {})
                    if "value" in fields:
                        values = fields["value"]
                        for metric_name, value in zip(index_values, values):
                            metrics[metric_name] = value

                # Handle other formats (raw dict with metric arrays)
                # XGBoost logs like: {"iteration": [0,1,2], "train_logloss": [0.5, 0.3, 0.2]}
                else:
                    for key, value in data_dict.items():
                        if isinstance(value, (int, float, list)):
                            metrics[key] = value

                # Also check for fields dict (Series format)
                # {"fields": {"train_logloss": [0.5, 0.3], "test_logloss": [0.6, 0.4]}}
                if "fields" in data_dict and isinstance(data_dict["fields"], dict):
                    for key, value in data_dict["fields"].items():
                        if isinstance(value, (int, float, list)):
                            metrics[key] = value

    return metrics


def get_logged_artifacts(run):
    """Extract logged artifacts from a run.

    Args:
        run: Artifacta run object

    Returns:
        List of artifact metadata dictionaries
    """
    if not hasattr(run.http_emitter, 'emitted_artifacts'):
        return []

    return run.http_emitter.emitted_artifacts


def assert_param_logged(run, param_name, expected_value=None):
    """Assert that a parameter was logged.

    Args:
        run: Artifacta run object
        param_name: Name of parameter to check
        expected_value: Optional expected value (if None, just check existence)

    Raises:
        AssertionError: If parameter not logged or value doesn't match
    """
    params = get_logged_params(run)
    assert param_name in params, f"Parameter '{param_name}' was not logged. Logged params: {list(params.keys())}"

    if expected_value is not None:
        actual_value = params[param_name]
        assert actual_value == expected_value, (
            f"Parameter '{param_name}' has value {actual_value}, expected {expected_value}"
        )


def assert_metric_logged(run, metric_name):
    """Assert that a metric was logged.

    Args:
        run: Artifacta run object
        metric_name: Name of metric to check

    Raises:
        AssertionError: If metric not logged
    """
    metrics = get_logged_metrics(run)
    assert metric_name in metrics, f"Metric '{metric_name}' was not logged. Logged metrics: {list(metrics.keys())}"


def assert_artifact_logged(run, artifact_name_contains):
    """Assert that an artifact was logged.

    Args:
        run: Artifacta run object
        artifact_name_contains: String that should be in the artifact name

    Raises:
        AssertionError: If no matching artifact found
    """
    artifacts = get_logged_artifacts(run)
    matching = [a for a in artifacts if artifact_name_contains in a.get("name", "")]
    assert len(matching) > 0, (
        f"No artifact with name containing '{artifact_name_contains}' found. "
        f"Logged artifacts: {[a.get('name') for a in artifacts]}"
    )


def assert_model_artifact_logged(run):
    """Assert that a model artifact was logged.

    Args:
        run: Artifacta run object

    Raises:
        AssertionError: If no model artifact found
    """
    artifacts = get_logged_artifacts(run)
    model_artifacts = [
        a for a in artifacts
        if "model" in a.get("name", "").lower() or
           a.get("metadata", {}).get("artifact_type", "").endswith("_model")
    ]
    assert len(model_artifacts) > 0, (
        f"No model artifact found. Logged artifacts: {[a.get('name') for a in artifacts]}"
    )


def count_logged_artifacts(run, exclude_config=False):
    """Count number of artifacts logged.

    Args:
        run: Artifacta run object
        exclude_config: If True, exclude config artifacts from count

    Returns:
        Number of artifacts logged
    """
    artifacts = get_logged_artifacts(run)
    if exclude_config:
        # Filter out config artifacts (auto-logged from update_config())
        artifacts = [a for a in artifacts if a.get("name") != "config.json"]
    return len(artifacts)


def get_logged_datasets(run):
    """Extract logged dataset metadata from a run.

    Dataset metadata is stored as JSON artifacts (not structured data).

    Args:
        run: Artifacta run object

    Returns:
        Dictionary of context -> dataset metadata
    """

    if not hasattr(run.http_emitter, 'emitted_artifacts'):
        return {}

    datasets = {}
    for artifact_data in run.http_emitter.emitted_artifacts:
        name = artifact_data.get("name", "")
        # Check if this is dataset metadata (name like "dataset_train.json")
        if name.startswith("dataset_") and name.endswith(".json"):
            # Extract context from name (e.g., "dataset_train.json" -> "train")
            context = name.replace("dataset_", "").replace(".json", "")

            # Parse JSON content from artifact
            # Artifacts are stored with content in emitted_data as well
            # We need to find the corresponding content
            content = artifact_data.get("content")
            if content:
                try:
                    # Content is a JSON string containing file collection
                    content_obj = json.loads(content) if isinstance(content, str) else content
                    # Extract the actual JSON content from the file
                    if "files" in content_obj and len(content_obj["files"]) > 0:
                        file_content = content_obj["files"][0].get("content", "{}")
                        metadata = json.loads(file_content) if isinstance(file_content, str) else file_content
                        datasets[context] = metadata
                except Exception:
                    pass

    return datasets


def assert_dataset_logged(run, context="train", expected_shape=None):
    """Assert that dataset metadata was logged.

    Args:
        run: Artifacta run object
        context: Dataset context ("train", "eval", etc.)
        expected_shape: Optional expected features shape tuple

    Raises:
        AssertionError: If dataset not logged or shape doesn't match
    """
    datasets = get_logged_datasets(run)
    assert context in datasets, (
        f"Dataset with context '{context}' was not logged. "
        f"Logged datasets: {list(datasets.keys())}"
    )

    dataset_meta = datasets[context]
    assert "features_shape" in dataset_meta, "Dataset missing features_shape"
    assert "features_dtype" in dataset_meta, "Dataset missing features_dtype"
    assert "features_digest" in dataset_meta, "Dataset missing features_digest"
    assert "context" in dataset_meta, "Dataset missing context"
    assert dataset_meta["context"] == context, (
        f"Dataset context mismatch: {dataset_meta['context']} != {context}"
    )

    if expected_shape is not None:
        actual_shape = tuple(dataset_meta["features_shape"])
        assert actual_shape == expected_shape, (
            f"Dataset shape mismatch: {actual_shape} != {expected_shape}"
        )


def patch_http_emitter():
    """Context manager to patch HTTPEmitter with MockHTTPEmitter.

    Usage:
        with patch_http_emitter():
            run = ds.init(...)
            # run.http_emitter will be MockHTTPEmitter
    """
    # Patch in multiple locations to ensure it's caught
    import artifacta.run
    return patch.object(artifacta.run, 'HTTPEmitter', MockHTTPEmitter)
