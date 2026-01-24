"""
End-to-end tests for System Metrics domain
System resource monitoring with different profiles

Run with: pytest tests/domains/test_system.py -v
"""

import os
import time

import pytest

import artifacta as ds
from tests.helpers import generate_mock_system_metrics


@pytest.mark.e2e
def test_system_metrics_runs():
    """System Metrics Multi-Run Test - 3 scenarios with different resource profiles"""
    print("\n=== System Metrics Multi-Run Test ===")

    scenarios = [
        ("CPU-Intensive", "cpu-intensive"),
        ("Memory-Intensive", "memory-intensive"),
        ("Balanced", "balanced"),
    ]

    for name, profile in scenarios:
        print(f"\nLogging system metrics for: {name} ({profile})")

        # Create run with profile configuration
        ds.init(project="system-metrics", name=name, config={"profile": profile})
        run = ds.get_run()

        # Generate realistic system metrics based on profile
        metrics = generate_mock_system_metrics(profile, num_samples=50)

        # Log each metric as a series primitive
        for metric_name, values in metrics.items():
            if metric_name == "timestamp":
                continue  # Skip timestamp (used as index)

            series_obj = ds.Series(
                index="time(s)",
                fields={metric_name: values},
                index_values=metrics["timestamp"],
            )
            series_obj._section = "System Metrics"
            ds.log(metric_name, series_obj)

        # Log system monitoring code artifact from fixtures
        code_path = os.path.join(os.path.dirname(__file__), "../fixtures/code/system_monitor.py")
        run.log_input(code_path)

        print(f"  Completed: {name} profile")
        time.sleep(0.3)
