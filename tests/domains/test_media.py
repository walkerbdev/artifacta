"""
End-to-end tests for Media Analysis domain
Satellite Imagery with parameter sweep

Run with: pytest tests/domains/test_media.py -v
"""

import os
import time

import numpy as np
import pytest

import artifacta as ds


def run_parameter_sweep(project_name, base_config, param_variations, run_fn):
    """
    Helper to run multiple experiments with parameter variations.

    Args:
        project_name: Project name for the runs
        base_config: Base configuration dict
        param_variations: List of dicts with parameter overrides
        run_fn: Function(config, seed) that executes the run logic
    """
    for idx, variation in enumerate(param_variations):
        config = {**base_config, **variation}
        # Use project name in run name to avoid collisions across different sweep tests
        run_name = f"{project_name}-run-{idx + 1}"
        ds.init(project=project_name, name=run_name, config=config)
        # Pass seed to make results vary but be reproducible
        run_fn(config, seed=42 + idx)
        time.sleep(0.3)


@pytest.mark.e2e
def test_media_analysis():
    """Test 8: Media Analysis - Satellite Imagery with parameter sweep"""

    def run_media_analysis(config, seed=42):
        run = ds.get_run()
        np.random.seed(seed)

        # Note: Images should be logged as dataset artifacts, not as plots
        # Media logging removed - use dataset artifact logging instead

        # Distribution: Classification confidence
        ds.log(
            "classification_confidence",
            ds.Distribution(
                values=np.random.beta(8, 2, 200).tolist(),
                groups=None,
            ),
        )

        # Log satellite imagery analysis code artifact
        code_path = os.path.join(os.path.dirname(__file__), "../fixtures/code/media")
        run.log_input(code_path)

    base_config = {"image_resolution": "10m", "num_classes": 5}
    param_variations = [
        {"classifier": "random_forest", "training_samples": 1000},
        {"classifier": "svm", "training_samples": 1000},
        {"classifier": "random_forest", "training_samples": 2000},
    ]

    run_parameter_sweep("media-analysis-sweep", base_config, param_variations, run_media_analysis)
