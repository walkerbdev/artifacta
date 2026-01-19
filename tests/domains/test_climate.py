"""
End-to-end tests for Climate Science domain
Temperature Analysis with parameter sweep

Run with: pytest tests/domains/test_climate.py -v
"""

import os
import time

import pytest

import artifacta as ds
from tests.helpers import generate_climate_data
from tests.helpers.notebook_html import create_climate_notebook

# Path to example code
EXAMPLE_CODE_PATH = os.path.join(os.path.dirname(__file__), "../examples/climate")


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
def test_climate(api_url):
    """Test 7: Climate Science - Temperature Analysis with parameter sweep"""

    def run_climate_model(config, seed=42):
        run = ds.get_run()

        # Series: Temperature anomaly - varies by model_resolution and ensemble_size
        climate_data = generate_climate_data(config, seed)
        ds.log(
            "temperature_anomaly",
            ds.Series(
                index="year",
                fields={
                    "global_anomaly": climate_data["global_anomaly"],
                    "northern_hemisphere": climate_data["northern_hemisphere"],
                    "southern_hemisphere": climate_data["southern_hemisphere"],
                },
                index_values=climate_data["years"],
            ),
        )

        # Matrix: Regional correlations
        ds.log(
            "regional_correlations",
            ds.Matrix(
                rows=["North America", "Europe", "Asia", "Africa"],
                cols=["North America", "Europe", "Asia", "Africa"],
                values=[
                    [1.0, 0.85, 0.72, 0.68],
                    [0.85, 1.0, 0.78, 0.71],
                    [0.72, 0.78, 1.0, 0.65],
                    [0.68, 0.71, 0.65, 1.0],
                ],
            ),
        )

        # Log climate modeling code artifact (example file)
        run.log_input(EXAMPLE_CODE_PATH)

    base_config = {"start_year": 2010, "end_year": 2020}
    param_variations = [
        {"model_resolution": "1deg", "ensemble_size": 10},
        {"model_resolution": "0.5deg", "ensemble_size": 10},
        {"model_resolution": "1deg", "ensemble_size": 20},
    ]

    run_parameter_sweep("climate-sweep", base_config, param_variations, run_climate_model)

    # Create rich notebook with climate metrics
    climate_metrics = {
        "projected_warming_2100": 2.1,
        "co2_concentration_ppm": 478,
        "sea_level_rise_cm": 47.3,
        "arctic_ice_reduction": 0.385,
        "ensemble_uncertainty": 0.42,
    }
    create_climate_notebook(
        project_id="climate-sweep",
        climate_metrics=climate_metrics,
        api_url=api_url,
    )
