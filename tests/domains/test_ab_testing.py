"""
End-to-end tests for A/B Testing domain
E-commerce Checkout Flow with parameter sweep

Run with: pytest tests/domains/test_ab_testing.py -v
"""

import os
import time

import pytest

import artifacta as ds
from tests.helpers import generate_ab_test_data
from tests.helpers.notebook_html import create_ab_testing_notebook

# Path to example code
EXAMPLE_CODE_PATH = os.path.join(os.path.dirname(__file__), "../examples/ab_testing")


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
def test_ab_testing(api_url):
    """Test 3: A/B Testing - E-commerce Checkout Flow with parameter sweep"""

    def run_ab_test(config, seed=42):
        run = ds.get_run()

        # Distribution: Conversion rates by variant - varies by traffic_split and sample_size
        ab_data = generate_ab_test_data(config, seed)
        ds.log(
            "conversion_rates",
            ds.Distribution(
                values=ab_data["values"],
                groups=ab_data["groups"],
            ),
        )

        # Series: Cumulative conversions over time
        ds.log(
            "cumulative_conversions",
            ds.Series(
                index="hour",
                fields={
                    "control": [10, 24, 41, 62, 87, 115, 148, 185, 227, 274],
                    "variant_a": [12, 31, 54, 82, 115, 153, 196, 244, 298, 356],
                },
                index_values=list(range(1, 11)),
            ),
        )

        # Table: Summary statistics
        # Handle Table special case: convert 'rows' to 'data' and fix columns format
        cols = ["Variant", "Users", "Conversions", "Rate", "Confidence"]
        col_types = ["string", "number", "number", "number", "string"]
        columns = [{"name": name, "type": typ} for name, typ in zip(cols, col_types)]

        ds.log(
            "summary_stats",
            ds.Table(
                columns=columns,
                data=[
                    ["Control", 8052, 998, 0.124, "-"],
                    ["Variant A", 8142, 1270, 0.156, "99.5%"],
                ],
            ),
        )

        # Log A/B test analysis code artifact (example file)
        run.log_input(EXAMPLE_CODE_PATH)

    base_config = {"test_duration_days": 10}
    param_variations = [
        {"traffic_split": 0.5, "min_sample_size": 1000, "significance_level": 0.05},
        {"traffic_split": 0.7, "min_sample_size": 1000, "significance_level": 0.05},
        {"traffic_split": 0.5, "min_sample_size": 500, "significance_level": 0.01},
    ]

    run_parameter_sweep("ab-test-sweep", base_config, param_variations, run_ab_test)

    # Create rich notebook with diverse content
    variant_a = {"conversion_rate": 0.0523, "avg_order_value": 125.50, "bounce_rate": 0.4210}
    variant_b = {"conversion_rate": 0.0587, "avg_order_value": 132.75, "bounce_rate": 0.3895}
    create_ab_testing_notebook(
        project_id="ab-test-sweep",
        variant_a_metrics=variant_a,
        variant_b_metrics=variant_b,
        api_url=api_url,
    )
