"""
End-to-end tests for Finance domain
Trading Strategy Backtest with parameter sweep

Run with: pytest tests/domains/test_finance.py -v
"""

import os
import time

import numpy as np
import pytest

import artifacta as ds
from tests.helpers import generate_finance_backtest_data
from tests.helpers.notebook_html import create_finance_notebook

# Path to example code
EXAMPLE_CODE_PATH = os.path.join(os.path.dirname(__file__), "../examples/finance")


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
def test_finance(api_url):
    """Test 5: Finance - Trading Strategy Backtest with parameter sweep"""

    def run_backtest(config, seed=42):
        run = ds.get_run()
        np.random.seed(seed)

        # Series: Portfolio performance - varies by lookback_period and stop_loss
        finance_data = generate_finance_backtest_data(config, seed)
        ds.log(
            "portfolio_performance",
            ds.Series(
                index="day",
                fields={
                    "portfolio_value": finance_data["portfolio_value"],
                    "benchmark": finance_data["benchmark"],
                },
                index_values=finance_data["days"],
            ),
        )

        # Distribution: Daily returns
        ds.log(
            "daily_returns",
            ds.Distribution(
                values=np.random.normal(0.0012, 0.015, 100).tolist(),
                groups=None,
            ),
        )

        # Note: Events primitive removed - trade signals not visualized

        # Log trading strategy code artifact (example file)
        run.log_input(EXAMPLE_CODE_PATH)

    base_config = {"strategy": "momentum", "rebalance_frequency": "daily"}
    param_variations = [
        {"lookback_period": 20, "stop_loss": 0.02},
        {"lookback_period": 10, "stop_loss": 0.02},
        {"lookback_period": 20, "stop_loss": 0.05},
    ]

    run_parameter_sweep("finance-sweep", base_config, param_variations, run_backtest)

    # Create rich notebook with portfolio metrics
    portfolio_metrics = {
        "sharpe_ratio": 1.85,
        "max_drawdown": -0.152,
        "annual_return": 0.247,
        "volatility": 0.133,
        "var_95": -0.0287,
    }
    create_finance_notebook(
        project_id="finance-sweep",
        portfolio_metrics=portfolio_metrics,
        api_url=api_url,
    )
