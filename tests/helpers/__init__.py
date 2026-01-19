"""
Helper functions for e2e tests
Re-exports all functions from submodules to maintain backward compatibility
"""

# API client functions
from .api import (
    create_run,
    finish_run,
    log_artifact,
    log_structured_data,
)

# Artifact creation helpers
from .artifacts import (
    create_fake_model_checkpoint,
    create_model_metadata,
    create_predictions_csv,
    create_training_log,
)

# Data generation functions
from .data_generation import (
    generate_ab_test_data,
    generate_climate_data,
    generate_csv_data,
    generate_finance_backtest_data,
    generate_mock_system_metrics,
    generate_path_planning_data,
    generate_simulation_data,
)

# Image generation functions
from .images import generate_synthetic_images

__all__ = [
    # API functions
    "create_run",
    "finish_run",
    "log_artifact",
    "log_structured_data",
    # Artifact functions
    "create_fake_model_checkpoint",
    "create_model_metadata",
    "create_predictions_csv",
    "create_training_log",
    # Data generation functions
    "generate_ab_test_data",
    "generate_climate_data",
    "generate_csv_data",
    "generate_finance_backtest_data",
    "generate_mock_system_metrics",
    "generate_path_planning_data",
    "generate_simulation_data",
    # Image functions
    "generate_synthetic_images",
]
