"""
End-to-end tests for Physics Simulation domain
Particle Dynamics with parameter sweep

Run with: pytest tests/domains/test_physics.py -v
"""

import os
import time

import pytest

import artifacta as ds
from tests.helpers import generate_simulation_data


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
def test_physics_simulation():
    """Test 2: Physics Simulation - Particle Dynamics with parameter sweep"""

    def run_physics(config, seed=42):
        run = ds.get_run()

        # Series: Energy conservation over time - varies by timestep
        sim_data = generate_simulation_data(config, seed)
        ds.log(
            "energy_conservation",
            ds.Series(
                index="time (s)",
                fields={
                    "kinetic_energy": sim_data["kinetic_energy"],
                    "potential_energy": sim_data["potential_energy"],
                    "total_energy": sim_data["total_energy"],
                },
                index_values=sim_data["time_values"],
            ),
        )

        # Note: Graph primitive removed - too specialized for general use

        # Table: Final particle states
        # Handle Table special case: convert 'rows' to 'data' and fix columns format
        cols = ["Particle", "Position X", "Position Y", "Velocity", "Energy"]
        col_types = ["string", "number", "number", "number", "number"]
        columns = [{"name": name, "type": typ} for name, typ in zip(cols, col_types)]

        ds.log(
            "final_states",
            ds.Table(
                columns=columns,
                data=[
                    ["Particle 1", 12.5, 8.3, 2.1, 44.1],
                    ["Particle 2", -5.2, 15.7, 1.8, 25.9],
                    ["Particle 3", 3.1, -9.4, 2.5, 75.0],
                    ["Particle 4", -8.6, 2.1, 1.5, 20.3],
                ],
            ),
        )

        # Log physics simulation code artifact
        # CONSTANT CODE - same for all runs to demonstrate hash.code grouping
        code_path = os.path.join(os.path.dirname(__file__), "../fixtures/code/physics")
        run.log_input(code_path)

    base_config = {"num_particles": 4, "simulation_duration": 1.0}
    param_variations = [
        {"timestep": 0.01, "integration_method": "runge-kutta-4"},
        {"timestep": 0.005, "integration_method": "runge-kutta-4"},
        {"timestep": 0.01, "integration_method": "euler"},
    ]

    run_parameter_sweep("physics-sweep", base_config, param_variations, run_physics)
