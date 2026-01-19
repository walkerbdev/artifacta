"""
End-to-end tests for Robotics domain
Path Planning with parameter sweep

Run with: pytest tests/domains/test_robotics.py -v
"""

import tempfile
import time

import pytest

import artifacta as ds
from tests.helpers import generate_path_planning_data


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
def test_robotics():
    """Test 6: Robotics - Path Planning with parameter sweep"""

    def run_path_planner(config, seed=42):
        run = ds.get_run()

        # Generate path data based on config
        path_data = generate_path_planning_data(config, seed)

        # Note: Graph primitive removed - too specialized for general use

        # Series: Planned trajectory - varies by max_iterations and step_size
        ds.log(
            "trajectory",
            ds.Series(
                index="time (s)",
                fields={
                    "x": path_data["x"],
                    "y": path_data["y"],
                    "velocity": path_data["velocity"],
                },
                index_values=path_data["time"],
            ),
        )

        # Note: Events primitive removed - planning milestones not visualized

        # Log robotics path planning code artifact
        algorithm = config.get("algorithm", "rrt")
        max_iter = config.get("max_iterations", 1000)
        step_size = config.get("step_size", 0.5)
        goal_tol = config.get("goal_tolerance", 0.1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as robotics_code:
            robotics_code.write(f"""import numpy as np
from scipy.spatial import distance

class PathPlanner:
    def __init__(self):
        self.algorithm = "{algorithm}"
        self.max_iterations = {max_iter}
        self.step_size = {step_size}
        self.goal_tolerance = {goal_tol}

    def rrt_plan(self, start, goal, obstacles):
        # Rapidly-exploring Random Tree (RRT) algorithm
        tree = [start]
        parent = {{0: None}}

        for i in range(self.max_iterations):
            # Sample random point
            if np.random.random() < 0.1:
                rand_point = goal  # Goal bias
            else:
                rand_point = np.random.uniform(0, 10, 2)

            # Find nearest node in tree
            distances = [distance.euclidean(node, rand_point) for node in tree]
            nearest_idx = np.argmin(distances)
            nearest = tree[nearest_idx]

            # Extend tree toward random point
            direction = (rand_point - nearest) / np.linalg.norm(rand_point - nearest)
            new_point = nearest + direction * self.step_size

            # Check collision with obstacles
            if not self.check_collision(new_point, obstacles):
                tree.append(new_point)
                parent[len(tree)-1] = nearest_idx

                # Check if goal reached
                if distance.euclidean(new_point, goal) < self.goal_tolerance:
                    return self.extract_path(tree, parent, len(tree)-1)

        return None

    def check_collision(self, point, obstacles):
        for obs in obstacles:
            if distance.euclidean(point, obs['center']) < obs['radius']:
                return True
        return False

    def extract_path(self, tree, parent, goal_idx):
        path = [tree[goal_idx]]
        current = goal_idx
        while parent[current] is not None:
            current = parent[current]
            path.append(tree[current])
        return path[::-1]

# Plan path
planner = PathPlanner()
start = np.array([0, 0])
goal = np.array([10, 10])
obstacles = [{{'center': np.array([5, 5]), 'radius': 1.0}}]
path = planner.rrt_plan(start, goal, obstacles)
print(f"Path found with {{len(path)}} waypoints")
""")
            robotics_code_path = robotics_code.name

        run.log_input(robotics_code_path)

    base_config = {"algorithm": "rrt", "goal_tolerance": 0.1}
    param_variations = [
        {"max_iterations": 1000, "step_size": 0.5},
        {"max_iterations": 500, "step_size": 0.5},
        {"max_iterations": 1000, "step_size": 0.3},
    ]

    run_parameter_sweep("robotics-sweep", base_config, param_variations, run_path_planner)
