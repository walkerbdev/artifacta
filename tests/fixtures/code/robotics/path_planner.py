"""Robot path planning simulation"""

import numpy as np


class PathPlanner:
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.obstacles = []

    def plan_path(self, start, goal):
        # Simple A* path planning
        path = [start]
        current = np.array(start)
        goal = np.array(goal)

        while np.linalg.norm(current - goal) > 1.0:
            direction = goal - current
            direction = direction / np.linalg.norm(direction)
            current = current + direction
            path.append(current.copy())

        return np.array(path)
