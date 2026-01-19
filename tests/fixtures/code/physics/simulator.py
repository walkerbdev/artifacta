"""Physics simulation - N-body problem"""

import numpy as np


class NBodySimulator:
    def __init__(self, n_bodies=3, dt=0.01):
        self.n_bodies = n_bodies
        self.dt = dt
        self.positions = np.random.rand(n_bodies, 3)
        self.velocities = np.random.rand(n_bodies, 3)
        self.masses = np.random.rand(n_bodies)

    def step(self):
        # Compute gravitational forces
        forces = np.zeros_like(self.positions)
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(r)
                force = self.masses[i] * self.masses[j] / (dist**2 + 1e-6)
                forces[i] += force * r / dist
                forces[j] -= force * r / dist

        # Update velocities and positions
        self.velocities += forces / self.masses[:, None] * self.dt
        self.positions += self.velocities * self.dt

        return self.positions.copy()
