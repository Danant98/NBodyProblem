import numpy as np


class numerical:

    @staticmethod
    def euler_cromer(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Euler-Cromer method for updating position and velocity
        """
        vel += acc * dt
        pos += vel * dt
        return vel, pos

    @staticmethod
    def verlet(pos: np.ndarray, vel: np.ndarray, dt: float, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Velocity Verlet method for updating position and velocity
        """
        pos += vel * dt + 0.5 * acc * dt**2
        vel += 0.5 * acc * dt
        return vel, pos

