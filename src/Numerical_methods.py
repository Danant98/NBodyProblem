import numpy as np


class numerical:

    def __init__(self) -> None:
        pass

    def euler_cromer(self, pos: np.ndarray, vel: np.ndarray, dt: float, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Euler-Cromer method for updating position and velocity
        """
        vel += acc * dt
        pos += vel * dt
        return vel, pos

    def verlet(self, pos: np.ndarray, vel: np.ndarray, dt: float, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Velocity Verlet method for updating position and velocity
        """
        pos += vel * dt + 0.5 * acc * dt**2
        vel += 0.5 * acc * dt
        return vel, pos

