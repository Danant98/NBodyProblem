# Importing modules and libraries
from .base import Gravity_Solver
import numpy as np
from numba import njit

class Direct_Numba(Gravity_Solver):
    """
    Using a direct approach with numba optimmization of loops
    """
    @njit
    def acceleration(self, system, epsilon: float = 1e-5) -> np.ndarray:
        """
        Computing acceleration for system

        Gravitational softning: epsilon = 1E-5 by defalut
        """
        # Setting number of bodies N and number of dimensions
        N = system.N()
        dims = system.pos.shape[1]

        # Initializing acceleration array
        a = np.zeros((N, dims))

        for i in range(N):
            for j in range(N):
                # Checking to not compare same bodies
                if i == j:
                    continue

                # Initalizing variable for distance
                r2 = 0.0

                # Computing distance between bodies
                for k in range(dims):
                    dx = system.pos[j, k] - system.pos[i, k]
                    r2 += dx**2 

                # Adding gravitastional softning
                r2 += epsilon**2

                # Computing 1 / r^3
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))

                for k in range(dims):
                    # Distance between bodies
                    dx = system.pos[j, k] - system.pos[i, k]

                    # Updating acceleration
                    a[i, k] += system.G * system.masses[j] * dx * inv_r3

        return a

