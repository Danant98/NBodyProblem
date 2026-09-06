# Importing modules and libraries
from .base import Gravity_Solver
import numpy as np

class Direct_Numpy(Gravity_Solver):
    """
    Gravity solver using a direct numpy approach
    """
    def acceleration(self, system, epsilon: float = 1e-5) -> np.ndarray:
        """
        Computing acceleration for the system
        
        Gravitational softning: epsilon = 1E-5 by default
        """
        # Creating displacement matrix, R_ij
        R_ij = system.pos[np.newaxis, :, :] - system.pos[:, np.newaxis, :]

        # Computing distance
        r_norm = np.linalg.norm(R_ij, axis = 2)

        # Computing 1/r^3
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            inv_r_cubed  = 1.0 / (r_norm**2 + epsilon**2)**(3 / 2)

        # Setting diagonal element to zero
        np.fill_diagonal(inv_r_cubed, 0.0)

        # Computing acceleration
        a = system.G * np.einsum("ijk,ij, j -> ik", R_ij, inv_r_cubed, system.masses)

        return a

