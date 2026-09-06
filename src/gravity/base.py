# Importing modules and libraries
from abc import ABC, abstractmethod
import numpy as np

class Gravity_Solver(ABC):
    """
    Base class for computing gravitational force and acceleration
    """

    @abstractmethod
    def acceleration(self, system: np.ndarray) -> np.ndarray:
        """
        Method for computing acceleration
        """
        pass

