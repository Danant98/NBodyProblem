# Importing modules and libraries
from abc import ABC, abstractmethod


class Integrator(ABC):
    """
    Base class for numerical time integrator
    """
    @abstractmethod
    def step(self, system, gravity, dt):
        """
        Advance system by one timestep
        """
        pass

    def reset(self):
        """
        Resets any internal state used by the integrator
        """
        pass