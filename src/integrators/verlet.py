# Importing libraries and modules
from .base import Integrator

class Verlet(Integrator):
    """
    Verlet numerical integrator
    """
    def __init__(self) -> None:
        self.a = None

    def reset(self) -> None:
        self.a = None

    def step(self, system, gravity, dt) -> None:

        # Checking if the acceleration is computed for the system
        if self.a is None:
            self.a = gravity.acceleration(system)

        # Computing velocity and position for a given system
        system.vel += 0.5 * self.a * dt
        system.pos += system.vel * dt

        # Computing new acceleration and updating system velocity
        anew = gravity.acceleration(system)
        system.vel += 0.5 * anew * dt

        # Updating acceleration
        self.a = anew