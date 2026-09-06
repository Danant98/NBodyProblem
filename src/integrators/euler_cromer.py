# Importing libraries and modules
from .base import Integrator

class Euler_Cromer(Integrator):
    """
    Euler-Cromer time integrator.
    """
    def step(self, system, gravity, dt) -> None:
        a = gravity.acceleration(system)

        system.vel += a * dt
        system.pos += system.vel * dt