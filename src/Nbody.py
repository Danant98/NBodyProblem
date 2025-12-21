#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, Applied physics and mathematics'

# Importing libraries and modules
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from Sim import simulation as sim

class nBody:

    def __init__(self, 
                 N: int, 
                 max_t: int = 10, 
                 G: float = 0.1, 
                 screen_size: tuple = (2.0, 2.0),
                 grid_points: int = 1000,
                 time_points: int = 100,
                 masses: None | list = None,
                 speed_factor: float = 0.1,
                 speed: None | list[list[float]] = None,
                 pos: None | list[list[float]] = None
                 ) -> None:
        # Number of particles, graviatational constant and total 
        self.N = N
        self.G = G 
        self.max_t = max_t
        self.speed_factor = speed_factor
        self.time, self.dt = np.linspace(0, max_t, time_points, retstep = True)
        
        # Setting size of screen
        self.screen_size = screen_size
        self.x = np.linspace(0., 1., grid_points)
        self.y = np.linspace(0., 1., grid_points)

        # Combinations for computing forces
        self.pairs = list(combinations(range(N), 2))

        # Position and velocity for the different particles
        self.pos = np.random.uniform(-1.0, 1.0, size = (N, 2)) if pos == None else np.array(pos)
        self.vel = np.random.uniform(-1.0, 1.0, size = (N, 2)) if speed == None else np.array(speed)
        if masses != None:
            assert len(masses) == N, "Masses must be specified for each of the partices"
            self.masses = masses
        else:
            self.masses = np.ones(N)

        # self.pos[:, 0] += 0.2; self.pos[:, 1] -= 0.2

        # Initializing array for storing position at each time step
        self.particles = np.zeros((self.time.shape[0], N, 2))
    
    def compute_cm(self) -> None:
        """
        Compute center of mass for the system
        """
        total_mass = np.sum(self.masses)
        cm = np.zeros(2)
        for i in range(self.N):
            cm += self.masses[i] * self.pos[i]
        cm /= total_mass
        # Update positions to center of mass frame
        for i in range(self.N):
            self.pos[i] -= cm
        

    def Euler_cromer(self, body_i: int, body_j: int, ai: np.ndarray, aj: np.ndarray) -> None:
        """
        Numerical scheme for updating position and velocity
        """
        self.vel[body_i] += ai * self.dt * self.speed_factor
        self.vel[body_j] += aj * self.dt* self.speed_factor

        self.pos[body_i] += self.vel[body_i] * self.dt
        self.pos[body_j] += self.vel[body_j] * self.dt        

    
    def run(self) -> None:
        """
        Main loop for simulation
        """
        for i in range(len(self.time)):
            # Iterating over pair of particles to update position and velocity
            for body_i, body_j in self.pairs:
                # Distance between particles
                ri, rj = self.pos[body_i], self.pos[body_j]
                r = ri - rj
                
                # Force from i to j
                ai = -self.G * (self.masses[body_j]) * r / (np.linalg.norm(r)**3)
                aj = self.G * (self.masses[body_i]) * r / (np.linalg.norm(r)**3)

                # Updating velocity and position
                self.Euler_cromer(body_i, body_j, ai, aj)

            self.particles[i] = self.pos
            self.compute_cm()
        
        plt.figure()
        for t in range(len(self.time)):
            plt.clf()
            plt.grid()
            plt.title(f'Time: {self.time[t]:.2f}')
            for n in range(self.N):
                plt.plot(*self.particles[t, n], 'o')
            plt.xlim(self.particles[:, :, 0].min() - 1.0, self.particles[:, :, 0].max() + 1.0)
            plt.ylim(self.particles[:, :, 1].min() - 1.0, self.particles[:, :, 1].max() + 1.0)
            plt.pause(0.01)
        plt.show()
        