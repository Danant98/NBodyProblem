#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt; plt.style.use('dark_background')


class nBody:

    def __init__(self, 
                 N: int, 
                 max_t: int = 5, 
                 G: float = 6.67408e-11, 
                 screen_size: tuple = (2.0, 2.0),
                 grid_points: int = 1000,
                 time_points: int = 100,
                 masses: None | list = None,
                 speed_factor: float = 0.1
                 ) -> None:
        # Number of particles, graviatational constant and total 
        self.N = N
        self.G = 0.1
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
        self.pos = np.random.uniform(0.2, 0.8, size = (N, 2))
        # self.vel = np.random.uniform(-0.5, 0.5, size = (N, 2))
        self.vel = np.zeros((N, 2))
        if masses != None:
            assert len(masses) == N, "Masses must be specified for each of the partices"
            self.masses = masses
        else:
            self.masses = np.ones(N)

        # self.pos[:, 0] += 0.2; self.pos[:, 1] -= 0.2

        # Initializing array for storing position at each time step
        self.particles = np.zeros((self.time.shape[0], N, 2))

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
        
        plt.figure(figsize=(7, 7))
        for t in range(len(self.time)):
            plt.clf()
            for n in range(self.N):
                plt.plot(*self.particles[t, n], 'wo')
            plt.xlim(self.particles[:, :, 0].min() - 1.0, self.particles[:, :, 0].max() + 1.0)
            plt.ylim(self.particles[:, :, 1].min() - 1.0, self.particles[:, :, 1].max() + 1.0)
            plt.pause(0.01)
        plt.show()
        


