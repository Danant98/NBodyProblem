#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, Applied physics and mathematics'

# Importing libraries and modules
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

class nBody:

    def __init__(self, 
                 N: int, 
                 max_t: int = 365, 
                 G: float = 2.9591220828559093e-4,  # AU^3 / (solar_mass * day^2)
                 time_points: int = 1000,
                 masses: None | list = None, 
                 speed_factor: float = 0.1,
                 pos: None | list[list[float]] = None,
                 speed: None | list[list[float]] = None
                 ) -> None:
        # Number of particles, graviatational constant and total 
        self.N = N
        self.G = G 
        self.max_t = max_t
        self.speed_factor = speed_factor
        self.time, self.dt = np.linspace(0, max_t, time_points, retstep = True)
        
        # Combinations for computing forces
        self.pairs = list(combinations(range(N), 2))

        # Position and velocity for the different particles
        self.pos = np.random.uniform(-1.0, 1.0, size = (N, 2)) if pos == None else np.array(pos)
        self.vel = np.random.uniform(-1.0, 1.0, size = (N, 2)) if speed == None else np.array(speed)
        if masses != None:
            assert len(masses) == N, "Masses must be specified for each of the partices"
            self.masses = np.array(masses)
        else:
            self.masses = np.ones(N)

        # Initializing array for storing position at each time step
        self.particles = np.zeros((self.time.shape[0], N, 2))
    
    def compute_cm(self) -> None:
        """
        Compute center of mass for the system
        """
        # Compute the total mass
        M = np.sum(self.masses)
        # Resetting the COM position and velocity of the system
        self.pos -= np.einsum("i, ij -> j", self.masses, self.pos) / M
        self.vel -= np.einsum("i, ij -> j", self.masses, self.vel) / M
        

    def Euler_cromer(self, a: np.ndarray) -> None:
        """
        Numerical scheme for updating position and velocity
        """
        self.vel = self.vel + a * self.dt
        self.pos = self.pos + self.vel * self.dt
    
    def run(self) -> None:
        """
        Main loop for simulation
        """
        for i in range(len(self.time)):
            # Creating acceleration array
            a = np.zeros((self.N, 2))

            # Creating displacement matrix
            R_ij = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]

            # Computing the distance
            r_norm = np.linalg.norm(R_ij, axis = 2)

            # Computing 1/r^3
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                inv_r_cubed = 1.0 / (r_norm * r_norm * r_norm)

            # Setting diagonal elements to zero
            np.fill_diagonal(inv_r_cubed, 0.0)
            
            # Computing acceleration array
            a[:] = self.G * np.einsum("ijk, ij, i -> jk", R_ij, inv_r_cubed, self.masses)

            # Computing velocity and position
            self.Euler_cromer(a)

            # Storing position
            self.particles[i] = self.pos

        plt.figure()
        for t in range(len(self.time)):
            plt.clf()
            plt.grid()
            plt.title(f'Day: {self.time[t]:.2f}')
            for n in range(self.N):
                plt.plot(*self.particles[t, n], 'o')
            plt.xlim(self.particles[:, :, 0].min() - 1.0, self.particles[:, :, 0].max() + 1.0)
            plt.ylim(self.particles[:, :, 1].min() - 1.0, self.particles[:, :, 1].max() + 1.0)
            plt.pause(0.01)
        plt.show()
        