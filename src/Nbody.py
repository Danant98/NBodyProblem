#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


class NBody:

    def __init__(self, 
                 N: int, 
                 max_t: int, 
                 G: float = 1.0, 
                 screen_size: tuple = (1.0, 1.0),
                 grid_points: int = 1000,
                 time_points: int = 100
                 ) -> None:
        # Number of particles, graviatational constant and total 
        self.N = N
        self.G = G
        self.max_t = max_t
        self.time = np.linspace(0, max_t, time_points)
        
        # Setting size of screen
        self.screen_size = screen_size
        self.x = np.linspace(0., 1., grid_points)
        self.y = np.linspace(0., 1., grid_points)

        # Combinations for computing forces
        self.pairs = list(combinations(range(N), 2))

        # Position and velocity for the different particles
        self.pos = np.random.rand(N, 2)
        self.vel = np.zeros((N, 2))

        self.pos[:, 0] += 0.2; self.pos[:, 1] -= 0.2

        # Initializing array for storing position at each time step
        self.particles = np.zeros((max_t, N, 2))

    
    def run(self) -> None:
        """
        Main loop for simulation
        """
        for i in range(self.max_t):
            break
        



