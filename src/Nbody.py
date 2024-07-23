#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt


class NBody:

    def __init__(self, N: int, 
                 max_t: int, 
                 G: float = 1.0, 
                 screen_size: tuple = (1.0, 1.0),
                 grid_points: int = 1000
                 ) -> None:
        # Number of particles, graviatational constant and total 
        self.N = N
        self.G = G
        self.max_t = max_t
        
        # Setting size of screen
        self.screen_size = screen_size
        self.x = np.linspace(-screen_size[0], screen_size[0], grid_points)
        self.y = np.linspace(-screen_size[1], screen_size[1], grid_points)

        # List containg all particles
        self.particles = []

    
    def Compute_force(self, particle1: np.ndarray, particle2: np.ndarray) -> float:
        """
        Computing the force bewteen two particles
        """
        raise NotImplementedError

    
    def run(self) -> None:
        """
        Main loop for simulation
        """
        raise NotImplementedError



