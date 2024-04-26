#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt


class NBody:

    def __init__(self, N: int, T: int, G: float = 1.0) -> None:
        # Number of particles
        self.N = N
        self.G = G

        # List containg all particles
        self.particles = []


    
    def Compute_force(self, particle1: np.ndarray, particle2: np.ndarray) -> float:
        """
        Computing the force bewteen two particles
        """
        raise NotImplementedError

    
    def run(self) -> None:
        """
        Main loop for simulatio
        """
        raise NotImplementedError



