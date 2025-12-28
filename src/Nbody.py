#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, Applied physics and mathematics'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

class nBody:

    def __init__(self, 
                 N: int, 
                 max_t: int = 365, 
                 G: float = 2.9591220828559093e-4,  # AU^3 / (solar_mass * day^2)
                 time_points: int = 1000,
                 masses: None | list = None,
                 pos: None | list[list[float]] = None,
                 speed: None | list[list[float]] = None,
                 colors: None | list[str] = None
                 ) -> None:
        # Number of particles, graviatational constant and total 
        self.N = N
        self.G = G 
        self.max_t = max_t
        self.time, self.dt = np.linspace(0, max_t, time_points, retstep = True)

        # Colors for the different particles
        diff_cols = [value for key, value in mcolors.TABLEAU_COLORS.items()]
        self.colors = np.random.choice(diff_cols, size = N) if colors == None else colors

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
        self.vel += a * self.dt
        self.pos += self.vel * self.dt
    
    def run(self) -> None:
        """
        Main loop for simulation
        """
        # Center the system so the center of mass is at the origin
        self.compute_cm()
        for i in range(len(self.time)):
            # Creating acceleration array
            a = np.zeros((self.N, 2))

            # Creating displacement matrix
            # R_ij = pos_j - pos_i for use in a_i = G * sum_j m_j * R_ij / r^3
            R_ij = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :]

            # Computing the distance
            r_norm = np.linalg.norm(R_ij, axis = 2)

            # Computing 1/r^3
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                inv_r_cubed = 1.0 / (r_norm * r_norm * r_norm)

            # Setting diagonal elements to zero
            np.fill_diagonal(inv_r_cubed, 0.0)
            
            # Computing acceleration array: a_i,k = G * sum_j R_ij,k * inv_r_cubed[i,j] * m_j
            a[:] = self.G * np.einsum("ijk, ij, j -> ik", R_ij, inv_r_cubed, self.masses)

            # Computing velocity and position
            self.Euler_cromer(a)

            # Storing position
            self.particles[i] = self.pos

        self.animate()

    def animate(self, fps: int = 60) -> None:
        """
        Animation of the N-body system
        """
        fig, ax = plt.subplots()
        ax.set_xlim(self.particles[:, :, 0].min() - 1.0, self.particles[:, :, 0].max() + 1.0)
        ax.set_ylim(self.particles[:, :, 1].min() - 1.0, self.particles[:, :, 1].max() + 1.0)
        
        # Use a scatter plot so we can update positions for each particle separately
        scat = ax.scatter([], [], c=self.colors, s=40)
        plt.grid(True)
        plt.xlabel(r'x (AU)')
        plt.ylabel(r'y (AU)')

        def update(frame):
            scat.set_offsets(self.particles[frame])
            ax.set_title(f'Day: {self.time[frame]:.0f}, FPS {fps}')
            return scat,

        interval_ms = 1000.0 / fps
        ani = FuncAnimation(fig, update, frames = len(self.time), blit = True, interval = interval_ms)
        plt.show()





