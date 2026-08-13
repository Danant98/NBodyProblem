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
                 colors: None | list[str] = None,
                 labels: None | list[str] = None,
                 fps: int = 60,
                 dark_mode: bool = False,
                 plot_labels: bool = True,
                 integrator: str = "euler-cromer",
                 save_animation: bool = False,
                 save_path: None | str = None
                 ) -> None:
        # Number of particles, graviatational constant and total 
        self.N = N
        self.G = G
        self.max_t = max_t
        self.time, self.dt = np.linspace(0, max_t, time_points, retstep = True)
        self.fps = fps
        self.dark_mode = dark_mode
        self.plot_labels = plot_labels

        self.save_animation = save_animation
        self.save_path = save_path
        if save_animation == True and save_path is None:
            raise ValueError("To save animation ")

        # Setting the integrator for the simulation
        self.integrator = integrator
        assert self.integrator in ["euler-cromer", "verlet"], "Integrator must be either 'euler-cromer' or 'verlet'"

        # Colors for the different particles and labels for the legend
        diff_cols = [value for key, value in mcolors.TABLEAU_COLORS.items()]
        self.colors = np.random.choice(diff_cols, size = N) if colors == None else colors
        if plot_labels:
            self.labels = labels if labels != None else [f"Particle {i+1}" for i in range(N)]
            assert len(self.labels) == N, "Labels must be specified for each of the partices"

        # Checking that the number of colors and labels matches the number of particles
        assert len(self.colors) == N, "Colors must be specified for each of the partices"

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
    
    def numerical_integrator(self, a: np.ndarray) -> None:
        """
        Numerical integrator for updating position and velocity
        """
        if self.integrator == "euler-cromer":
            self.vel += a * self.dt
            self.pos += self.vel * self.dt
        elif self.integrator == "verlet":
            self.pos += self.vel * self.dt + 0.5 * a * self.dt**2
            # Compute new acceleration
            a_new = self.acceleration()
            self.vel += 0.5 * (a + a_new) * self.dt

    
    def acceleration(self, epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute acceleration for each particle
        """
        # Creating displacement matrix
        # R_ij = pos_j - pos_i for use in a_i = G * sum_j m_j * R_ij / r^3
        R_ij = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :]

        # Computing the distance
        r_norm = np.linalg.norm(R_ij, axis = 2)

        # Computing 1/r^3
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            inv_r_cubed = 1.0 / (r_norm**2 + epsilon**2)**(3/2)  # Adding epsilon to avoid singularity

        # Setting diagonal elements to zero
        np.fill_diagonal(inv_r_cubed, 0.0)
        
        # Computing acceleration array: a_i,k = G * sum_j R_ij,k * inv_r_cubed[i,j] * m_j
        a = self.G * np.einsum("ijk, ij, j -> ik", R_ij, inv_r_cubed, self.masses)
        return a

    
    def run(self) -> None:
        """
        Main loop for simulation
        """
        # Center the system so the center of mass is at the origin
        self.compute_cm()

        # Storing initial positions
        self.particles[0] = self.pos

        for i in range(len(self.time)):
            # Computing acceleration
            a = self.acceleration()

            # Computing velocity and position
            self.numerical_integrator(a)

            # Storing position
            self.particles[i] = self.pos

        self.animate()

    def animate(self) -> None:
        """
        Animation of the N-body system
        """
        # Setting dark mode if specified
        if self.dark_mode:
            plt.style.use('dark_background')

        if self.fps <= 0:
            raise ValueError("FPS must be a positive integer")

        if not np.isfinite(self.particles).all():
            raise ValueError("Particles contain non-finite values. Check the simulation parameters.")
    
        fig, ax = plt.subplots()
        ax.set_xlim(self.particles[:, :, 0].min() - 1.0, self.particles[:, :, 0].max() + 1.0)
        ax.set_ylim(self.particles[:, :, 1].min() - 1.0, self.particles[:, :, 1].max() + 1.0)

        scats = []
        # Use a scatter plot so we can update positions for each particle separately
        if self.plot_labels:
            for i, label in enumerate(self.labels):
                scat = ax.scatter(self.particles[0, i, 0], self.particles[0, i, 1], s = 40, c = self.colors[i], label = label)
                scats.append(scat)
        else:
            for i in range(self.N):
                scat = ax.scatter(self.particles[0, i, 0], self.particles[0, i, 1], s = 40, c = self.colors[i])
                scats.append(scat)

        ax.set_xlabel(r"X (AU)")
        ax.set_ylabel(r"Y (AU)")
        ax.set_aspect('equal', adjustable = 'box')
        if self.plot_labels:
            ax.legend()
        title = ax.set_title(f'Year {self.time[0] // 365:.0f}, Day: {self.time[0] % 365:.0f}, FPS {self.fps}')

        def init():
            for i, scat in enumerate(scats):
                scat.set_offsets(self.particles[0, i])

            title.set_text(f'Year {self.time[0] // 365:.0f}, Day: {self.time[0] % 365:.0f}, FPS {self.fps}')
            return scats, title

        def update(frame):
            for i in range(self.N):
                scats[i].set_offsets(self.particles[frame, i])
            title.set_text(f'Year {self.time[frame] // 365:.0f}, Day: {self.time[frame] % 365:.0f}, FPS {self.fps}')
            return scats, title

        interval_ms = 1000.0 / self.fps
        self.ani = FuncAnimation(fig, update, init_func = init, frames = len(self.time), blit = False, interval = interval_ms, repeat = True)

        if self.save_animation:
            self.ani.save(f'{self.save_path}.gif', writer = "pillow", fps = self.fps)

        plt.show()





