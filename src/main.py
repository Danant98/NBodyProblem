#!/usr/bin/env python

from Nbody import nBody

if __name__ == '__main__':
    # Defining mass of bodies
    MS = 1.989E30 # Mass of sun in kg
    ME = 5.97219E24 # Mass of earth in kg
    MJ = 1.898E27 # Mass of Jupiter in kg
    
    nb = nBody(3, 
               masses = [MS / MS, ME / MS, MJ / MS], 
               pos = [[0.0, 0.0], [1.0, 0.0], [5.2, 0.0]],
               speed = [[0.0, 0.0], [0.0, 0.01720209895], [0.0, 0.007126]],
               max_t = 365 * 12,
               fps = 30
               )
    nb.run()


