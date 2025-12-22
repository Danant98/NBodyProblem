#!/usr/bin/env python

from Nbody import nBody

if __name__ == '__main__':
    # Defining mass of bodies
    MS = 1.989E30 # Mass of sun in kg
    ME = 5.97219E24 # Mass of earth in kg
    
    nb = nBody(2, 
               masses = [MS / MS, ME / MS], 
               pos = [[0.0, 0.0], [1.0, 0.0]],
               speed = [[0.0, 0.0], [0.0, 0.01720209895]]
               )
    nb.run()


