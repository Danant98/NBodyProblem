#!/usr/bin/env python

from Nbody import nBody

if __name__ == '__main__':
    # Defining mass of bodies
    MS = 1.989E30 # Mass of sun in kg
    ME = 5.97219E24 # Mass of earth in kg
    MJ = 1.898E27 # Mass of Jupiter in kg
    MM = 6.39E23 # Mass of Mars in kg

    nb = nBody(4, 
               masses = [MS / MS, ME / MS, MJ / MS, MM / MS], 
               pos = [[0.0, 0.0], [1.0, 0.0], [5.2, 0.0], [1.524, 0.0]],
               speed = [[0.0, 0.0], [0.0, 0.01720209895], [0.0, 0.007544], [0.0, 0.013934]],
               max_t = 3 * 365,
               fps = 60,
               colors = ['yellow', 'blue', 'orange', 'red'],
               labels = ['Sun', 'Earth', 'Jupiter', 'Mars'],
               dark_mode = True
               )
    
    nb.run()


