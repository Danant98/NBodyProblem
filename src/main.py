#!/usr/bin/env python

from Nbody import nBody

if __name__ == '__main__':
    nb = nBody(2, 
               masses = [0.01, 100.0], 
               pos = [[10, 10], [0.0, 0.0]],
               speed = [[-0.1, 0.1], [0.0, 0.0]]
               )
    nb.run()


