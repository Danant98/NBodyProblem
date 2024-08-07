#!/usr/bin/env python

from Nbody import nBody
import numpy as np

if __name__ == '__main__':
    nb = nBody(3, masses = [1.0, 10.0, 1.0])
    nb.run()


