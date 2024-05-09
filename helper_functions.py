import numpy as np

def potential(x):

    # Calculates potential, given distance
    func = 0.5 * x**2

    return func

def T(x):

    return -1/(2*(x**2))

def gauss(x, V0):
    function = -V0*np.exp(-np.abs(x))
    return function