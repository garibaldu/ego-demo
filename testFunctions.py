"""
A test function for GPO to try to optimize
"""

from numpy import dot,sin
import numpy.random as rng

def funcA(x):
    # an inverted parabolic bowl
    y = 1 -dot(x,x)  + 0.01*rng.normal()
    return y

def funcB(x):
    # just noise!
    y = rng.normal()
    return y

def funcC(x):
    # a slope in 1D.
    y = -abs(x) + 0.1*rng.normal()
    return y

def funcD(x):
    # a noisy sinusoid in 1D
    y = sin(x) + 0.1*rng.normal()
    return y

