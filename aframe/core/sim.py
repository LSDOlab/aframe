import numpy as np
# import csdl_alpha as csdl
import aframe as af
from scipy.integrate import solve_ivp


class Simulation:

    def __init__(self, M, K, F):
        self.M = M
        self.K = K
        self.F = F

    def _ode(self, t, y):
        return 

    def evaluate(self):
        pass