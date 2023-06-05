import numpy as np
import csdl




class Model(csdl.Model):
    def initialize(self):
        self.parameters.declare('dim')
    def define(self):
        dim = self.parameters['dim']
        

        K = self.declare_variable('K',shape=(dim,dim))
        U = self.declare_variable('U',shape=(dim))
        Fi = self.declare_variable('Fi',shape=(dim))

        R = csdl.matvec(K,U) - Fi

        self.register_output('R', R)