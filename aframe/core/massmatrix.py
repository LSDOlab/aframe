import numpy as np
import csdl
import python_csdl_backend



class MassMatrix(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('bounds', default={})

    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']
        bounds = self.parameters['bounds']