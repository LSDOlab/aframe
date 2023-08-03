import csdl
import numpy as np

class Buckle(csdl.Model):
    def initialize(self):
        self.parameters.declare('element_name')
        self.parameters.declare('E')
        self.parameters.declare('k', default=3.0)
        self.parameters.declare('v', default=0.33)
    def define(self):
        element_name = self.parameters['element_name']
        k = self.parameters['k']
        v = self.parameters['v']
        E = self.parameters['E']


        w = self.declare_variable(element_name + '_w')
        h = self.declare_variable(element_name + '_h')

        tcap = self.declare_variable(element_name + '_tcap')
        tweb = self.declare_variable(element_name + '_tweb')

        sp_cap = k*E*((tcap/w)**2)/(1 - v**2)
        # sp_web = k*E*((tweb/h)**2)/(1 - v**2)

        self.register_output(element_name + 'sp_cap', sp_cap)
        # self.register_output('sp_web', sp_web)


        stress_array = self.declare_variable(element_name + '_stress_array', shape=(5))

        cap_stress = (stress_array[0] + stress_array[1])/2

        self.register_output(element_name + 'bkl_ratio', cap_stress/sp_cap)