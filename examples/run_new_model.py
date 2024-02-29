import numpy as np
import csdl
import python_csdl_backend
from aframe.core.beam_model import BeamModel
from aframe.core.dataclass import Beam, BoundaryCondition, Joint, Material

num_nodes = 10
aluminum = Material(name='aluminum', E=69E9, G=26E9, rho=2700)
wing = Beam(name='wing', num_nodes=num_nodes, material=aluminum, cs='tube')
boundary_condition_1 = BoundaryCondition(beam=wing, node=0)

class Run(csdl.Model):
    def initialize(self):
        pass
    def define(self):

        mesh = np.zeros((num_nodes,3))
        mesh[:,1] = np.linspace(-20,20,num_nodes)
        self.create_input('wing_mesh', shape=(num_nodes, 3), val=mesh)

        # solve the beam group:
        self.add(BeamModel(beams=[wing],
                           boundary_conditions=[boundary_condition_1],
                           joints=[]))
        
        






if __name__ == '__main__':

    sim = python_csdl_backend.Simulator(Run())
    sim.run()