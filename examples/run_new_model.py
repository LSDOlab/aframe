import numpy as np
import csdl
import python_csdl_backend
from aframe.core.beam_model import BeamModel
from aframe.core.dataclass import Beam, BoundaryCondition, Joint, Material
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

num_nodes = 5
aluminum = Material(name='aluminum', E=69E9, G=26E9, rho=2700)
wing = Beam(name='wing', num_nodes=num_nodes, material=aluminum, cs='tube')
fuselage = Beam(name='fuselage', num_nodes=num_nodes, material=aluminum, cs='tube')
boundary_condition_1 = BoundaryCondition(beam=wing, node=2)
joint_1 = Joint(beams=[wing, fuselage], nodes=[2, 2])

class Run(csdl.Model):
    def initialize(self):
        pass
    def define(self):

        wing_mesh = np.zeros((num_nodes, 3))
        wing_mesh[:, 1] = np.linspace(-20, 20, num_nodes)
        self.create_input('wing_mesh', shape=(num_nodes, 3), val=wing_mesh)

        fuse_mesh = np.zeros((num_nodes, 3))
        fuse_mesh[:, 0] = np.linspace(-20, 20, num_nodes)
        self.create_input('fuselage_mesh', shape=(num_nodes, 3), val=fuse_mesh)

        forces = np.zeros((num_nodes, 3))
        forces[:, 2] = 1000
        self.create_input('wing_forces', shape=(num_nodes, 3), val=forces)

        self.add(BeamModel(beams=[wing, fuselage],
                           boundary_conditions=[boundary_condition_1],
                           joints=[joint_1]))
        
        






if __name__ == '__main__':

    sim = python_csdl_backend.Simulator(Run())
    sim.run()

    undeformed_wing_mesh = sim['wing_mesh']
    deformed_wing_mesh = sim['wing_deformed_mesh']
    undeformed_fuselage_mesh = sim['fuselage_mesh']
    deformed_fuselage_mesh = sim['fuselage_deformed_mesh']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=35, azim=-10)
    ax.set_box_aspect((1, 4, 1))

    ax.scatter(undeformed_wing_mesh[:,0], undeformed_wing_mesh[:,1], undeformed_wing_mesh[:,2], color='yellow', s=50)
    ax.plot(undeformed_wing_mesh[:,0], undeformed_wing_mesh[:,1], undeformed_wing_mesh[:,2])
    ax.scatter(deformed_wing_mesh[:,0], deformed_wing_mesh[:,1], deformed_wing_mesh[:,2], color='blue', s=50)
    ax.plot(deformed_wing_mesh[:,0], deformed_wing_mesh[:,1], deformed_wing_mesh[:,2], color='blue')

    ax.scatter(undeformed_fuselage_mesh[:,0], undeformed_fuselage_mesh[:,1], undeformed_fuselage_mesh[:,2], color='red', s=50)
    ax.plot(undeformed_fuselage_mesh[:,0], undeformed_fuselage_mesh[:,1], undeformed_fuselage_mesh[:,2])
    ax.scatter(deformed_fuselage_mesh[:,0], deformed_fuselage_mesh[:,1], deformed_fuselage_mesh[:,2], color='green', s=50)
    ax.plot(deformed_fuselage_mesh[:,0], deformed_fuselage_mesh[:,1], deformed_fuselage_mesh[:,2], color='green')

    plt.show()

    # element_loads = sim['element_loads_storage']
    # print(element_loads)
    # plt.plot(element_loads[:, 0])
    # plt.plot(element_loads[:, 1])
    # plt.plot(element_loads[:, 2])
    # #plt.plot(element_loads[:, 3])
    # plt.plot(element_loads[:, 4])
    # plt.plot(element_loads[:, 5])

    # plt.plot(element_loads[:, 6])
    # plt.plot(element_loads[:, 7])
    # plt.plot(element_loads[:, 8])
    # #plt.plot(element_loads[:, 9])
    # plt.plot(element_loads[:, 10])
    # plt.plot(element_loads[:, 11])

    # plt.grid()


    # plt.show()