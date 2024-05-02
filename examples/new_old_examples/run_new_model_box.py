import numpy as np
import csdl
import python_csdl_backend
from aframe.core.beam_model import BeamModel
from aframe.core.dataclass import Beam, BoundaryCondition, Joint, Material
from aframe.utils.plot import plot_box
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update(plt.rcParamsDefault)


num_nodes = 11
aluminum = Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)
wing = Beam(name='wing', num_nodes=num_nodes, material=aluminum, cs='box')
boundary_condition_1 = BoundaryCondition(beam=wing, node=5)

class Run(csdl.Model):
    def initialize(self):
        pass
    def define(self):

        wing_mesh = np.zeros((num_nodes, 3))
        wing_mesh[:, 1] = np.linspace(-20, 20, num_nodes)
        self.create_input('wing_mesh', shape=(num_nodes, 3), val=wing_mesh)

        wing_forces = np.zeros((num_nodes, 3))
        wing_forces[:, 2] = 800
        self.create_input('wing_forces', shape=(num_nodes, 3), val=wing_forces)

        self.create_input('wing_width', shape=(wing.num_elements), val=0.7)
        self.create_input('wing_height', shape=(wing.num_elements), val=0.5)
        self.create_input('wing_tweb', shape=(wing.num_elements), val=0.001)
        self.create_input('wing_ttop', shape=(wing.num_elements), val=0.001)
        self.create_input('wing_tbot', shape=(wing.num_elements), val=0.001)

        wing_thickness = np.ones(wing.num_elements)*0.001
        self.create_input('wing_thickness', shape=(wing.num_elements), val=wing_thickness)

        self.add(BeamModel(beams=[wing],
                           boundary_conditions=[boundary_condition_1],
                           joints=[]))
        
        






if __name__ == '__main__':

    sim = python_csdl_backend.Simulator(Run())
    sim.run()

    undeformed_wing_mesh = sim['wing_mesh']
    deformed_wing_mesh = sim['wing_deformed_mesh']
    wing_width = sim['wing_width']
    wing_height = sim['wing_height']
    wing_stress = sim['wing_stress']

    # plot the box cross sections
    undeformed_vertices = plot_box(undeformed_wing_mesh, wing_width, wing_height)
    deformed_vertices = plot_box(deformed_wing_mesh, wing_width, wing_height)
    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=35, azim=-10)
    ax.set_box_aspect((1, 2, 1))

    ax.scatter(undeformed_wing_mesh[:,0], undeformed_wing_mesh[:,1], undeformed_wing_mesh[:,2], color='yellow', s=50)
    ax.plot(undeformed_wing_mesh[:,0], undeformed_wing_mesh[:,1], undeformed_wing_mesh[:,2])
    ax.scatter(deformed_wing_mesh[:,0], deformed_wing_mesh[:,1], deformed_wing_mesh[:,2], color='blue', s=50)
    ax.plot(deformed_wing_mesh[:,0], deformed_wing_mesh[:,1], deformed_wing_mesh[:,2], color='blue')

    for i in range(wing.num_elements):
        ax.add_collection3d(Poly3DCollection(undeformed_vertices[i], facecolors='black', linewidths=1, edgecolors='red', alpha=0.4))
        ax.add_collection3d(Poly3DCollection(deformed_vertices[i], facecolors='cyan', linewidths=1, edgecolors='red', alpha=0.4))


    ax.set_xlim(-1, 1)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-2, 2)
    # ax.grid(False)
    plt.axis('off')

    plt.show()