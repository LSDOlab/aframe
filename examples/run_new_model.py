import numpy as np
import csdl
import python_csdl_backend
from aframe.core.beam_model import BeamModel
from aframe.core.dataclass import Beam, BoundaryCondition, Joint, Material
from aframe.utils.plot import plot_box, plot_circle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update(plt.rcParamsDefault)
import pickle

np.set_printoptions(edgeitems=30, linewidth=100,)

num_nodes = 21
aluminum = Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)
wing = Beam(name='wing', num_nodes=num_nodes, material=aluminum, cs='tube')
fuselage = Beam(name='fuselage', num_nodes=num_nodes, material=aluminum, cs='tube')
boundary_condition_1 = BoundaryCondition(beam=wing, node=10)
joint_1 = Joint(beams=[wing, fuselage], nodes=[10, 10])

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

        wing_forces = np.zeros((num_nodes, 3))
        wing_forces[:, 2] = 20000
        self.create_input('wing_forces', shape=(num_nodes, 3), val=wing_forces)

        fuse_forces = np.zeros((num_nodes, 3))
        fuse_forces[:, 2] = 20000
        self.create_input('fuselage_forces', shape=(num_nodes, 3), val=fuse_forces)

        self.create_input('wing_radius', shape=(wing.num_elements), val=0.5)
        self.create_input('wing_thickness', shape=(wing.num_elements), val=0.001)
        self.create_input('fuselage_radius', shape=(fuselage.num_elements), val=0.5)
        self.create_input('fuselage_thickness', shape=(fuselage.num_elements), val=0.001)

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
    wing_stress = sim['wing_stress']
    wing_radius = sim['wing_radius']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=35, azim=-10)
    ax.set_box_aspect((1, 2, 1))

    ax.scatter(undeformed_wing_mesh[:,0], undeformed_wing_mesh[:,1], undeformed_wing_mesh[:,2], color='yellow', edgecolor='black', s=50)
    ax.plot(undeformed_wing_mesh[:,0], undeformed_wing_mesh[:,1], undeformed_wing_mesh[:,2])
    ax.scatter(deformed_wing_mesh[:,0], deformed_wing_mesh[:,1], deformed_wing_mesh[:,2], color='blue', s=50)
    ax.plot(deformed_wing_mesh[:,0], deformed_wing_mesh[:,1], deformed_wing_mesh[:,2], color='blue')

    ax.scatter(undeformed_fuselage_mesh[:,0], undeformed_fuselage_mesh[:,1], undeformed_fuselage_mesh[:,2], color='red', s=50)
    ax.plot(undeformed_fuselage_mesh[:,0], undeformed_fuselage_mesh[:,1], undeformed_fuselage_mesh[:,2])
    ax.scatter(deformed_fuselage_mesh[:,0], deformed_fuselage_mesh[:,1], deformed_fuselage_mesh[:,2], color='green', s=50)
    ax.plot(deformed_fuselage_mesh[:,0], deformed_fuselage_mesh[:,1], deformed_fuselage_mesh[:,2], color='green')

    vertices = plot_circle(undeformed_wing_mesh, wing_radius*5, num_circle=20)
    for i in range(wing.num_elements):
        ax.add_collection3d(Poly3DCollection(vertices[i], facecolors='black', linewidths=1, edgecolors='red', alpha=0.4))

    deformed_vertices = plot_circle(deformed_wing_mesh, wing_radius*5, num_circle=20)
    for i in range(wing.num_elements):
        ax.add_collection3d(Poly3DCollection(deformed_vertices[i], facecolors='cyan', linewidths=1, edgecolors='red', alpha=0.4))


    # ax.axis('equal')
    # plt.axis('off')
    # plt.show()

    # plt.plot(wing_stress)
    # plt.show()

    # np.set_printoptions(edgeitems=30, linewidth=100000,)
    # print(sim['global_mass_matrix'])

    with open('matrix1.pkl', 'wb') as f:
            pickle.dump(sim['global_stiffness_matrix_pre'], f)