import numpy as np
import csdl
import python_csdl_backend
from aframe.core.beam_model import BeamModel
from aframe.core.dataclass import Beam, BoundaryCondition, Joint, Material
from aframe.utils.plot import plot_box, plot_circle, plot_mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update(plt.rcParamsDefault)
from stl import mesh


aluminum = Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)

wingspan = 25
half_span = wingspan / 2
quarter_span = wingspan / 4

num_center_wing = 11
center_wing_mesh = np.zeros((num_center_wing, 3))
center_wing_mesh[:, 0] = 3.25
center_wing_mesh[:, 1] = np.linspace(-quarter_span, quarter_span, num_center_wing)
center_wing = Beam(name='center_wing', num_nodes=num_center_wing, material=aluminum, cs='box')
boundary_condition_1 = BoundaryCondition(beam=center_wing, node=5)

num_outer_wing = 6
right_outer_wing_mesh = np.zeros((num_outer_wing, 3))
right_outer_wing_mesh[:, 0] = 3.25
right_outer_wing_mesh[:, 1] = np.linspace(quarter_span, half_span, num_outer_wing)
right_outer_wing_mesh[:, 2] = np.linspace(0, quarter_span * np.tan(np.deg2rad(15)), num_outer_wing)
right_outer_wing = Beam(name='right_outer_wing', num_nodes=num_outer_wing, material=aluminum, cs='box')
joint_1 = Joint(beams=[center_wing, right_outer_wing], nodes=[num_center_wing - 1, 0])

left_outer_wing_mesh = np.zeros((num_outer_wing, 3))
left_outer_wing_mesh[:, 0] = 3.25
left_outer_wing_mesh[:, 1] = np.linspace(-quarter_span, -half_span, num_outer_wing)
left_outer_wing_mesh[:, 2] = np.linspace(0, quarter_span * np.tan(np.deg2rad(15)), num_outer_wing)
left_outer_wing = Beam(name='left_outer_wing', num_nodes=num_outer_wing, material=aluminum, cs='box')
joint_2 = Joint(beams=[center_wing, left_outer_wing], nodes=[0, 0])

num_fwd_fuse = 4
fwd_fuse_mesh = np.zeros((num_fwd_fuse, 3))
fwd_fuse_mesh[:, 0] = np.linspace(0, 3.25, num_fwd_fuse)
fwd_fuse = Beam(name='fwd_fuse', num_nodes=num_fwd_fuse, material=aluminum, cs='tube')
joint_3 = Joint(beams=[center_wing, fwd_fuse], nodes=[5, num_fwd_fuse - 1])

num_aft_fuse = 6
aft_fuse_mesh = np.zeros((num_aft_fuse, 3))
aft_fuse_mesh[:, 0] = np.linspace(3.25, 8.5, num_aft_fuse)
aft_fuse = Beam(name='aft_fuse', num_nodes=num_aft_fuse, material=aluminum, cs='tube')
joint_4 = Joint(beams=[center_wing, aft_fuse], nodes=[5, 0])

num_vtail = 4
vtail_mesh = np.zeros((num_vtail, 3))
vtail_mesh[:, 0] = np.linspace(8.5, 8.625, num_vtail)
vtail_mesh[:, 2] = np.linspace(0, 1.5, num_vtail)
vtail = Beam(name='vtail', num_nodes=num_vtail, material=aluminum, cs='box')
joint_5 = Joint(beams=[aft_fuse, vtail], nodes=[num_aft_fuse - 1, 0])
# beams directly along the z axis will fail

num_htail = 5
htail_mesh = np.zeros((num_htail, 3))
htail_mesh[:, 0] = 8.625
htail_mesh[:, 1] = np.linspace(-2, 2, num_htail)
htail_mesh[:, 2] = 1.5
htail = Beam(name='htail', num_nodes=num_htail, material=aluminum, cs='box')
joint_6 = Joint(beams=[vtail, htail], nodes=[num_vtail - 1, 2])





class Run(csdl.Model):
    def define(self):

        self.create_input('center_wing_mesh', shape=(num_center_wing, 3), val=center_wing_mesh)
        self.create_input('right_outer_wing_mesh', shape=(num_outer_wing, 3), val=right_outer_wing_mesh)
        self.create_input('left_outer_wing_mesh', shape=(num_outer_wing, 3), val=left_outer_wing_mesh)
        self.create_input('fwd_fuse_mesh', shape=(num_fwd_fuse, 3), val=fwd_fuse_mesh)
        self.create_input('aft_fuse_mesh', shape=(num_aft_fuse, 3), val=aft_fuse_mesh)
        self.create_input('vtail_mesh', shape=(num_vtail, 3), val=vtail_mesh)
        self.create_input('htail_mesh', shape=(num_htail, 3), val=htail_mesh)

        self.create_input('center_wing_width', shape=(center_wing.num_elements), val=1)
        self.create_input('center_wing_height', shape=(center_wing.num_elements), val=0.4)
        self.create_input('center_wing_tweb', shape=(center_wing.num_elements), val=0.001)
        self.create_input('center_wing_ttop', shape=(center_wing.num_elements), val=0.001)
        self.create_input('center_wing_tbot', shape=(center_wing.num_elements), val=0.001)

        self.create_input('right_outer_wing_width', shape=(right_outer_wing.num_elements), val=1)
        self.create_input('right_outer_wing_height', shape=(right_outer_wing.num_elements), val=0.4)
        self.create_input('right_outer_wing_tweb', shape=(right_outer_wing.num_elements), val=0.001)
        self.create_input('right_outer_wing_ttop', shape=(right_outer_wing.num_elements), val=0.001)
        self.create_input('right_outer_wing_tbot', shape=(right_outer_wing.num_elements), val=0.001)

        self.create_input('left_outer_wing_width', shape=(left_outer_wing.num_elements), val=1)
        self.create_input('left_outer_wing_height', shape=(left_outer_wing.num_elements), val=0.4)
        self.create_input('left_outer_wing_tweb', shape=(left_outer_wing.num_elements), val=0.001)
        self.create_input('left_outer_wing_ttop', shape=(left_outer_wing.num_elements), val=0.001)
        self.create_input('left_outer_wing_tbot', shape=(left_outer_wing.num_elements), val=0.001)

        self.create_input('fwd_fuse_radius', shape=(fwd_fuse.num_elements), val=0.2)
        self.create_input('fwd_fuse_thickness', shape=(fwd_fuse.num_elements), val=0.001)

        self.create_input('aft_fuse_radius', shape=(aft_fuse.num_elements), val=0.2)
        self.create_input('aft_fuse_thickness', shape=(aft_fuse.num_elements), val=0.001)

        self.create_input('vtail_width', shape=(vtail.num_elements), val=0.7)
        self.create_input('vtail_height', shape=(vtail.num_elements), val=0.5)
        self.create_input('vtail_tweb', shape=(vtail.num_elements), val=0.001)
        self.create_input('vtail_ttop', shape=(vtail.num_elements), val=0.001)
        self.create_input('vtail_tbot', shape=(vtail.num_elements), val=0.001)

        self.create_input('htail_width', shape=(htail.num_elements), val=0.7)
        self.create_input('htail_height', shape=(htail.num_elements), val=0.5)
        self.create_input('htail_tweb', shape=(htail.num_elements), val=0.001)
        self.create_input('htail_ttop', shape=(htail.num_elements), val=0.001)
        self.create_input('htail_tbot', shape=(htail.num_elements), val=0.001)

        # wing_forces = np.zeros((num_nodes, 3))
        # wing_forces[:, 2] = 20000
        # self.create_input('wing_forces', shape=(num_nodes, 3), val=wing_forces)

        # fuse_forces = np.zeros((num_nodes, 3))
        # fuse_forces[:, 2] = 1000
        # self.create_input('fuselage_forces', shape=(num_nodes, 3), val=fuse_forces)

        # self.create_input('wing_radius', shape=(wing.num_elements), val=0.5)
        # self.create_input('wing_thickness', shape=(wing.num_elements), val=0.001)

        self.add(BeamModel(beams=[center_wing, right_outer_wing, left_outer_wing, fwd_fuse, aft_fuse, vtail, htail],
                           boundary_conditions=[boundary_condition_1],
                           joints=[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]))
        

if __name__ == '__main__':

    sim = python_csdl_backend.Simulator(Run())
    sim.run()

    center_wing_mesh = sim['center_wing_mesh']
    right_outer_wing_mesh = sim['right_outer_wing_mesh']
    left_outer_wing_mesh = sim['left_outer_wing_mesh']
    fwd_fuse_mesh = sim['fwd_fuse_mesh']
    aft_fuse_mesh = sim['aft_fuse_mesh']
    vtail_mesh = sim['vtail_mesh']
    htail_mesh = sim['htail_mesh']


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=35, azim=-10)
    # ax.set_box_aspect((1, 2, 1))

    size = 30
    plot_mesh(ax=ax, mesh=center_wing_mesh, size=size, line_color='black', marker_color='yellow', edge_color='black')
    plot_mesh(ax=ax, mesh=right_outer_wing_mesh, size=size, line_color='black', marker_color='yellow', edge_color='black')
    plot_mesh(ax=ax, mesh=left_outer_wing_mesh, size=size, line_color='black', marker_color='yellow', edge_color='black')
    plot_mesh(ax=ax, mesh=fwd_fuse_mesh, size=size, line_color='black', marker_color='yellow', edge_color='black')
    plot_mesh(ax=ax, mesh=aft_fuse_mesh, size=size, line_color='black', marker_color='yellow', edge_color='black')
    plot_mesh(ax=ax, mesh=vtail_mesh, size=size, line_color='black', marker_color='yellow', edge_color='black')
    plot_mesh(ax=ax, mesh=htail_mesh, size=size, line_color='black', marker_color='yellow', edge_color='black')


    airplane = mesh.Mesh.from_file('zephyr.stl')
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(airplane.vectors, alpha=0.2, facecolor='black'))


    vertices = plot_circle(fwd_fuse_mesh, radius=sim['fwd_fuse_radius'], num_circle=20)
    for i in range(fwd_fuse.num_elements):
        ax.add_collection3d(Poly3DCollection(vertices[i], facecolors='cyan', linewidths=1, edgecolors='red', alpha=0.4))

    vertices = plot_circle(aft_fuse_mesh, radius=sim['aft_fuse_radius'], num_circle=20)
    for i in range(aft_fuse.num_elements):
        ax.add_collection3d(Poly3DCollection(vertices[i], facecolors='cyan', linewidths=1, edgecolors='red', alpha=0.4))

    vertices = plot_box(center_wing_mesh, width=sim['center_wing_width'], height=sim['center_wing_height'])
    for i in range(center_wing.num_elements):
        ax.add_collection3d(Poly3DCollection(vertices[i], facecolors='cyan', linewidths=1, edgecolors='red', alpha=0.4))

    vertices = plot_box(right_outer_wing_mesh, width=sim['right_outer_wing_width'], height=sim['right_outer_wing_height'])
    for i in range(right_outer_wing.num_elements):
        ax.add_collection3d(Poly3DCollection(vertices[i], facecolors='cyan', linewidths=1, edgecolors='red', alpha=0.4))

    vertices = plot_box(left_outer_wing_mesh, width=sim['left_outer_wing_width'], height=sim['left_outer_wing_height'])
    for i in range(left_outer_wing.num_elements):
        ax.add_collection3d(Poly3DCollection(vertices[i], facecolors='cyan', linewidths=1, edgecolors='red', alpha=0.4))

    ax.axis('equal')
    plt.axis('off')
    plt.show()