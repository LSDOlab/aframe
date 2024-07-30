import csdl_alpha as csdl
import numpy as np
import aframe as af
import pyvista as pv


# feet
s = 4
z = 0
foot_points = np.array([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]])

# base
d = 4
z = 2
base_points = np.array([[-d, 0, z], [0, -d, z], [d, 0, z], [0, d, z]])

# leg attach
y = 3
z = 3.5
leg_points = np.array([[y, y, z], [-y, y, z], [-y, -y, z], [y, -y, z]])

# top frame
q = 3.6
z = 5
top_points = np.array([[-q, 0, z], [0, -q, z], [q, 0, z], [0, q, z]])


points = np.vstack((foot_points, 
                  base_points, 
                  leg_points,
                  top_points,
                  ))

edges = np.array([[0, 4], 
                  [0, 5],
                  [0, 10],
                  [1, 5],
                  [1, 6],
                  [1, 11],
                  [2, 6],
                  [2, 7],
                  [2, 8],
                  [3, 7],
                  [3, 4],
                  [3, 9],
                  [4, 10],
                  [5, 10],
                  [5, 11],
                  [6, 11],
                  [7, 8],
                  [6, 8],
                  [7, 9],
                  [4, 9],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 4],
                  [12, 13],
                  [13, 14],
                  [14, 15],
                  [15, 12],
                  [9, 12],
                  [9, 15],
                  [8, 14],
                  [8, 15],
                  [11, 14],
                  [11, 13],
                  [10, 12],
                  [10, 13],
                  ])


joints = np.array()


nodes_per_edge = 5
meshes = af.mesh_from_points_and_edges(points, edges, nodes_per_edge)

"""
plotter = pv.Plotter()
for i in range(meshes.shape[0]):
    mesh = meshes[i, :, :]

    af.plot_mesh(plotter, mesh, color='lightblue', line_width=5)

# plot the feet
for i in range(4):
    cyl = pv.Cylinder(center=foot_points[i, :], direction=[0, 0, 1], radius=0.5, height=0.1)
    plotter.add_mesh(cyl, color='red')

# toroidal tank
torus = pv.ParametricTorus(ringradius=2.375, crosssectionradius=1.4, center=(0, 0, 3.5))
plotter.add_mesh(torus, color='skyblue')

# helium
# num_helium_tanks = 6
# r = 2.25
# for i in range(num_helium_tanks):
#     angle = (i * (2 * np.pi) / num_helium_tanks) + np.pi / 4
#     x, y = r * np.cos(angle), r * np.sin(angle)
#     sph = pv.Sphere(center=(x, y, 5.75), radius=0.7)
#     plotter.add_mesh(sph, color='green')


# engine
cone = pv.Cone(center=(0, 0, 1.75), direction=(0, 0, 1), height=1.75, radius=1, resolution=25, capping=False)
plotter.add_mesh(cone, color='orange')
cap = pv.Capsule(center=(0, 0, 3), direction=(0, 0, 1), radius=0.75, cylinder_length=1)
plotter.add_mesh(cap, color='orange')

# plate
plate = pv.Plane(center=(0, 0, 5), direction=(0, 0, 1), i_size=(2 * q**2)**0.5, j_size=(2 * q**2)**0.5)
plate.rotate_z(45, inplace=True)
plotter.add_mesh(plate, color='gray', opacity=0.5)

plate = pv.Plane(center=(0, 0, 2), direction=(0, 0, 1), i_size=32**0.5, j_size=32**0.5)
plate.rotate_z(45, inplace=True)
plotter.add_mesh(plate, color='gray', opacity=0.5)

# top tank
sph = pv.Sphere(center=(0, 0, 5.25), radius=1.5)
plotter.add_mesh(sph, color='beige')


plotter.show()
"""



recorder = csdl.Recorder(inline=True)
recorder.start()

aluminum = af.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, density=2700)

frame = af.Frame()

for i in range(meshes.shape[0]):
    mesh = meshes[i, :, :]
    num_nodes = mesh.shape[0]
    beam_mesh = csdl.Variable(value=mesh)
    beam_radius = csdl.Variable(value=np.ones(num_nodes - 1) * 0.2)
    beam_thickness = csdl.Variable(value=np.ones(num_nodes - 1) * 0.001)
    beam_cs = af.CSTube(radius=beam_radius, thickness=beam_thickness)
    beam = af.Beam(name='beam_'+str(i), mesh=beam_mesh, material=aluminum, cs=beam_cs)

    beam.add_boundary_condition(node=0)
    beam.add_boundary_condition(node=num_nodes - 1)

    frame.add_beam(beam)


solution = frame.evaluate()

recorder.stop()