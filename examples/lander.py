import csdl_alpha as csdl
import numpy as np
import aframe as af
import pyvista as pv

# lower ring points
num_points = 6
ring_radius = 1
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
x = ring_radius * np.cos(angles)
y = ring_radius * np.sin(angles)
z = np.zeros(num_points)
lower_ring_points = np.vstack((x, y, z)).T


# foot points
foot_radius = 3
angles = np.linspace(np.pi / 6, (2 * np.pi) + (np.pi / 6), num_points, endpoint=False)
x = foot_radius * np.cos(angles)
y = foot_radius * np.sin(angles)
z = np.ones(num_points) * -1.5
foot_points = np.vstack((x, y, z)).T

# upper ring points





points = np.vstack((lower_ring_points, foot_points))

edges = np.array([[0, 1], 
                  [1, 2], 
                  [2, 3], 
                  [3, 4], 
                  [4, 5], 
                  [5, 0], 
                  [0, 6], 
                  [1, 6], 
                  [1, 7], 
                  [2, 7], 
                  [2, 8], 
                  [3, 8],
                  [3, 9],
                  [4, 9],
                  [4, 10],
                  [5, 10],
                  [5, 11],
                  [0, 11]])

nodes_per_edge = 5
meshes = af.mesh_from_points_and_edges(points, edges, nodes_per_edge)

plotter = pv.Plotter()
for i in range(meshes.shape[0]):
    mesh = meshes[i, :, :]

    af.plot_mesh(plotter, mesh, color='lightblue', line_width=10)

plotter.show()
exit()

# upper ring points


recorder = csdl.Recorder(inline=True)
recorder.start()


loads = np.zeros((nodes_per_edge, 6))
loads[:, 2] = 200000
loads = csdl.Variable(value=loads)



aluminum = af.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, density=2700)

frame = af.Frame()

# add the lower ring meshes
for i in range(lower_ring_meshes.shape[0]):
    num_nodes = lower_ring_meshes.shape[1]
    beam_mesh = csdl.Variable(value=lower_ring_meshes[i, :, :])
    beam_radius = csdl.Variable(value=np.ones(num_nodes - 1) * 0.2)
    beam_thickness = csdl.Variable(value=np.ones(num_nodes - 1) * 0.001)
    beam_cs = af.CSTube(radius=beam_radius, thickness=beam_thickness)

    beam = af.Beam(name='beam_'+str(i), mesh=beam_mesh, material=aluminum, cs=beam_cs)
    beam.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
    beam.add_boundary_condition(node=num_nodes - 1, dof=[1, 1, 1, 1, 1, 1])
    beam.add_load(loads)

    frame.add_beam(beam)


solution = frame.evaluate()

recorder.stop()


plotter = pv.Plotter()

for beam in frame.beams:
    mesh0 = beam.mesh.value
    mesh1 = solution.get_mesh(beam).value

    stress = solution.get_stress(beam).value

    af.plot_mesh(plotter, mesh0, color='lightblue', line_width=10)
    # plot_mesh(plotter, mesh1, cell_data=stress, cmap='viridis', line_width=20)
    # plot_points(plotter, mesh1, color='blue', point_size=30)

    radius = np.ones((beam.num_elements)) * 0.1
    af.plot_cyl(plotter, mesh1, cell_data=stress, radius=radius)

    # height = np.ones((beam.num_elements)) * 0.2
    # width = np.ones((beam.num_elements)) * 0.2
    # af.plot_box(plotter, mesh1, height, width, cell_data=None)

plotter.show()