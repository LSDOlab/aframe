import numpy as np
import meshio
import csdl_alpha as csdl
import aframe as af
import pyvista as pv


mesh = meshio.read('examples/bunny/bunny.stl')
points = mesh.points

triangles = mesh.cells_dict['triangle']

# Extract edges from triangles (edges are undirected)
edges = set()
for tri in triangles:
    i, j, k = tri
    edges.update({
        tuple(sorted((i, j))),
        tuple(sorted((j, k))),
        tuple(sorted((k, i)))
    })


meshes = []
lines = []

# for i, (start_idx, end_idx) in enumerate(edges):
#     mesh_points = np.array([points[start_idx], points[end_idx]])
#     meshes.append(mesh_points)
#     lines.append(pv.Line(mesh_points[0], mesh_points[1]))


num_nodes = 2

# Subdivide edges into multiple points
for (start_idx, end_idx) in edges:
    p0 = points[start_idx]
    p1 = points[end_idx]
    
    # Linearly spaced nodes between p0 and p1
    mesh_points = np.linspace(p0, p1, num=num_nodes)
    meshes.append(mesh_points)
    
    # Add line only for visualization (2-point line)
    lines.append(pv.Line(p0, p1))


# # plot the lines
# plotter = pv.Plotter()
# for line in lines:
#     plotter.add_mesh(line, color='blue', line_width=3)

# plotter.add_axes()
# plotter.show_grid()
# plotter.show()

# # check for duplicate beam meshes
# tuple_list = [tuple(arr.flatten()) for arr in meshes]
# has_duplicates = len(meshes) != len(set(tuple_list))
# print("Duplicates found!" if has_duplicates else "No duplicates.")





material = af.Material(name='default_material', E=69e9, G=26e9, density=2700)

recorder = csdl.Recorder(inline=True)
recorder.start()

# generic cross-sectional properties
radius = csdl.Variable(value=2E-2)
thickness = csdl.Variable(value=1E-3)
cs = af.CSTube(radius=radius, thickness=thickness)


beams = []
joints = []

for i, mesh1 in enumerate(meshes):

    # check if beam is vertical along z
    vec = mesh1[1] - mesh1[0]
    norm = np.linalg.norm(vec)

    if norm > 0:
        unit_vec = vec / norm
        is_vertical = np.isclose(unit_vec[0], 0, atol=1e-3) and np.isclose(unit_vec[1], 0, atol=1e-3)
    else:
        is_vertical = False

    # print(f"Is vertical: {is_vertical}")


    # Mesh for the beam (2 nodes only)
    beam_mesh = csdl.Variable(value=mesh1)

    beam_name = f'beam_{i}'

    if is_vertical:
        beam = af.Beam(name=beam_name, mesh=beam_mesh, material=material, cs=cs, z=True)
    else:
        beam = af.Beam(name=beam_name, mesh=beam_mesh, material=material, cs=cs)


    # beam.fix(0) # for now

    beams.append(beam)




for i, mesh1 in enumerate(meshes):
    for j, mesh2 in enumerate(meshes):
        if i != j:

            for k, node1 in enumerate(mesh1):
                for l, node2 in enumerate(mesh2):

                    if np.allclose(node1, node2, atol=1E-3):
                        # print(f"Found joint between beam {i} and beam {j} at node {node1}")

                        joints.append(af.Joint(members=[beams[i], beams[j]], nodes=[k, l]))
                        # print('kl: ', k, l)




beams[0].fix(0)
beams[0].fix(1)


beam_loads = np.zeros((num_nodes, 6))
beam_loads[:, 1] = 2
beam_loads = csdl.Variable(value=beam_loads)
beams[16].add_load(beam_loads)



frame = af.Frame(beams=beams, joints=joints)
frame.solve()



recorder.stop()


new_meshes = []
for beam in beams:
    beam_displacement = frame.displacement[beam.name].value
    new_mesh = beam.mesh.value + beam_displacement
    new_meshes.append(new_mesh)



# # Create PyVista lines from deformed meshes
# deformed_lines = []

# for new_mesh in new_meshes:
#     deformed_lines.append(pv.Line(new_mesh.value[0], new_mesh.value[-1]))

# # Plot both original and deformed meshes
# plotter = pv.Plotter()
# # Original in blue
# for line in lines:
#     plotter.add_mesh(line, color='blue', line_width=2)

# # Deformed in red
# for line in deformed_lines:
#     plotter.add_mesh(line, color='red', line_width=2)

# plotter.add_axes()
# plotter.show_grid()
# plotter.show()


# --- PLOT DEFORMED MESH ---
deformed_lines = []
midpoints = []
for mesh in new_meshes:
    for i in range(len(mesh) - 1):
        line = pv.Line(mesh[i], mesh[i+1])
        deformed_lines.append(line)
        points = line.points
        midpoint = (points[0] + points[1]) / 2
        midpoints.append(midpoint.flatten())

plotter = pv.Plotter()
for line in lines:
    plotter.add_mesh(line, color='blue', line_width=3, opacity=0.4)
for line in deformed_lines:
    plotter.add_mesh(line, color='red', line_width=3)



# Plot nodes as small spheres or points
node_radius = 0.5

# Original nodes
for mesh in meshes:
    for point in mesh:
        sphere = pv.Sphere(radius=node_radius, center=point)
        plotter.add_mesh(sphere, color='blue', opacity=0.2)

# Deformed nodes
for mesh in new_meshes:
    for point in mesh:
        sphere = pv.Sphere(radius=node_radius, center=point)
        plotter.add_mesh(sphere, color='red')



labels = [str(i) for i in range(len(midpoints))]
plotter.add_point_labels(midpoints, labels, font_size=14, text_color="black")



plotter.add_axes()
plotter.show_grid()
plotter.show()