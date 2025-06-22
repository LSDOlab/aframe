import numpy as np
import csdl_alpha as csdl
import aframe as af
import pyvista as pv
import pickle




with open('examples/ucsd_lunar_lander/lunar_lander_meshes.pkl', 'rb') as file:
    meshes, radius = pickle.load(file)

n = meshes.shape[1]

aluminum = af.Material(name='aluminum', E=69E9, G=26E9, density=2700)

# dummy_load = np.zeros((n, 6))
# dummy_load[:, 2] = -10000

recorder = csdl.Recorder(inline=True)
recorder.start()


beams = []
for i in range(28):
    thickness = csdl.Variable(value=np.ones(n - 1) * 0.001)
    beam_radius = csdl.Variable(value=np.ones(n - 1) * radius[i])
    cs = af.CSTube(radius=beam_radius, thickness=thickness)
    beam = af.Beam(name='beam_'+str(i), mesh=meshes[i, :, :], material=aluminum, cs=cs)

    if i in [0, 4, 6, 10]: # fix the feet
        beam.fix(0)
        # beam.fix(n - 1)

    inertial_mass = csdl.Variable(value=np.ones(n) * 10)
    if i in [20, 21, 22, 23]: # add mass to bottom frame
        beam.add_inertial_mass(inertial_mass)

    # # add loads to four base beams:
    # # if i in [20, 21, 22, 23]:
    # #     beam.add_load(dummy_load)

    inertial_mass = csdl.Variable(value=np.ones(n) * 1)
    if i in [24, 25, 26, 27]:  # add mass to top frame
        beam.add_inertial_mass(inertial_mass)

    beams.append(beam)




ne = n - 1
joints = []
# foot joints
joints.append(af.Joint(members=[beams[0], beams[1], beams[2]], nodes=[0, 0, 0]))
joints.append(af.Joint(members=[beams[3], beams[4], beams[5]], nodes=[0, 0, 0]))
joints.append(af.Joint(members=[beams[6], beams[7], beams[8]], nodes=[0, 0, 0]))
joints.append(af.Joint(members=[beams[9], beams[10], beams[11]], nodes=[0, 0, 0]))
# middle outside joints
joints.append(af.Joint(members=[beams[1], beams[3], beams[13], beams[14], beams[20], beams[21]], nodes=[ne, ne, 0, 0, ne, 0]))
joints.append(af.Joint(members=[beams[4], beams[6], beams[15], beams[16], beams[21], beams[22]], nodes=[ne, ne, 0, 0, ne, 0]))
joints.append(af.Joint(members=[beams[7], beams[9], beams[17], beams[18], beams[22], beams[23]], nodes=[ne, ne, 0, 0, ne, 0]))
joints.append(af.Joint(members=[beams[10], beams[0], beams[12], beams[19], beams[20], beams[23]], nodes=[ne, ne, 0, 0, 0, ne]))
# leg strut attach joints
joints.append(af.Joint(members=[beams[2], beams[12], beams[13], beams[24], beams[25]], nodes=[ne, ne, ne, ne, 0]))
joints.append(af.Joint(members=[beams[5], beams[14], beams[15], beams[25], beams[26]], nodes=[ne, ne, ne, ne, 0]))
joints.append(af.Joint(members=[beams[8], beams[16], beams[17], beams[26], beams[27]], nodes=[ne, ne, ne, ne, 0]))
joints.append(af.Joint(members=[beams[11], beams[18], beams[19], beams[27], beams[24]], nodes=[ne, ne, ne, ne, 0]))



acc = csdl.Variable(value=np.array([0, 0, -9.81 * 20, 0, 0, 0]))

frame = af.Frame(beams=beams, joints=joints, acc=acc)




frame.solve()
disp = frame.displacement
disp = frame.displacement
stress = frame.compute_stress()



recorder.stop()


# convert disp and stress dictionaries to copies where every value calls .value
disp = {k: v.value for k, v in disp.items()}
stress = {k: v.value for k, v in stress.items()}

# exit()

import matplotlib as mpl

def _colorize(cell_data, cmap, color, n):

    if cell_data is not None:
        top = cell_data - cell_data.min()
        bot = cell_data.max() - cell_data.min()
        n_cell_data = top / bot

        colormap = mpl.colormaps[cmap]

        colors = colormap(n_cell_data)
    else:
        colors = [color] * n

    return colors


def plot_points(plotter,
                mesh,
                color='red', 
                point_size=50,
                render_points_as_spheres=True):
    
    n = mesh.shape[0]
    nodes = mesh
    edges = np.zeros((n - 1, 2)).astype(int)
    for i in range(n - 1): edges[i, :] = [i, i + 1]
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    mesh = pv.PolyData(nodes, edges_w_padding)

    plotter.add_points(mesh.points, color=color,
                        point_size=point_size,
                        render_points_as_spheres=render_points_as_spheres,
                        )

def plot_cyl(plotter, 
             mesh,
             radius,
             cell_data=None, 
             cmap='viridis', 
             color='lightblue',
             ):
    
    n = mesh.shape[0]

    colors = _colorize(cell_data, cmap, color, n)

    for i in range(n - 1):
        start = mesh[i, :]
        end = mesh[i + 1, :]

        cyl = pv.Cylinder(center=(start + end) / 2, 
                          direction=end - start, 
                          radius=radius[i], 
                          height=np.linalg.norm(end - start))

        plotter.add_mesh(cyl, color=colors[i])




plotter = pv.Plotter()

for i, beam in enumerate(frame.beams):
    mesh0 = beam.mesh
    d = disp[beam.name]
    mesh1 = mesh0 + 25 * d

    radius = beam.cs.radius.value

    s = stress[beam.name]

    # af.plot_mesh(plotter, mesh0, color='lightblue', line_width=10)
    # plot_mesh(plotter, mesh1, cell_data=stress, cmap='viridis', line_width=20)
    plot_points(plotter, mesh1, color='blue', point_size=10)

    # radius = np.ones((beam.num_elements)) * 0.1
    plot_cyl(plotter, mesh1, cell_data=s, radius=radius, cmap='plasma')

    if i in [0, 4, 6, 10]:
        cyl = pv.Cylinder(center=mesh1[0, :], direction=[0, 0, 1], radius=0.6, height=0.2)
        plotter.add_mesh(cyl, color='thistle')



zo = np.array([0, 0, -0.2])
scale = 70

ft1 = pv.read('examples/ucsd_lunar_lander/foot.stl')
ft1.scale(scale, inplace=True)
ft1.translate(meshes[0,0,:] + zo, inplace=True)
plotter.add_mesh(ft1, color='red')

ft2 = pv.read('examples/ucsd_lunar_lander/foot.stl')
ft2.scale(scale, inplace=True)
ft2.translate(meshes[4,0,:] + zo, inplace=True)
plotter.add_mesh(ft2, color='red')

ft3 = pv.read('examples/ucsd_lunar_lander/foot.stl')
ft3.scale(scale, inplace=True)
ft3.translate(meshes[6,0,:] + zo, inplace=True)
plotter.add_mesh(ft3, color='red')

ft4 = pv.read('examples/ucsd_lunar_lander/foot.stl')
ft4.scale(scale, inplace=True)
ft4.translate(meshes[10,0,:] + zo, inplace=True)
plotter.add_mesh(ft4, color='red')

# toroidal tank
torus = pv.ParametricTorus(ringradius=2.375, crosssectionradius=1.4, center=(0, 0, 3.5))
plotter.add_mesh(torus, color='skyblue', opacity=0.5)

# engine
eng = pv.read('examples/ucsd_lunar_lander/j2engine.stl')
eng.scale(0.01, inplace=True)
eng.translate([0, 0, 1], inplace=True)
plotter.add_mesh(eng, color='orange')

# top tank
sph = pv.Sphere(center=(0, 0, 5.25), radius=1.5)
plotter.add_mesh(sph, color='beige', opacity=0.5)










# Load image as texture
texture = pv.read_texture("examples/ucsd_lunar_lander/lunar_surface.jpg")

# Create a plane for the texture
width, height = 1.5e1, 1.5e1  # Scale in x and y
plane = pv.Plane(center=(0, 0, -0.2), direction=(0, 0, 1),
                 i_size=width, j_size=height, i_resolution=1, j_resolution=1)

# Add to plotter
plotter.add_mesh(plane, texture=texture)






plotter.show()