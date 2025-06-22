import numpy as np
import pickle

import pyvista as pv
import numpy as np
import aframe as af


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
z = 5
leg_points = np.array([[y, y, z], [-y, y, z], [-y, -y, z], [y, -y, z]])

points = np.vstack((foot_points, 
                  base_points, 
                  leg_points,
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
                  [6, 8],
                  [7, 8],
                  [7, 9],
                  [4, 9],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 4],
                  [9, 10],
                  [10, 11],
                  [11, 8],
                  [8, 9],
                  ])


sr = 0.05
lr = 0.15
radius = np.array([sr, sr, lr, sr, sr, lr, sr, sr, lr, sr, sr, lr, sr, sr, sr, sr, sr, sr, sr, sr, lr, lr, lr, lr, sr, sr, sr, sr])



nodes_per_edge = 6
meshes = af.mesh_from_points_and_edges(points, edges, nodes_per_edge)


with open('lunar_lander_meshes.pkl', 'wb') as file:
    pickle.dump((meshes, radius), file)


# exit()
plotter = pv.Plotter()

for i in range(meshes.shape[0]):
    mesh = meshes[i, :, :]

    af.plot_cyl(plotter, mesh, cell_data=None, radius=np.ones((nodes_per_edge - 1)) * radius[i], color='lightblue')

    if i in [0, 4, 6, 10]:
        cyl = pv.Cylinder(center=mesh[0, :], direction=[0, 0, 1], radius=0.6, height=0.2)
        plotter.add_mesh(cyl, color='thistle')



zo = np.array([0, 0, -0.2])
scale = 70

ft1 = pv.read('examples/stl/foot.stl')
ft1.scale(scale, inplace=True)
ft1.translate(meshes[0,0,:] + zo, inplace=True)
plotter.add_mesh(ft1, color='red')

ft2 = pv.read('examples/stl/foot.stl')
ft2.scale(scale, inplace=True)
ft2.translate(meshes[4,0,:] + zo, inplace=True)
plotter.add_mesh(ft2, color='red')

ft3 = pv.read('examples/stl/foot.stl')
ft3.scale(scale, inplace=True)
ft3.translate(meshes[6,0,:] + zo, inplace=True)
plotter.add_mesh(ft3, color='red')

ft4 = pv.read('examples/stl/foot.stl')
ft4.scale(scale, inplace=True)
ft4.translate(meshes[10,0,:] + zo, inplace=True)
plotter.add_mesh(ft4, color='red')


# toroidal tank
torus = pv.ParametricTorus(ringradius=2.375, crosssectionradius=1.4, center=(0, 0, 3.5))
plotter.add_mesh(torus, color='skyblue', opacity=0.5)

# helium
# num_helium_tanks = 6
# r = 2.25
# for i in range(num_helium_tanks):
#     angle = (i * (2 * np.pi) / num_helium_tanks) + np.pi / 4
#     x, y = r * np.cos(angle), r * np.sin(angle)
#     sph = pv.Sphere(center=(x, y, 5.75), radius=0.7)
#     plotter.add_mesh(sph, color='green')


# engine
eng = pv.read('examples/stl/j2engine.stl')
eng.scale(0.01, inplace=True)
eng.translate([0, 0, 1], inplace=True)
plotter.add_mesh(eng, color='orange')

# plate
plate = pv.Plane(center=(0, 0, 5), direction=(0, 0, 1), i_size=6, j_size=6)
# plate.rotate_z(45, inplace=True)
plotter.add_mesh(plate, color='gray', opacity=0.5)

plate = pv.Plane(center=(0, 0, 2), direction=(0, 0, 1), i_size=32**0.5, j_size=32**0.5)
plate.rotate_z(45, inplace=True)
plotter.add_mesh(plate, color='gray', opacity=0.5)

# top tank
sph = pv.Sphere(center=(0, 0, 5.25), radius=1.5)
plotter.add_mesh(sph, color='beige', opacity=0.5)


plotter.show()
