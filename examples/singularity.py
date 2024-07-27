import csdl_alpha as csdl
import numpy as np
import aframe as af


"""
Test the singularity when beams are vertical along the z axis
"""


# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam 1 mesh
num_nodes_1 = 7
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 2] = np.linspace(0, 10, num_nodes_1)
beam_1_mesh = csdl.Variable(value=beam_1_mesh)

# create beam 1 loads
beam_1_loads = np.zeros((num_nodes_1, 6))
beam_1_loads[:, 2] = 0
beam_1_loads = csdl.Variable(value=beam_1_loads)

# create a material
aluminum = af.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, density=2700)

# create cs properties for beam 1
beam_1_radius = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.5)
beam_1_thickness = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.001)
beam_1_cs = af.CSTube(radius=beam_1_radius, thickness=beam_1_thickness)

# create beam 1 with boundary conditions and loads
beam_1 = af.Beam(name='beam_1', mesh=beam_1_mesh, material=aluminum, cs=beam_1_cs, z=True)
beam_1.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
beam_1.add_load(beam_1_loads)

# define the accelerations
# a_x, a_y, a_z, alpha_x, alpha_y, alpha_z
acc = csdl.Variable(value=np.array([0, 0, -9.81, 0, 0, 0]))

# instantiate the frame model and add all beams and joints
frame = af.Frame(acc=acc)
frame.add_beam(beam_1)

# evaluating the frame model returns a solution dataclass
solution = frame.evaluate()

beam_1_displacement = solution.get_displacement(beam_1)
beam_1_def_mesh = solution.get_mesh(beam_1)
beam_1_stress = solution.get_stress(beam_1)

recorder.stop()

print(beam_1_displacement.value)

import matplotlib.pyplot as plt
plt.grid()
plt.plot(beam_1_def_mesh.value[:, 1], beam_1_def_mesh.value[:, 2], color='black', linewidth=2)
plt.scatter(beam_1_def_mesh.value[:, 1], beam_1_def_mesh.value[:, 2], zorder=10, edgecolor='black', s=50, color='green')
plt.show()