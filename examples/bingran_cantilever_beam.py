import csdl_alpha as csdl
import numpy as np
import aframe as af
import matplotlib.pyplot as plt

# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam 1 mesh
num_nodes_1 = 21
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 1] = np.linspace(0, 10, num_nodes_1)
beam_1_mesh = csdl.Variable(value=beam_1_mesh)

# create beam 1 loads
beam_1_loads = np.zeros((num_nodes_1, 6))
beam_1_loads[:, 2] = 20000
beam_1_loads = csdl.Variable(value=beam_1_loads)

# create a material
aluminum = af.Material(name='aluminum', E=69E9, G=26E9, density=2700)

# create cs properties for beam 1
beam_1_radius = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.5)
beam_1_thickness = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.001)
beam_1_cs = af.CSTube(radius=beam_1_radius, thickness=beam_1_thickness)

# create beam 1 with boundary conditions and loads
beam_1 = af.Beam(name='beam_1', mesh=beam_1_mesh, material=aluminum, cs=beam_1_cs)
beam_1.fix(node=0)
beam_1.add_load(beam_1_loads)

# instantiate the frame model and add all beams and joints
frame = af.Frame(beams=[beam_1])

# solve the linear system
frame.solve()

# get the displacement
beam_1_displacement = frame.displacement[beam_1.name]

# make the displaced mesh
beam_1_def_mesh = beam_1_mesh + beam_1_displacement

# get the cg
cg = frame.cg

# stress
stress = frame.compute_stress()
beam_1_stess = stress[beam_1.name]

# finish up
recorder.stop()
# recorder.visualize_graph(trim_loops=True)



# sim = csdl.experimental.PySimulator(recorder)
# sim = csdl.experimental.JaxSimulator(recorder=recorder)
# sim.run()



print('cg: ', cg.value)
print('displacement: ', beam_1_displacement.value)


plt.grid()
plt.plot(beam_1_def_mesh.value[:, 1], beam_1_def_mesh.value[:, 2], color='black', linewidth=2)
plt.scatter(beam_1_def_mesh.value[:, 1], beam_1_def_mesh.value[:, 2], zorder=10, edgecolor='black', s=50, color='green')
plt.xlabel('Dr. Wang')
plt.ylabel('Dr. Wang')
plt.title('Dr. Bingran Wang, PhD, MD, MBA')
plt.show()