import csdl_alpha as csdl
import numpy as np
import aframe as af


recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam 1 mesh
num_nodes_1 = 21
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 1] = np.linspace(-20, 20, num_nodes_1)
beam_1_mesh = csdl.Variable(value=beam_1_mesh)

# create a 1D beam 2 mesh
num_nodes_2 = 21
beam_2_mesh = np.zeros((num_nodes_2, 3))
beam_2_mesh[:, 0] = np.linspace(-20, 20, num_nodes_2)
beam_2_mesh = csdl.Variable(value=beam_2_mesh)

# create beam 1 loads
beam_1_loads = np.zeros((num_nodes_1, 6))
beam_1_loads[:, 2] = 20000
beam_1_loads = csdl.Variable(value=beam_1_loads)

# create beam 2 loads
beam_2_loads = np.zeros((num_nodes_2, 6))
beam_2_loads[:, 2] = 20000
beam_2_loads = csdl.Variable(value=beam_2_loads)

aluminum = af.Material(name='aluminum', E=69E9, G=26E9, density=2700)

# create cs properties for beam 1
beam_1_radius = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.5)
beam_1_thickness = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.001)
beam_1_cs = af.CSTube(radius=beam_1_radius, thickness=beam_1_thickness)

# create cs properties for beam 2
beam_2_radius = csdl.Variable(value=np.ones(num_nodes_2 - 1) * 0.5)
beam_2_thickness = csdl.Variable(value=np.ones(num_nodes_2 - 1) * 0.001)
beam_2_cs = af.CSTube(radius=beam_2_radius, thickness=beam_2_thickness)

# create beam 1 with boundary conditions and loads
beam_1 = af.Beam(name='beam_1', mesh=beam_1_mesh, material=aluminum, cs=beam_1_cs)
beam_1.fix(10)
beam_1.add_load(beam_1_loads)

# create beam 2 with boundary conditions and loads
beam_2 = af.Beam(name='beam_2', mesh=beam_2_mesh, material=aluminum, cs=beam_2_cs)
beam_2.add_load(beam_2_loads)

# instantiate the frame model and add all beams and joints
frame = af.Frame()
frame.add_beam(beam_1)
frame.add_beam(beam_2)
frame.add_joint(members=[beam_1, beam_2], nodes=[10, 10])

frame.solve()

beam_1_displacement = frame.displacement[beam_1.name]
beam_2_displacement = frame.displacement[beam_2.name]
cg = frame.cg

recorder.stop()
print(beam_1_displacement.value)
print('cg: ', cg.value)