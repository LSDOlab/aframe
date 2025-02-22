import csdl_alpha as csdl
import numpy as np
import aframe as af


# because Marius says aframe fails for 1 element beams
# this script runs, so he is clearly wrong

recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam 1 mesh
num_nodes_1 = 2
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 1] = np.linspace(0, 1, num_nodes_1)
beam_1_mesh = csdl.Variable(value=beam_1_mesh)

# create a 1D beam 2 mesh
num_nodes_2 = 2
beam_2_mesh = np.zeros((num_nodes_2, 3))
beam_2_mesh[:, 1] = np.linspace(1, 2, num_nodes_2)
beam_2_mesh = csdl.Variable(value=beam_2_mesh)

# create a 1D beam 3 mesh
num_nodes_3 = 2
beam_3_mesh = np.zeros((num_nodes_3, 3))
beam_3_mesh[:, 1] = np.linspace(2, 3, num_nodes_3)
beam_3_mesh = csdl.Variable(value=beam_3_mesh)

# create a 1D beam 4 mesh
num_nodes_4 = 2
beam_4_mesh = np.zeros((num_nodes_4, 3))
beam_4_mesh[:, 1] = np.linspace(3, 4, num_nodes_3)
beam_4_mesh = csdl.Variable(value=beam_4_mesh)

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
beam_1.fix(0)
beam_1.add_load(beam_1_loads)

# create beam 2 with boundary conditions and loads
beam_2 = af.Beam(name='beam_2', mesh=beam_2_mesh, material=aluminum, cs=beam_2_cs)
beam_2.add_load(beam_2_loads)

beam_3 = af.Beam(name='beam_3', mesh=beam_3_mesh, material=aluminum, cs=beam_2_cs)

beam_4 = af.Beam(name='beam_4', mesh=beam_4_mesh, material=aluminum, cs=beam_2_cs)


joint_1 = af.Joint(members=[beam_1, beam_2], nodes=[1, 0])
joint_2 = af.Joint(members=[beam_2, beam_3], nodes=[1, 0])
joint_3 = af.Joint(members=[beam_3, beam_4], nodes=[1, 0])

# instantiate the frame model and add all beams and joints
frame = af.Frame(beams=[beam_1, beam_2, beam_3, beam_4], joints=[joint_1, joint_2, joint_3])


# solve the linear system
frame.solve()

beam_1_displacement = frame.displacement[beam_1.name]
beam_2_displacement = frame.displacement[beam_2.name]
beam_3_displacement = frame.displacement[beam_3.name]
beam_4_displacement = frame.displacement[beam_4.name]
cg = frame.cg

recorder.stop()


print(beam_1_displacement.value)
print(beam_2_displacement.value)
print(beam_3_displacement.value)
print(beam_4_displacement.value)
print('cg: ', cg.value)