import csdl_alpha as csdl
import numpy as np
import aframe as af

np.set_printoptions(edgeitems=30, linewidth=100,)

# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam 1 mesh
num_nodes_1 = 21
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 1] = np.linspace(-20, 20, num_nodes_1)
beam_1_mesh = csdl.Variable(value=beam_1_mesh)

# create beam 1 loads
beam_1_loads = np.zeros((num_nodes_1, 6))
beam_1_loads[:, 2] = 20000
beam_1_loads = csdl.Variable(value=beam_1_loads)

# create a material
aluminum = af.Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)

# create cs properties for beam 1
height = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.5)
width = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.5)
ttop = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.01)
tbot = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.01)
tweb = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.01)
beam_1_cs = af.CSBox(height=height, width=width, ttop=ttop, tbot=tbot, tweb=tweb)

# create beam 1 with boundary conditions and loads
beam_1 = af.Beam(name='beam_1', mesh=beam_1_mesh, material=aluminum, cs=beam_1_cs)
beam_1.add_boundary_condition(node=10, dof=[1, 1, 1, 1, 1, 1])
beam_1.add_load(beam_1_loads)

# instantiate the frame model and add all beams and joints
frame = af.Frame()
frame.add_beam(beam_1)

# evaluating the frame model returns a solution dataclass
solution = frame.evaluate()

# displacement
beam_1_displacement = solution.get_displacement(beam_1)

# stress
beam_1_stress = solution.get_stress(beam_1)

cg = solution.cg
dcg = solution.dcg

# finish up
recorder.stop()
# recorder.visualize_graph(trim_loops=True)
print(beam_1_displacement.value)
print('cg: ', cg.value)
print('deformed cg: ', dcg.value)