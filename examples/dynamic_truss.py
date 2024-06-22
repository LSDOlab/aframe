import csdl_alpha as csdl
import numpy as np
import aframe as af


# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create beam 1 mesh
num_nodes_1 = 11
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 1] = np.linspace(0, 1, num_nodes_1)
beam_1_mesh = csdl.Variable(value=beam_1_mesh)

# create beam 2 mesh
num_nodes_2 = 11
beam_2_mesh = np.zeros((num_nodes_2, 3))
beam_2_mesh[:, 0] = 1
beam_2_mesh[:, 1] = np.linspace(0, 1, num_nodes_2)
beam_2_mesh = csdl.Variable(value=beam_2_mesh)

# create beam 3 mesh
num_nodes_3 = 11
beam_3_mesh = np.zeros((num_nodes_3, 3))
beam_3_mesh[:, 0] = np.linspace(0, 1, num_nodes_3)
beam_3_mesh[:, 1] = 1
beam_3_mesh = csdl.Variable(value=beam_3_mesh)

# create beam 1 loads
beam_1_loads = np.zeros((num_nodes_1, 6))
beam_1_loads[-1, 0] = 100000
beam_1_loads = csdl.Variable(value=beam_1_loads)

# create a material
aluminum = af.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, density=2700)

# create cs properties for beam 1
beam_1_radius = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.1)
beam_1_thickness = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.0001)
beam_1_cs = af.CSTube(radius=beam_1_radius, thickness=beam_1_thickness)

# create cs properties for beam 2
beam_2_radius = csdl.Variable(value=np.ones(num_nodes_2 - 1) * 0.1)
beam_2_thickness = csdl.Variable(value=np.ones(num_nodes_2 - 1) * 0.0001)
beam_2_cs = af.CSTube(radius=beam_2_radius, thickness=beam_2_thickness)

# create cs properties for beam 3
beam_3_radius = csdl.Variable(value=np.ones(num_nodes_3 - 1) * 0.06)
beam_3_thickness = csdl.Variable(value=np.ones(num_nodes_3 - 1) * 0.0001)
beam_3_cs = af.CSTube(radius=beam_3_radius, thickness=beam_3_thickness)

# create beam 1 with boundary conditions and loads
beam_1 = af.Beam(name='beam_1', mesh=beam_1_mesh, material=aluminum, cs=beam_1_cs)
beam_1.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
beam_1.add_load(beam_1_loads)

# create beam 2 with boundary conditions and loads
beam_2 = af.Beam(name='beam_2', mesh=beam_2_mesh, material=aluminum, cs=beam_2_cs)
beam_2.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
# beam_2.add_load(beam_2_loads)

# create beam 3 with boundary conditions and loads
beam_3 = af.Beam(name='beam_3', mesh=beam_3_mesh, material=aluminum, cs=beam_3_cs)
# beam_3.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
# beam_2.add_load(beam_2_loads)

# instantiate the frame model and add all beams and joints
frame = af.Frame()
frame.add_beam(beam_1)
frame.add_beam(beam_2)
frame.add_beam(beam_3)
frame.add_joint(joint_beams=[beam_1, beam_3], joint_nodes=[10, 0])
frame.add_joint(joint_beams=[beam_2, beam_3], joint_nodes=[10, 10])

# evaluating the frame model returns a solution dataclass
solution = frame.evaluate()

# finish up
recorder.stop()


start = 0
stop = 0.01
nt = 250
sim = af.Simulation(solution, start, stop, nt)
t, u = sim.solve()
beam_1_def_mesh = sim.parse_u(u, beam_1)
beam_2_def_mesh = sim.parse_u(u, beam_2)
beam_3_def_mesh = sim.parse_u(u, beam_3)

sim.create_frames([beam_1_def_mesh, beam_2_def_mesh, beam_3_def_mesh], xlim=[-0.5, 1.5], ylim=[-0.25, 1.3], figsize=(5, 5), ax1=0, ax2=1)
sim.gif(filename='dynamic_struct.gif', fps=25)

