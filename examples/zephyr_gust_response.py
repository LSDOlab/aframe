import csdl_alpha as csdl
import numpy as np
import aframe as af


# airplane params
wingspan = 25
half_span = wingspan / 2
quarter_span = wingspan / 4

# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create the center wing mesh
num_center_wing = 11
center_wing_mesh = np.zeros((num_center_wing, 3))
center_wing_mesh[:, 0] = 3.25
center_wing_mesh[:, 1] = np.linspace(-quarter_span, quarter_span, num_center_wing)
center_wing_mesh = csdl.Variable(value=center_wing_mesh)
# boundary_condition_1 = BoundaryCondition(beam=center_wing, node=5)

num_outer_wing = 6
right_outer_wing_mesh = np.zeros((num_outer_wing, 3))
right_outer_wing_mesh[:, 0] = 3.25
right_outer_wing_mesh[:, 1] = np.linspace(quarter_span, half_span, num_outer_wing)
right_outer_wing_mesh[:, 2] = np.linspace(0, quarter_span * np.tan(np.deg2rad(15)), num_outer_wing)
right_outer_wing_mesh = csdl.Variable(value=right_outer_wing_mesh)
# joint_1 = Joint(beams=[center_wing, right_outer_wing], nodes=[num_center_wing - 1, 0])

left_outer_wing_mesh = np.zeros((num_outer_wing, 3))
left_outer_wing_mesh[:, 0] = 3.25
left_outer_wing_mesh[:, 1] = np.linspace(-quarter_span, -half_span, num_outer_wing)
left_outer_wing_mesh[:, 2] = np.linspace(0, quarter_span * np.tan(np.deg2rad(15)), num_outer_wing)
left_outer_wing_mesh = csdl.Variable(value=left_outer_wing_mesh)
# joint_2 = Joint(beams=[center_wing, left_outer_wing], nodes=[0, 0])

num_fwd_fuse = 4
fwd_fuse_mesh = np.zeros((num_fwd_fuse, 3))
fwd_fuse_mesh[:, 0] = np.linspace(0, 3.25, num_fwd_fuse)
fwd_fuse_mesh = csdl.Variable(value=fwd_fuse_mesh)
# joint_3 = Joint(beams=[center_wing, fwd_fuse], nodes=[5, num_fwd_fuse - 1])

num_aft_fuse = 6
aft_fuse_mesh = np.zeros((num_aft_fuse, 3))
aft_fuse_mesh[:, 0] = np.linspace(3.25, 8.5, num_aft_fuse)
aft_fuse_mesh = csdl.Variable(value=aft_fuse_mesh)
# joint_4 = Joint(beams=[center_wing, aft_fuse], nodes=[5, 0])

num_vtail = 4
vtail_mesh = np.zeros((num_vtail, 3))
vtail_mesh[:, 0] = np.linspace(8.5, 8.625, num_vtail)
vtail_mesh[:, 2] = np.linspace(0, 1.5, num_vtail)
vtail_mesh = csdl.Variable(value=vtail_mesh)
# joint_5 = Joint(beams=[aft_fuse, vtail], nodes=[num_aft_fuse - 1, 0])

num_htail = 5
htail_mesh = np.zeros((num_htail, 3))
htail_mesh[:, 0] = 8.625
htail_mesh[:, 1] = np.linspace(-2, 2, num_htail)
htail_mesh[:, 2] = 1.5
htail_mesh = csdl.Variable(value=htail_mesh)
# joint_6 = Joint(beams=[vtail, htail], nodes=[num_vtail - 1, 2])

# create the center wing loads
center_wing_loads = np.zeros((num_center_wing, 6))
center_wing_loads[:, 2] = 10000
center_wing_loads = csdl.Variable(value=center_wing_loads)

# create the right outer wing loads
right_outer_wing_loads = np.zeros((num_outer_wing, 6))
right_outer_wing_loads[:, 2] = 10000
right_outer_wing_loads = csdl.Variable(value=right_outer_wing_loads)

# create the left outer wing loads
left_outer_wing_loads = np.zeros((num_outer_wing, 6))
left_outer_wing_loads[:, 2] = 10000
left_outer_wing_loads = csdl.Variable(value=left_outer_wing_loads)

# create the htail loads
htail_loads = np.zeros((num_htail, 6))
htail_loads[:, 2] = -3000
htail_loads = csdl.Variable(value=htail_loads)

# create the fwd fuse loads
fwd_fuse_loads = np.zeros((num_fwd_fuse, 6))
fwd_fuse_loads[:, 2] = -10000
fwd_fuse_loads = csdl.Variable(value=fwd_fuse_loads)



# create a material
aluminum = af.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, density=2700)



# create cs properties for the center wing
center_wing_radius = csdl.Variable(value=np.ones(num_center_wing - 1) * 0.5)
center_wing_thickness = csdl.Variable(value=np.ones(num_center_wing - 1) * 0.001)
center_wing_cs = af.CSTube(radius=center_wing_radius, thickness=center_wing_thickness)

# create cs properties for the outer wings
outer_wing_radius = csdl.Variable(value=np.ones(num_outer_wing - 1) * 0.5)
outer_wing_thickness = csdl.Variable(value=np.ones(num_outer_wing - 1) * 0.001)
outer_wing_cs = af.CSTube(radius=outer_wing_radius, thickness=outer_wing_thickness)

# create cs properties for the fwd fuse
fwd_fuse_radius = csdl.Variable(value=np.ones(num_fwd_fuse - 1) * 0.5)
fwd_fuse_thickness = csdl.Variable(value=np.ones(num_fwd_fuse - 1) * 0.001)
fwd_fuse_cs = af.CSTube(radius=fwd_fuse_radius, thickness=fwd_fuse_thickness)

# create cs properties for the aft fuse
aft_fuse_radius = csdl.Variable(value=np.ones(num_aft_fuse - 1) * 0.5)
aft_fuse_thickness = csdl.Variable(value=np.ones(num_aft_fuse - 1) * 0.001)
aft_fuse_cs = af.CSTube(radius=aft_fuse_radius, thickness=aft_fuse_thickness)

# create cs properties for the htail
htail_radius = csdl.Variable(value=np.ones(num_htail - 1) * 0.5)
htail_thickness = csdl.Variable(value=np.ones(num_htail - 1) * 0.001)
htail_cs = af.CSTube(radius=htail_radius, thickness=htail_thickness)

# create cs properties for the vtail
vtail_radius = csdl.Variable(value=np.ones(num_vtail - 1) * 0.5)
vtail_thickness = csdl.Variable(value=np.ones(num_vtail - 1) * 0.001)
vtail_cs = af.CSTube(radius=vtail_radius, thickness=vtail_thickness)



# create the center wing
center_wing = af.Beam(name='center_wing', mesh=center_wing_mesh, material=aluminum, cs=center_wing_cs)
center_wing.add_boundary_condition(node=5, dof=[1, 1, 1, 1, 1, 1])
center_wing.add_load(center_wing_loads)

# create the left outer wing
left_outer_wing = af.Beam(name='left_outer_wing', mesh=left_outer_wing_mesh, material=aluminum, cs=outer_wing_cs)
left_outer_wing.add_load(left_outer_wing_loads)

# create the right outer wing
right_outer_wing = af.Beam(name='right_outer_wing', mesh=right_outer_wing_mesh, material=aluminum, cs=outer_wing_cs)
right_outer_wing.add_load(right_outer_wing_loads)

# create the fwd fuse
fwd_fuse = af.Beam(name='fwd_fuse', mesh=fwd_fuse_mesh, material=aluminum, cs=fwd_fuse_cs)
fwd_fuse.add_load(fwd_fuse_loads)

# create the aft fuse
aft_fuse = af.Beam(name='aft_fuse', mesh=aft_fuse_mesh, material=aluminum, cs=aft_fuse_cs)

# create the htail
htail = af.Beam(name='htail', mesh=htail_mesh, material=aluminum, cs=htail_cs)
htail.add_load(htail_loads)

# create the vtail
vtail = af.Beam(name='vtail', mesh=vtail_mesh, material=aluminum, cs=vtail_cs)



# instantiate the frame model and add all beams and joints
frame = af.Frame()
frame.add_beam(center_wing)
frame.add_beam(right_outer_wing)
frame.add_beam(left_outer_wing)
frame.add_beam(fwd_fuse)
frame.add_beam(aft_fuse)
frame.add_beam(htail)
frame.add_beam(vtail)

frame.add_joint(joint_beams=[center_wing, right_outer_wing], joint_nodes=[num_center_wing - 1, 0])
frame.add_joint(joint_beams=[center_wing, left_outer_wing], joint_nodes=[0, 0])
frame.add_joint(joint_beams=[center_wing, fwd_fuse], joint_nodes=[5, num_fwd_fuse - 1])
frame.add_joint(joint_beams=[center_wing, aft_fuse], joint_nodes=[5, 0])
frame.add_joint(joint_beams=[aft_fuse, vtail], joint_nodes=[num_aft_fuse - 1, 0])
frame.add_joint(joint_beams=[vtail, htail], joint_nodes=[num_vtail - 1, 2])


# evaluating the frame model returns a solution dataclass
solution = frame.evaluate()

# finish up
recorder.stop()


start = 0
stop = 0.1
nt = 200
sim = af.Simulation(solution, start, stop, nt)
t, u = sim.solve()
center_wing_def_mesh = sim.parse_u(u, center_wing)
left_outer_wing_def_mesh = sim.parse_u(u, left_outer_wing)
right_outer_wing_def_mesh = sim.parse_u(u, right_outer_wing)
fwd_fuse_def_mesh = sim.parse_u(u, fwd_fuse)
aft_fuse_def_mesh = sim.parse_u(u, aft_fuse)
htail_def_mesh = sim.parse_u(u, htail)
vtail_def_mesh = sim.parse_u(u, vtail)

sim.create_frames_3d([center_wing_def_mesh,left_outer_wing_def_mesh,right_outer_wing_def_mesh,fwd_fuse_def_mesh,aft_fuse_def_mesh,htail_def_mesh,vtail_def_mesh],
                       figsize=(9, 8),
                       dpi=300)
sim.gif(filename='a_gusty_zephyr.gif', fps=25)
