import csdl_alpha as csdl
import numpy as np
import aframe as af


# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam mesh
num_nodes = 21
wing_mesh = np.zeros((num_nodes, 3))
wing_mesh[:, 1] = np.linspace(-20, 20, num_nodes)
wing_mesh = csdl.Variable(value=wing_mesh)

# create a material
aluminum = af.Material(name='aluminum', E=69E9, G=26E9, rho=2700, v=0.33)

# create cs properties

# create a beam
wing = af.Beam(name='wing', mesh=wing_mesh, material=aluminum, cs='tube')

# create a boundary condition
wing.AddBoundaryCondition(node=10)

# create a load
wing_loads = np.zeros((num_nodes, 6))
wing_loads[:, 2] = 20000
wing_loads = csdl.Variable(value=wing_loads)

wing.add_load(wing_loads)

# wing_forces = af.Load(name='wing_forces', beam=wing, value=wing_forces)

# instantiate the beam model and construct the F and K matrices
frame = af.Frame(beams=[wing], joints=[])
# or
frame = af.Frame()
frame.add(wing)
frame.add(joint_1)

# evaluate the beam model
wing_displacment = frame.evaluate(beam=wing, quantity='displacement')
wing_stress = frame.evaluate(beam=wing, quantity='stress')


# finish up
recorder.stop()
print(wing_stress.value)
