import aframe as af
import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP

recorder = csdl.Recorder(inline=True)
recorder.start()

n = 21
mesh = np.zeros((n, 3))
mesh[:, 1] = np.linspace(0, 10, n)
mesh = csdl.Variable(value=mesh)

aluminum = af.Material(E=70e9, G=26e9, density=2700)

radius = csdl.Variable(value=np.ones(n - 1) * 0.5)
radius.set_as_design_variable(lower=0.1, scaler=1E1)
thickness = csdl.Variable(value=np.ones(n - 1) * 0.001)
cs = af.CSTube(radius=radius, thickness=thickness)

loads = np.zeros((n, 6))
loads[:, 2] = 20000
loads = csdl.Variable(value=loads)

beam = af.Beam(name='beam_1', mesh=mesh, material=aluminum, cs=cs)
beam.fix(0)
beam.add_load(loads)

frame = af.Frame()
frame.add_beam(beam)

acc = csdl.Variable(value=np.array([0, 0, -9.81, 0, 0, 0]))
frame.add_acc(acc)

frame.solve()
disp = frame.displacement['beam_1']
disp.set_as_constraint(upper=0.5)


mass = frame.mass
mass.set_as_objective(scaler=1)

recorder.stop()
# recorder.visualize_graph()
# print(disp.value)






sim = csdl.experimental.JaxSimulator(recorder=recorder)
sim.run()
prob = CSDLAlphaProblem(problem_name='single_beam', simulator=sim)


optimizer = SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-6, 'disp': True})
optimizer.solve()
optimizer.print_results()


print(radius.value)
print(disp.value)